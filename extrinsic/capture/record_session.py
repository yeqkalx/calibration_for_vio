"""相机–IMU 采集和外参标定。PyCharm 直接运行此文件，不需要命令行参数。

它做五件事：
  1) 体检：相机在不在、飞控串口在不在、标定板配置和内参分辨率对不对
  2) 先起 IMU 录制线程，再拉起相机采集子进程（时间戳统一到宿主机 CLOCK_REALTIME）
  3) 相机跑完 → 收 IMU → 写 imu.csv
  4) capture_manifest.json → images.csv
  5) 打印采集体检；标定模式下自动用两份 CSV 求解外参

所有可调参数都在 extrinsics_config.py。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extrinsics_config as cfg                    # noqa: E402
from capture import image_health                   # noqa: E402
from capture import imu_recorder                   # noqa: E402

CAMERA_SCRIPT = HERE / "capture_calibration.py"


# --------------------------------------------------------------------------
# capture_manifest.json -> images.csv
# --------------------------------------------------------------------------

class ManifestError(RuntimeError):
    pass


def load_manifest(session_dir):
    path = Path(session_dir) / "capture_manifest.json"
    if not path.is_file():
        raise ManifestError(
            f"没找到 {path}。相机那一步没跑成功？"
            "（这个文件由 capture/capture_calibration.py 在采集结束时写出）")
    data = json.loads(path.read_text(encoding="utf-8"))
    saved = data.get("saved") or []
    if not saved:
        raise ManifestError(f"{path} 里一张图都没有 —— 采集是不是一开始就被中断了？")
    return data, saved


def build_images_csv(session_dir, out_name="images.csv", log=print):
    """写出 images.csv，返回统计 dict。"""
    session_dir = Path(session_dir)
    data, saved = load_manifest(session_dir)

    rows, missing = [], []
    for entry in saved:
        stamp = int(entry["host_receive_time_ns"])
        name = entry["file"]
        if not (session_dir / name).is_file():
            missing.append(name)
        rows.append((stamp, name))

    if missing:
        raise ManifestError(
            f"manifest 里列了 {len(missing)} 张图，但磁盘上不存在，例如 {missing[:3]}。"
            "别继续 —— 这种数据标出来的外参没有意义。")

    stamps = [s for s, _ in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ManifestError("图像时间戳不是严格递增的 —— manifest 被改过，或者文件名时间戳重复。")

    out = session_dir / out_name
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "filename"])
        writer.writerows(rows)

    gaps_ms = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
    gaps_sorted = sorted(gaps_ms)
    span = (stamps[-1] - stamps[0]) / 1e9
    stats = {
        "images": len(rows),
        "span_s": span,
        "observed_fps": (len(rows) - 1) / span if span > 0 else 0.0,
        "median_gap_ms": gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else 0.0,
        "max_gap_ms": max(gaps_ms) if gaps_ms else 0.0,
        "gaps_over_100ms": sum(1 for g in gaps_ms if g > 100.0),
        "gaps_over_350ms": sum(1 for g in gaps_ms if g > 350.0),
        "board_detected": sum(1 for e in saved if e.get("board_detected")),
        "csv": str(out),
        "manifest_mode": (data.get("mode") or {}).get("width", "?"),
        "manifest_resolution": "%sx%s" % ((data.get("mode") or {}).get("width", "?"),
                                          (data.get("mode") or {}).get("height", "?")),
        "manifest_termination": data.get("termination"),
    }
    log(f"写出 {out} —— {stats['images']} 张，跨度 {span:.2f}s，实测 {stats['observed_fps']:.1f} fps")
    return stats


make_images_csv = SimpleNamespace(
    ManifestError=ManifestError,
    load_manifest=load_manifest,
    build=build_images_csv,
)


def auto_calibration_enabled():
    return cfg.CAPTURE_PURPOSE == "calibration" and cfg.RUN_CALIBRATION_AFTER


def hr(title=""):
    print("=" * 76)
    if title:
        print(title)
        print("=" * 76)


# --------------------------------------------------------------------------
# 体检
# --------------------------------------------------------------------------

def scan_cameras(expected="2bc5:0511"):
    """扫描 /dev/video*，返回 USB 身份匹配且是采集节点（index=0）的设备。"""
    vendor, product = expected.lower().split(":")
    found, seen = [], []
    for node in sorted(Path("/dev").glob("video*")):
        info = Path("/sys/class/video4linux") / node.name
        try:
            name = (info / "name").read_text().strip()
            index = (info / "index").read_text().strip()
            sysdev = (info / "device").resolve()
        except OSError:
            continue
        ids = None
        for directory in (sysdev, *sysdev.parents):
            if (directory / "idVendor").is_file():
                ids = ((directory / "idVendor").read_text().strip().lower(),
                       (directory / "idProduct").read_text().strip().lower())
                break
        seen.append(f"{node}  {name}  {ids}  index={index}")
        if ids == (vendor, product) and index == "0":
            found.append((node, name))
    return found, seen


def usb_devices(ids="2bc5:0511"):
    """按 USB 身份在 sysfs 里找设备（不看 /dev/video*）。"""
    vendor, product = ids.lower().split(":")
    hits = []
    for d in sorted(Path("/sys/bus/usb/devices").glob("*")):
        if ":" in d.name:
            continue
        try:
            v = (d / "idVendor").read_text().strip().lower()
            p = (d / "idProduct").read_text().strip().lower()
        except OSError:
            continue
        if (v, p) == (vendor, product):
            hits.append(d.name)
    return hits


def read_target(path):
    """读标定板配置。两种板子的字段语义不同，所以返回 dict 而不是裸元组。"""
    from calib.data import read_yaml
    entry = read_yaml(path)
    kind = entry.get("target_type")
    if kind in ("checkerboard", "chessboard"):
        cols, rows = int(entry.get("cols", 0)), int(entry.get("rows", 0))
        size = float(entry.get("square_size", 0))
        span = ((cols - 1) * size, (rows - 1) * size)
        label = (f"棋盘 {cols}x{rows} 内角点，格边 {size * 100:g} cm，"
                 f"角点跨度 {span[0] * 100:.1f}x{span[1] * 100:.1f} cm")
    else:
        cols, rows = int(entry.get("tagCols", 0)), int(entry.get("tagRows", 0))
        size = float(entry.get("tagSize", 0))
        spacing = float(entry.get("tagSpacing", 0.0))
        pitch = size * (1 + spacing)
        span = ((cols - 1) * pitch + size, (rows - 1) * pitch + size)
        label = (f"AprilGrid {cols}x{rows} tag，tagSize {size * 100:g} cm，"
                 f"整板 {span[0] * 100:.1f}x{span[1] * 100:.1f} cm")
    return {"kind": kind, "cols": cols, "rows": rows, "size": size,
            "board_m": span, "label": label, "path": str(path)}


def preflight():
    """体检。通过则返回 (相机设备, 串口, 标定板信息)；不通过抛异常。"""
    print(f"会话           {cfg.SESSION_NAME}")
    print(f"会话目录       {cfg.SESSION_DIR}")
    print(f"结果目录       {cfg.RESULTS_DIR}")
    print(f"图像           {cfg.RESOLUTION}（{cfg.WIDTH}x{cfg.HEIGHT} @ {cfg.FPS} fps）")
    print(f"相机内参       {cfg.CAMERA_YAML}")
    print(f"标定板         {cfg.TARGET_YAML}")
    print(f"采集时长       {cfg.DURATION_S:g} s"
          f"（IMU 前后各多录 {cfg.IMU_PRIME_S:g} / {cfg.IMU_TAIL_S:g} s）")
    print()

    print("[1/4] 找相机……")
    if not CAMERA_SCRIPT.is_file():
        raise RuntimeError(f"缺少相机采集脚本 {CAMERA_SCRIPT}")
    cameras, seen = scan_cameras(cfg.CAMERA_USB_ID)
    if not cameras:
        print(f"  ❌ 没找到 USB 身份为 {cfg.CAMERA_USB_ID} 的相机；"
              f"/dev/video* 上现在挂着的是：")
        for line in (seen or ["     （一个都没有）"]):
            print(f"     {line}")
        on_bus = usb_devices(cfg.CAMERA_USB_ID)
        if on_bus:
            node = on_bus[0]
            print()
            print(f"  ⚠️  但它在 USB 总线上是在的：{', '.join(on_bus)}")
            print("     => 设备没掉线，是内核没给它建 /dev/video 节点（uvcvideo 未绑定）。")
            print("     恢复（任选一个，a 最稳）：")
            print("       a) 把相机的 USB 线拔掉、等 3 秒、插回去")
            print(f"       b) sudo sh -c 'echo 0 > /sys/bus/usb/devices/{node}/authorized;"
                  f" sleep 1; echo 1 > /sys/bus/usb/devices/{node}/authorized'")
            print("       c) sudo modprobe -r uvcvideo && sudo modprobe uvcvideo")
            raise RuntimeError("相机在总线上但没有 /dev/video 节点（uvcvideo 没绑定回去）。")
        raise RuntimeError("相机没插或没被识别：USB 数据线插好了吗？")
    device, name = cameras[0]
    print(f"  ✅ {device}  ({name})")

    print("[2/4] 找飞控串口……")
    port = cfg.IMU_PORT
    if port == "auto":
        port, report = imu_recorder.find_mavlink_port(baud=cfg.IMU_BAUD)
        if port is None:
            for k, v in report.items():
                print(f"     {k}: {v}")
            raise RuntimeError("没找到在讲 MAVLink 的串口：飞控通电了吗？USB 线是数据线吗？"
                               "或在 extrinsics_config.py 里把 IMU_PORT 写死成 /dev/ttyACM0 试试。")
    else:
        print(f"  使用配置里写死的 {port}")
    print(f"  ✅ {port}")

    print("[3/4] 检查内参与标定板……")
    if not Path(cfg.CAMERA_YAML).is_file():
        raise RuntimeError(f"内参文件不存在：{cfg.CAMERA_YAML}")
    from calib.data import Camera, Target, read_yaml
    from extrinsics_config import SETTINGS
    cam = Camera.load(cfg.CAMERA_YAML)
    if tuple(cam.size) != (cfg.WIDTH, cfg.HEIGHT):
        raise RuntimeError(
            f"内参写的分辨率 {cam.size} 与本次采集的 {cfg.WIDTH}x{cfg.HEIGHT} 不一致。"
            "改 extrinsics_config.py 的 RESOLUTION，或换一份匹配的内参 —— 工具不会自动缩放。")
    print(f"  ✅ 内参 {cam.size}，fx={cam.params[0]:.1f}")

    if not Path(cfg.TARGET_YAML).is_file():
        raise RuntimeError(f"标定板文件不存在：{cfg.TARGET_YAML}")
    if cfg.CAPTURE_PURPOSE == "calibration":
        target = Target(read_yaml(cfg.TARGET_YAML), min_tags=SETTINGS.min_tags,
                        april_corner_window_px=SETTINGS.april_corner_window_px)
        if target.kind == "aprilgrid" and target.rows * target.cols < target.min_tags:
            raise RuntimeError(f"整块 AprilGrid 的 tag 总数少于每帧检出下限 {target.min_tags}。"
                               "请核对板子行列数及 extrinsics_config.py 的 min_tags。")
    board = read_target(cfg.TARGET_YAML)
    print(f"  ✅ 标定板 {board['label']}")
    if board["kind"] in ("checkerboard", "chessboard"):
        print("     ⚠️ 行列数必须和实物完全一致，否则检测出 0 帧。数的是【内角点】，不是格子数。")
    else:
        print("     ⚠️ tagRows/tagCols 是【tag 个数】不是内角点；tagSize 是黑框外沿，卡尺实测。")
        print(f"        按当前配置，整板 {board['board_m'][0] * 100:.1f}x"
              f"{board['board_m'][1] * 100:.1f} cm —— 和实物对不上就是 0 帧。")

    print("[4/4] 检查目录……")
    session_dir = Path(cfg.SESSION_DIR)
    if session_dir.exists() and any(session_dir.iterdir()):
        if auto_calibration_enabled() or not cfg.REUSE_EXISTING_SESSION:
            raise RuntimeError(
                f"采集目录 {session_dir} 非空。请在 extrinsics_config.py 改用新的 SESSION_NAME。")
        print("  ⚠️ 复用已有目录（REUSE_EXISTING_SESSION=True）")
    results_dir = Path(cfg.RESULTS_DIR)
    if results_dir.exists() and any(results_dir.iterdir()):
        if auto_calibration_enabled():
            raise RuntimeError(
                f"标定结果目录 {results_dir} 非空。请在 extrinsics_config.py 改用新的 SESSION_NAME。")
        print(f"  ⚠️ 结果目录 {results_dir} 非空 —— 标定程序会拒绝往里写，"
              "换个 SESSION_NAME 或把旧结果移走。")
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {session_dir}")

    return device, port, board


# --------------------------------------------------------------------------
# 采集
# --------------------------------------------------------------------------

def camera_exposure_args():
    from calib.data import Camera

    calibration = cfg.CAPTURE_PURPOSE == "calibration"
    expected_rate = (cfg.CALIBRATION_AUTO_EXPOSE_EXPECTED_RATE_DPS
                     if calibration else cfg.AUTO_EXPOSE_EXPECTED_RATE_DPS)
    focal_px = float(Camera.load(cfg.CAMERA_YAML).params[0])
    args = ["--auto-expose" if cfg.AUTO_EXPOSE else "--no-auto-expose",
            "--auto-expose-max-exposure-ms", str(cfg.AUTO_EXPOSE_MAX_EXPOSURE_MS),
            "--auto-expose-target", str(cfg.AUTO_EXPOSE_TARGET),
            "--auto-expose-min-mean", str(cfg.AUTO_EXPOSE_MIN_MEAN),
            "--auto-expose-blur-budget-px", str(cfg.AUTO_EXPOSE_BLUR_BUDGET_PX),
            "--auto-expose-expected-rate-dps", str(expected_rate),
            "--focal-px", str(focal_px),
            "--auto-expose-max-gain", str(cfg.AUTO_EXPOSE_MAX_GAIN)]
    if calibration:
        args.append("--allow-below-exposure-target")
    if cfg.ALLOW_DARK:
        args.append("--allow-dark")
    if cfg.SKIP_EXPOSURE_CHECK:
        args.append("--skip-exposure-check")
    return args


def camera_command(device, target_info):
    cols, rows = target_info["cols"], target_info["rows"]
    return [
        sys.executable, str(CAMERA_SCRIPT),
        "--device", str(device),
        "--output", str(cfg.SESSION_DIR),
        "--width", str(cfg.WIDTH), "--height", str(cfg.HEIGHT), "--fps", str(cfg.FPS),
        "--exposure", str(cfg.EXPOSURE),
        "--white-balance", str(cfg.WHITE_BALANCE),
        "--gain", str(cfg.GAIN),
        "--brightness", str(cfg.BRIGHTNESS),
        "--contrast", str(cfg.CONTRAST),
        "--sharpness", str(cfg.SHARPNESS),
        "--power-line-frequency", str(cfg.POWER_LINE_FREQUENCY),
        "--backlight-compensation", str(cfg.BACKLIGHT_COMPENSATION),
        "--cols", str(cols), "--rows", str(rows),
        "--auto-interval", f"{0.5 / cfg.FPS:.6f}",
        "--max-images", "0",
        "--duration", str(cfg.DURATION_S),
    ] + camera_exposure_args() + ([] if cfg.SHOW_PREVIEW else ["--no-preview"])


def _pump_with_timer(proc):
    """边采集边报时：每秒一行"已过几秒 / 还剩几秒"。"""
    import queue
    import threading

    lines = queue.Queue()

    def _reader():
        try:
            for ln in proc.stdout:
                lines.put(ln.rstrip("\n"))
        finally:
            lines.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    t0 = time.monotonic()
    saved = 0
    last_beat = -1.0
    last_qual = -99.0
    beeped = set()
    duration = float(cfg.DURATION_S)
    width = 78
    latest = None
    qual = None
    try:
        import cv2
        _fast = cv2.FastFeatureDetector.create(20, True)
    except Exception:                                          # noqa: BLE001
        cv2, _fast = None, None

    def _clear():
        print("\r" + " " * width + "\r", end="")

    while True:
        drained = False
        try:
            ln = lines.get(timeout=0.1)
            if ln is None:
                drained = True
            elif ln.startswith("保存 "):
                saved += 1
                parts = ln.split()
                if len(parts) >= 2:
                    latest = parts[1]
            elif ln.strip() == "@记录开始@":
                t0 = time.monotonic()
                last_beat = -1.0
                beeped.clear()
                _clear()
                print("   ▶ 曝光已锁定并验证，计时从现在开始")
            elif ln.strip():
                _clear()
                print(f"   {ln}")
        except queue.Empty:
            pass

        now = time.monotonic()
        elapsed = now - t0
        remain = duration - elapsed
        if cv2 is not None and latest and now - last_qual >= 2.0:
            last_qual = now
            img = cv2.imread(latest, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                nf = len(_fast.detect(img, None))
                qual = (float(img.mean()), nf)

        if now - last_beat >= 1.0:
            last_beat = now
            tail = f" | 还剩 {remain:4.1f}s" if remain >= 0 else " | 超时了"
            if qual is not None:
                mean_v, nf = qual
                mark = "OK" if (mean_v >= 40 and nf >= 400) else "!!"
                tail += f" | 亮度 {mean_v:3.0f} 特征 {nf:4d} {mark}"
            print(f"\r  [{elapsed:5.1f}s / {duration:g}s]  存图 {saved:4d}{tail}",
                  end="", flush=True)

        sec = int(round(remain))
        if 0 <= sec <= 10 and sec not in beeped:
            beeped.add(sec)
            n = 3 if sec == 0 else 1
            sys.stdout.write("\a" * n)
            sys.stdout.flush()

        if drained:
            break

    try:
        proc.wait(timeout=5)
    except Exception:                                          # noqa: BLE001
        pass
    _clear()
    print(f"  采集结束：{saved} 张图，用时 {time.monotonic() - t0:.1f} s")
    if cfg.CAPTURE_PURPOSE == "vio":
        print("  请确认你此刻正站在起点地贴上 —— 闭环要靠这个")


def record(device, port, target_info):
    """录一段。返回 (imu_stats, 相机返回码, 是否被中断)。imu.csv 在这里写。"""
    print()
    hr("开始采集")
    print(f"  相机 {device}   |   IMU {port}   |   {cfg.DURATION_S:g} 秒")
    if cfg.SHOW_PREVIEW:
        print("  预览窗口：空格手动存一张，Q 提前结束")
    if cfg.CAPTURE_PURPOSE == "calibration":
        print("  让标定板保持可见，缓慢绕三轴转动整套设备，并做一些平移。")
    if cfg.CAPTURE_PURPOSE == "calibration" and target_info.get("kind") == "aprilgrid":
        print("  注意：下行每句末尾的 board=no 是采图脚本自带的【棋盘】检测器打的，"
              "对 AprilGrid 恒为 no，可以忽略")
    print()

    recorder = imu_recorder.ImuRecorder(port, request_hz=cfg.IMU_REQUEST_HZ, baud=cfg.IMU_BAUD,
                                        startup_discard_s=cfg.IMU_STARTUP_DISCARD_S)
    recorder.start()
    print(f"  IMU 已在录，先跑 {cfg.IMU_PRIME_S:g} 秒保证两路有重叠……")
    proc, interrupted = None, False
    try:
        time.sleep(cfg.IMU_PRIME_S)
        proc = subprocess.Popen(camera_command(device, target_info),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        _pump_with_timer(proc)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  收到中断，正在收尾……")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        if not interrupted:
            time.sleep(cfg.IMU_TAIL_S)
        imu_stats = recorder.stop()
        csv_path = recorder.write_csv(Path(cfg.SESSION_DIR) / "imu.csv")
        imu_stats["csv"] = str(csv_path)

    return imu_stats, (proc.returncode if proc is not None else None), interrupted


# --------------------------------------------------------------------------
# 报告与自动求解
# --------------------------------------------------------------------------

def report(imu_stats, cam_rc, img_stats, interrupted):
    hr("采集结果")
    print("IMU")
    print(imu_recorder.format_stats(imu_stats))
    imu_level, imu_text = imu_recorder.verdict(imu_stats)
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[imu_level]
    print(f"  {icon} {imu_text}")

    print()
    print("图像")
    if img_stats is None:
        print("  ❌ 没有生成 images.csv（相机那一步没成功）")
    else:
        print(f"  张数 {img_stats['images']}，跨度 {img_stats['span_s']:.2f} s，"
              f"实测 {img_stats['observed_fps']:.1f} fps")
        print(f"  最大帧间隔 {img_stats['max_gap_ms']:.1f} ms"
              f"（>100ms: {img_stats['gaps_over_100ms']}, >350ms: {img_stats['gaps_over_350ms']}）")
        from calib.data import read_yaml
        if read_yaml(cfg.TARGET_YAML).get("target_type") in ("checkerboard", "chessboard"):
            print(f"  采集脚本自带的棋盘检出 {img_stats['board_detected']}/{img_stats['images']}"
                  "（仅供参考；标定程序用的是更鲁棒的检测器）")
        else:
            print("  采集脚本自带的棋盘检出  N/A（采图脚本只会找棋盘；AprilGrid 看下面的画面体检）")
        print(f"  结束状态 {img_stats['manifest_termination']}")
        manifest_path = Path(cfg.SESSION_DIR) / "capture_manifest.json"
        try:
            mm = json.loads(manifest_path.read_text(encoding="utf-8"))
            ae = mm.get("auto_exposure") or {}
            if ae.get("achieved_mean") is not None:
                print(f"  自动测光 {ae.get('status')}：曝光 {ae.get('exposure_ms')} ms，"
                      f"增益 {ae.get('gain')} → 灰度 {ae.get('achieved_mean')}"
                      f"（目标 {ae.get('target')}）")
                if ae.get("gain_range"):
                    print(f"           增益可用范围 {ae['gain_range']}，"
                          f"曝光范围 {ae['exposure_range']}")
                for row in (ae.get("ladder") or [])[-4:]:
                    note = f"  {row['note']}" if row.get("note") else ""
                    print(f"           {row.get('exposure', 0) * 0.1:5.1f} ms  "
                          f"gain {row.get('gain'):>4}  → 灰度 {row.get('mean')}{note}")
        except Exception:                                          # noqa: BLE001
            pass
        if str(img_stats["manifest_termination"] or "").startswith("failed"):
            print("  ❌ 相机采集没正常完成，这次数据不要用")

    slow = (img_stats is not None and img_stats["observed_fps"] < cfg.FPS * 0.6
            and img_stats["span_s"] > 2.0)
    if slow:
        print()
        print(f"  ⚠️ 实测 {img_stats['observed_fps']:.1f} fps，远低于请求的 {cfg.FPS} fps。")
        print("     按可能性排查：1) SHOW_PREVIEW 是否开启 2) RESOLUTION 可改 720p 3) CPU 负载过高")

    health = None
    health_level = None
    session_dir = Path(cfg.SESSION_DIR)
    images = sorted((session_dir / "cam0").glob("*.png"))
    board = read_target(cfg.TARGET_YAML) if Path(cfg.TARGET_YAML).is_file() else None
    try:
        pnp_gate = float(getattr(cfg.SETTINGS, "max_pnp_rms_px", 2.0))
        min_tags = int(getattr(cfg.SETTINGS, "min_tags", 4))
    except Exception:                                          # noqa: BLE001
        pnp_gate, min_tags = 2.0, 4
    purpose = str(getattr(cfg, "CAPTURE_PURPOSE", "calibration")).lower()
    if images and board is not None and purpose == "vio":
        b = image_health.brightness_stats(images, sample=20)
        print()
        print("画面体检（VIO 用途：画面里没有标定板，不做 tag 检出）")
        if b:
            print(f"  亮度            p50={b['p50']:.0f}  p90={b['p90']:.0f}"
                  f"  p99={b['p99']:.0f}  均值={b['mean']:.0f}")
            level, text = image_health.brightness_verdict(b)
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(level, "•")
            print(f"  {icon} {text}")
        print("  标定板检出      跳过（这份数据是给 VIO 用的，不是内参/外参标定）")
    elif images and board is not None and not auto_calibration_enabled():
        print()
        print("画面体检")
        health = image_health.analyze(images, cfg.TARGET_YAML, sample=20)
        try:
            health["quality"] = image_health.calib_quality_stats(
                images, cfg.CAMERA_YAML, cfg.TARGET_YAML, window=24,
                min_tags=min_tags, max_pnp_rms=pnp_gate)
        except Exception as exc:                               # noqa: BLE001
            print(f"  （标定质量抽检跳过：{exc}）")
        health_level, health_actions = image_health.report(health)
        for action in health_actions:
            print(f"     → {action}")
        print()
        print("逐帧复算（跑标定程序自己的检测+PnP，约 1~2 分钟）……")
        try:
            print(f"（按 extrinsics_config 的 max_pnp_rms_px={pnp_gate:g}、"
                  f"min_tags={min_tags} 复算）")
            tl = image_health.full_timeline(
                images, cfg.CAMERA_YAML, cfg.TARGET_YAML,
                max_pnp_rms=pnp_gate, min_tags=min_tags, log=lambda *_a: None)
            window_ok, _window_note = image_health.report_timeline(tl)
            health["timeline"] = tl
            health["window_ok"] = window_ok
        except Exception as exc:                               # noqa: BLE001
            print(f"  （逐帧复算跳过：{exc}）")

    print()
    hr("结论")
    fatal = interrupted
    if interrupted:
        print("⚠️  这次是被中断的，数据可能不完整")
    if cam_rc != 0:
        print(f"❌ 相机采集返回码 {cam_rc} —— 看上面相机子进程打印的报错")
        fatal = True
    if imu_level == "fail":
        print(f"❌ IMU：{imu_text}")
        fatal = True
    if img_stats is None:
        fatal = True
    elif str(img_stats.get("manifest_termination") or "").startswith(("failed", "user_interrupted")):
        print(f"❌ 相机采集没有正常完成：{img_stats['manifest_termination']}")
        fatal = True
    elif img_stats["images"] < 35:
        print(f"❌ 图像只有 {img_stats['images']} 张，低于标定下限（35 张有效检出）")
        fatal = True

    window_ok = bool(health and health.get("window_ok"))
    if health is not None and not window_ok:
        print("❌ 逐帧复算：没有一段【连续】≥5 秒 且 ≥35 帧的可用窗口 —— 直接重录。")
        fatal = True
    if health is not None and health_level == "fail":
        d = health["detection"]
        print(f"❌ 画面：{d['detected']}/{d['tried']} 检出（{100 * d['rate']:.0f}%）")
        print("   先点 capture/check_camera.py 把取景和曝光调对，别浪费一轮采集")
        fatal = True
    if not fatal and auto_calibration_enabled():
        print("✅ 两份 CSV 就绪；接下来由求解器逐帧检测标定板并检查残差与运动激励。")
    elif not fatal and purpose == "calibration":
        print("✅ 两份 CSV 就绪、画面过关、也有连续可用窗口 —— 可以往下标定了")
    elif not fatal:
        print("✅ VIO 采集的两份 CSV 就绪；不执行外参标定。")
    return fatal


def calibration_command():
    session_dir = Path(cfg.SESSION_DIR)
    return [sys.executable, str(ROOT / "calibrate_extrinsics.py"),
            "--camera", str(cfg.CAMERA_YAML), "--target", str(cfg.TARGET_YAML),
            "--images", str(session_dir / "images.csv"), "--imu", str(session_dir / "imu.csv"),
            "--image-unit", "ns", "--imu-unit", "ns", "--output", str(cfg.RESULTS_DIR)]


def run_calibration():
    hr("开始外参求解 · 不再占用相机和飞控串口")
    print(f"  使用标定板 {cfg.TARGET_YAML}", flush=True)
    result = subprocess.run(calibration_command(), cwd=ROOT, check=False)
    report_path = Path(cfg.RESULTS_DIR) / "report.json"
    if result.returncode:
        print(f"❌ 外参标定未通过（退出码 {result.returncode}）；原始采集数据已保留。")
        if report_path.is_file():
            print(f"   失败原因见 {report_path}")
    else:
        print(f"✅ 外参标定通过。结果和诊断见 {report_path}")
    return result.returncode


def main():
    hr("相机–IMU 外参标定 · 采集")
    try:
        device, port, target_info = preflight()
    except Exception as exc:                                   # noqa: BLE001
        print()
        hr("❌ 体检没通过，没有开始采集")
        print(f"{exc}")
        return 2

    try:
        imu_stats, cam_rc, interrupted = record(device, port, target_info)
    except (RuntimeError, OSError) as exc:
        print(f"❌ 采集失败：{exc}")
        return 1

    session_dir = Path(cfg.SESSION_DIR)
    img_stats = None
    if (session_dir / "capture_manifest.json").is_file():
        try:
            img_stats = make_images_csv.build(session_dir)
        except make_images_csv.ManifestError as exc:
            print(f"❌ images.csv 生成失败：{exc}")
    else:
        print(f"⚠️  {session_dir}/capture_manifest.json 不存在 —— 相机一张都没存下。")

    fatal = report(imu_stats, cam_rc, img_stats, interrupted)
    print()
    print(f"数据在 {session_dir}")
    if fatal:
        print("未开始外参标定。解决采集问题后，请换一个 SESSION_NAME 再录。")
        return 1
    if auto_calibration_enabled():
        return run_calibration()
    if cfg.CAPTURE_PURPOSE == "calibration":
        print("只采集模式。若要为本次会话单独求解，可在终端运行：")
        print(f"  {shlex.join(calibration_command())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
