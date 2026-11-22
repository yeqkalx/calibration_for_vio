#!/usr/bin/env python3
"""★ 相机取景 / 曝光 / 标定板一体化体检与实时预览（不需要飞控 IMU）

将原先分散在 framing_check.py、check_board.py、preview_aprilgrid.py 的功能整合为一处：
  1. 默认模式（PyCharm 直接点运行 / `python capture/check_camera.py`）：
     录制 10 秒图像到 data/_framing_check/，自动完成：
       · 亮度判读（p90 白格亮度是否落在可用区间）
       · 标定板检出率与画面边距/切边诊断
       · 逐帧平面 PnP 几何残差检查
       · 若为 AprilGrid，自动扫描反解最优 tagSpacing 与 tag_rotation_deg
  2. 已有图片体检（`python capture/check_camera.py --dir data/session01/cam0`）：
     不开相机，直接对指定目录下的图片做上述亮度 + 检出 + PnP + 板面参数反解。
  3. 实时预览模式（`python capture/check_camera.py --preview`）：
     打开实时 HUD 窗口（或 `--headless` 终端报数），画框显示检出角点/tag 数、尺寸与 PnP 残差。
  4. 离线自检（`python capture/check_camera.py --self-test`）：
     无需相机硬件，合成 AprilGrid 验证检测、PnP 与 HUD 叠加。
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extrinsics_config as cfg                    # noqa: E402
from capture import image_health                   # noqa: E402
from capture import record_session as rs           # noqa: E402

CHECK_DIR = ROOT / "data" / "_framing_check"
SAVE_DIR = ROOT / "data" / "_preview"
SECONDS = 10.0
DISPLAY_WIDTH = 880

# ---- 最小 V4L2 控件封装（实时预览锁定/恢复用） ----
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
V4L2_CTRL_FMT = "Ii"
CIDS = {
    "exposure_auto": 0x009A0901,
    "exposure": 0x009A0902,
    "auto_white_balance": 0x0098090C,
    "white_balance_temperature": 0x0098091A,
    "gain": 0x00980913,
    "brightness": 0x00980900,
    "contrast": 0x00980901,
    "sharpness": 0x0098091B,
    "power_line_frequency": 0x00980918,
    "backlight_compensation": 0x0098091C,
    "gamma": 0x00980910,
}
_RESTORE_ORDER = ["exposure", "white_balance_temperature", "gain", "brightness",
                  "contrast", "sharpness", "backlight_compensation", "gamma",
                  "power_line_frequency", "exposure_auto", "auto_white_balance"]


def _draw_aruco_marker(dictionary, tag_id, size_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, size_px)
    return cv2.aruco.drawMarker(dictionary, tag_id, size_px)


def set_ctrl(fd, cid, value):
    fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack(V4L2_CTRL_FMT, cid, int(value)))


def get_ctrl(fd, cid):
    raw = fcntl.ioctl(fd, VIDIOC_G_CTRL, struct.pack(V4L2_CTRL_FMT, cid, 0))
    return struct.unpack(V4L2_CTRL_FMT, raw)[1]


def lock_controls(device):
    """锁定影响亮度的 V4L2 控件，返回 (fd, 原值, 不支持列表)。"""
    try:
        fd = os.open(str(device), os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        print(f"  ⚠️ 打不开 {device} 调 V4L2 控件：{exc}")
        return None, None, []

    previous, unsupported = {}, []
    for name in CIDS:
        try:
            previous[name] = get_ctrl(fd, CIDS[name])
        except OSError:
            unsupported.append(name)

    wanted = [("exposure_auto", 1), ("exposure", int(cfg.EXPOSURE)),
              ("auto_white_balance", 0),
              ("white_balance_temperature", int(cfg.WHITE_BALANCE)),
              ("gain", int(cfg.GAIN)),
              ("brightness", int(getattr(cfg, "BRIGHTNESS", 0))),
              ("contrast", int(getattr(cfg, "CONTRAST", 32))),
              ("sharpness", int(getattr(cfg, "SHARPNESS", 3))),
              ("backlight_compensation", int(getattr(cfg, "BACKLIGHT_COMPENSATION", 0))),
              ("power_line_frequency", int(getattr(cfg, "POWER_LINE_FREQUENCY", 1)))]
    wanted = [(name, value) for name, value in wanted if name in previous]
    if "gamma" in previous:
        wanted.append(("gamma", previous["gamma"]))

    failed = 0
    try:
        for name, value in wanted:
            try:
                set_ctrl(fd, CIDS[name], value)
                actual = get_ctrl(fd, CIDS[name])
                if actual != value:
                    failed += 1
                    print(f"  ⚠️ {name} 想要 {value}，实际 {actual}")
            except OSError as exc:
                failed += 1
                print(f"  ⚠️ {name}={value} 设置失败：{exc}")
        if not failed:
            print(f"  控件已锁：曝光 {cfg.EXPOSURE}（100µs）, 白平衡 {cfg.WHITE_BALANCE}K, 增益 {cfg.GAIN}")
    except BaseException:
        restore_controls(fd, previous)
        raise
    return fd, previous, unsupported


def restore_controls(fd, previous):
    if fd is None:
        return
    try:
        if previous:
            for name in _RESTORE_ORDER:
                if name not in previous:
                    continue
                try:
                    set_ctrl(fd, CIDS[name], previous[name])
                except OSError:
                    pass
    finally:
        os.close(fd)


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {device}；请关闭其他占用相机的程序后重试")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    got = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        cap.release()
        raise RuntimeError(f"相机实际分辨率 {got}，与请求的 {width}x{height} 不符")
    return cap


# --------------------------------------------------------------------------
# 标定板检测、PnP 残差与参数反解
# --------------------------------------------------------------------------

def images_of(path: Path):
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return sorted(path.parent.glob(path.name))


def make_target(entry, min_tags, window_px):
    from calib.data import Target
    return Target(entry, min_tags=min_tags, april_corner_window_px=window_px)


def pnp_rms(target, camera, gray, stamp=0):
    """返回 (单元数, 每tag边长中位|None, PnP残差) 或 None。"""
    from calib.data import make_observation
    result = target.detect(gray)
    if result is None:
        return None
    xyz, uv = result
    checker = target.kind in ("checkerboard", "chessboard")
    obs = make_observation(stamp, xyz, uv, camera, None, checker=checker, max_points=0)
    if obs is None:
        return None
    tag_px = None
    if target.kind == "aprilgrid":
        quads = np.asarray(uv, float).reshape(-1, 4, 2)
        tag_px = float(np.median(np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)))
        count = len(xyz) // 4
    else:
        count = len(xyz)
    return count, tag_px, float(obs.pnp_rms)


def scan_entry(entry, camera, grays, min_tags, window_px, key, values):
    """把 entry[key] 逐个换成 values 试，返回 (最优值, 中位残差, 表)。"""
    rows, best = [], None
    for v in values:
        trial = dict(entry, **{key: v})
        try:
            target = make_target(trial, min_tags, window_px)
        except Exception:                                  # noqa: BLE001
            continue
        rms = [r for r in (pnp_rms(target, camera, g, i) for i, g in enumerate(grays)) if r]
        if len(rms) < 2:
            continue
        med = float(np.median([r[2] for r in rms]))
        rows.append((v, med, len(rms)))
        if best is None or med < best[1]:
            best = (v, med, len(rms))
    return best, rows


def inspect_directory(image_dir: Path, camera_yaml: Path, target_yaml: Path,
                      sample: int = 16, max_pnp_px: float | None = None) -> int:
    """对目录中的图像做亮度 + 检出 + 逐帧 PnP + AprilGrid 参数反解。"""
    from calib.data import Camera, read_yaml

    if max_pnp_px is None:
        max_pnp_px = float(getattr(cfg.SETTINGS, "max_pnp_rms_px", 2.0))

    files = images_of(image_dir)
    if not files:
        print(f"❌ {image_dir} 里没有图片。")
        return 2

    res = image_health.analyze([str(f) for f in files], target_yaml, sample=min(len(files), max(sample, 20)))
    level, actions = image_health.report(res)
    print()
    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "❔"}[level]
    print(f"{icon} 画面与检出判定：{level}")
    for a in actions:
        print(f"   → {a}")

    if len(files) > sample:
        idx = np.linspace(0, len(files) - 1, sample).astype(int)
        sampled_files = [files[i] for i in idx]
    else:
        sampled_files = files
    grays = [g for g in (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sampled_files) if g is not None]
    if not grays:
        return 2

    entry = read_yaml(target_yaml)
    kind = entry.get("target_type")
    camera = Camera.load(camera_yaml)
    d = res["detection"]
    min_tags = d.get("min_tags") or int(getattr(cfg.SETTINGS, "min_tags", 4))
    window_px = int(getattr(cfg.SETTINGS, "april_corner_window_px", 4))

    print()
    print("=" * 76)
    print("  逐帧几何与 PnP 残差检查")
    print("=" * 76)
    target = make_target(entry, min_tags, window_px)
    rms_list = []
    unit_name = "tag" if kind == "aprilgrid" else "角点"
    for i, g in enumerate(grays):
        r = pnp_rms(target, camera, g, i)
        if r is None:
            print(f"  帧 {i:>2}  检出失败")
            continue
        n_unit, tag_px, rms = r
        rms_list.append(rms)
        extra = f"tag {tag_px:5.0f} px" if tag_px else ""
        flag = "" if rms <= max_pnp_px else "  ← 残差偏大，检查模糊或板面参数"
        print(f"  帧 {i:>2}  {n_unit:>3} {unit_name}  {extra:>12}  PnP {rms:6.2f} px{flag}")

    if not rms_list:
        print("\n❌ 没有一帧解出有效几何。请先根据上方提示调整补光、距离或标定板行列数。")
        return 1

    med = float(np.median(rms_list))
    print(f"\n  PnP 残差中位数：{med:.2f} px（门槛 {max_pnp_px:g} px）")

    if kind == "aprilgrid" and (med > max_pnp_px or len(rms_list) >= 2):
        best, _rows = scan_entry(entry, camera, grays, min_tags, window_px,
                                 "tagSpacing", [round(0.1 + 0.01 * i, 2) for i in range(0, 71)])
        best_rot, _rot_rows = scan_entry(entry, camera, grays, min_tags, window_px,
                                         "tag_rotation_deg", [0, 90, 180, 270])
        if best and best_rot:
            cur_sp = float(entry.get("tagSpacing", 0.3))
            cur_rot = int(entry.get("tag_rotation_deg", 180))
            if med > max_pnp_px or abs(best[0] - cur_sp) >= 0.03 or best_rot[0] != cur_rot:
                print("\n  [AprilGrid 参数反解建议]")
                print(f"    当前配置: tagSpacing={cur_sp:.2f}, tag_rotation_deg={cur_rot}")
                print(f"    最优反解: tagSpacing={best[0]:.2f} (PnP {best[1]:.2f} px), "
                      f"tag_rotation_deg={best_rot[0]} (PnP {best_rot[1]:.2f} px)")

    ok = (level != "fail") and (med <= max_pnp_px)
    print()
    if ok:
        print("✅ 取景、亮度与标定板几何均正常，可以运行 capture/record_session.py 采集正式数据。")
        return 0
    print("❌ 体检未通过，请按上方提示调整后重新运行 capture/check_camera.py。")
    return 1


# --------------------------------------------------------------------------
# 实时预览与合成自检
# --------------------------------------------------------------------------

def analyse(target, camera, gray, max_points=0):
    """在全分辨率灰度图上检测 + 解平面位姿。返回 (uv|None, 每tag边长px|None, PnP残差|None)。"""
    from calib.data import make_observation
    result = target.detect(gray)
    if result is None:
        return None, None, None
    uv = np.asarray(result[1], float)
    checker = target.kind in ("checkerboard", "chessboard")
    tag_px = None
    if not checker:
        quads = uv.reshape(-1, 4, 2)
        tag_px = float(np.median(np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)))
    obs = make_observation(0, result[0], uv, camera, None, checker=checker,
                           max_points=max_points)
    return uv, tag_px, (None if obs is None else float(obs.pnp_rms))


def render(gray, uv, min_tags, tag_px, rms, fps, width=DISPLAY_WIDTH):
    """把全分辨率灰度图缩到 width 宽，叠上检测框和 HUD。返回 (BGR图, n_tag)。"""
    h, w = gray.shape[:2]
    k = width / w
    small = cv2.resize(gray, (width, round(h * k)), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    n_tag = 0 if uv is None else len(uv) // 4
    if uv is not None:
        colour = (0, 230, 0) if n_tag >= min_tags else (0, 200, 255)
        if len(uv) % 4 == 0:
            for quad in np.asarray(uv, float).reshape(-1, 4, 2) * k:
                pts = np.rint(quad).astype(int)
                cv2.polylines(canvas, [pts], True, colour, 2)
                cv2.circle(canvas, tuple(pts[0]), 3, (0, 0, 255), -1)
        else:
            for pt in np.asarray(uv, float).reshape(-1, 2) * k:
                cv2.circle(canvas, (round(pt[0]), round(pt[1])), 3, colour, -1)
    hud = [
        (f"tags {n_tag:2d} / need >={min_tags}", (0, 230, 0) if n_tag >= min_tags else (0, 200, 255)),
        (f"tag size {tag_px:5.1f} px  (want >=45)" if tag_px else "tag size    -  ",
         (0, 230, 0) if (tag_px and tag_px >= 45) else ((0, 0, 255) if tag_px else (200, 200, 200))),
        (f"PnP {rms:5.2f} px  (want <3)" if rms else "PnP    -   ",
         (0, 230, 0) if (rms and rms < 3.0) else ((0, 0, 255) if rms else (200, 200, 200))),
        (f"{fps:4.1f} fps   Q quit   S save", (200, 200, 200)),
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (420, 24 + 27 * len(hud)), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    for i, (text, colour) in enumerate(hud):
        y = 28 + 27 * i
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, 2)
    return canvas, n_tag


def synthetic_frame(cols=6, rows=6, tag_px=100, spacing=0.56):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gap = max(1, int(round(tag_px * spacing)))
    pitch = tag_px + gap
    board = np.full((rows * pitch - gap, cols * pitch - gap), 225, np.uint8)
    for y in range(rows):
        for x in range(cols):
            marker = _draw_aruco_marker(d, y * cols + x, tag_px)
            board[(rows - 1 - y) * pitch:(rows - 1 - y) * pitch + tag_px,
                  x * pitch:x * pitch + tag_px] = marker
    frame = np.full((1080, 1920), 170, np.uint8)
    y0, x0 = (1080 - board.shape[0]) // 2, (1920 - board.shape[1]) // 2
    frame[y0:y0 + board.shape[0], x0:x0 + board.shape[1]] = board
    return frame


def self_test(args):
    from calib.data import Camera, Target, read_yaml
    entry = dict(read_yaml(args.target))
    if entry.get("target_type") != "aprilgrid":
        entry = dict(read_yaml(ROOT / "configs" / "aprilgrid.yaml"))
    entry["tag_rotation_deg"] = 0
    target = Target(entry, min_tags=1)
    camera = Camera.load(args.camera)
    gray = synthetic_frame(int(entry["tagCols"]), int(entry["tagRows"]),
                           spacing=float(entry["tagSpacing"]))
    uv, tag_px, rms = analyse(target, camera, gray)
    canvas, n_tag = render(gray, uv, 8, tag_px, rms, 0.0)
    out = SAVE_DIR / "selftest_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"  检出 {n_tag} 个 tag，tag {tag_px and round(tag_px, 1)} px，"
          f"PnP {rms and round(rms, 2)} px")
    print(f"  叠加图已存 {out}")
    return 0 if n_tag >= 8 and (rms or 99) < 3 else 1


def run_live_preview(args):
    from calib.data import Camera, Target, read_yaml
    entry = read_yaml(args.target)
    target = Target(entry, min_tags=1)
    min_tags = int(getattr(cfg.SETTINGS, "min_tags", 8))
    camera = Camera.load(args.camera)

    cameras, seen = rs.scan_cameras(cfg.CAMERA_USB_ID)
    if not cameras:
        print(f"❌ 没找到 USB {cfg.CAMERA_USB_ID} 的相机。当前 /dev/video*：")
        for line in (seen or ["     （一个都没有）"]):
            print(f"     {line}")
        return 2
    device, name = cameras[0]
    print("=" * 74)
    print(f"实时标定板预览   {device} ({name})")
    print(f"  画面    {cfg.WIDTH}x{cfg.HEIGHT} @ {cfg.FPS} fps，预览缩到 {DISPLAY_WIDTH} 宽")
    print("  操作    Q / Esc 退出，S 存一张原图")
    print("=" * 74)
    fd, previous, _unsupported = lock_controls(device)
    try:
        cap = open_camera(device, cfg.WIDTH, cfg.HEIGHT, cfg.FPS)
    except BaseException:
        restore_controls(fd, previous)
        raise

    root = label = None
    if not args.headless:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            root = tk.Tk()
            root.title("Calibration preview   (Q quit / S save)")
            label = tk.Label(root, bd=0)
            label.pack()
        except Exception as exc:                       # noqa: BLE001
            print(f"⚠️ 起不了图形窗口（{exc}），退回终端统计模式")
            root = label = None
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"quit": False, "save": False}

    if root is not None:
        def on_key(event):
            char = (event.char or "").lower()
            if char == "q" or event.keysym == "Escape":
                state["quit"] = True
            elif char == "s":
                state["save"] = True
        root.bind("<KeyPress>", on_key)

    frames, t0, last_print = 0, time.time(), 0.0
    try:
        while not state["quit"]:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ 读帧失败，退出")
                return 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            uv, tag_px, rms = analyse(target, camera, gray)
            fps = frames / max(1e-6, time.time() - t0)
            canvas, n_tag = render(gray, uv, min_tags, tag_px, rms, fps)
            frames += 1
            if state["save"]:
                state["save"] = False
                out = SAVE_DIR / f"{time.time_ns()}.png"
                cv2.imwrite(str(out), frame)
                print(f"  已存 {out}")
            now = time.time()
            if now - last_print > 1.0:
                print(f"  {n_tag:2d} tag | tag {tag_px and round(tag_px)} px | "
                      f"PnP {rms and round(rms, 2)} px | {fps:.1f} fps", flush=True)
                last_print = now
            if root is not None:
                photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
                label.configure(image=photo)
                label.image = photo
                root.update_idletasks()
                root.update()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        restore_controls(fd, previous)
        if root is not None:
            root.destroy()
    print("退出。")
    return 0


def run_framing_capture_and_check(args):
    """默认模式：录 10 秒图 → 跑亮度 + 检出 + PnP + AprilGrid 参数反解。"""
    print("=" * 76)
    print("取景 / 曝光 / 标定板体检 —— 录 10 秒并自动诊断（不占用飞控 IMU）")
    print("=" * 76)

    cameras, seen = rs.scan_cameras(cfg.CAMERA_USB_ID)
    if not cameras:
        print(f"❌ 没找到 USB 身份为 {cfg.CAMERA_USB_ID} 的相机。当前 /dev/video* 上是：")
        for line in (seen or ["     （一个都没有）"]):
            print(f"     {line}")
        return 2
    device, name = cameras[0]
    board = rs.read_target(args.target)
    checker = board["kind"] in ("checkerboard", "chessboard")
    print(f"  相机      {device}  ({name})")
    print(f"  分辨率    {cfg.WIDTH}x{cfg.HEIGHT} @ {cfg.FPS} fps")
    print(f"  标定板    {args.target.name} → {board['label']}")
    print("─" * 76)
    print(f"  接下来 {SECONDS:.0f} 秒，把相机对准标定板缓慢移动，测试当前距离与光照：")
    print("  （若想看实时画框窗口，可运行：python capture/check_camera.py --preview）")
    print("─" * 76)

    shutil.rmtree(CHECK_DIR, ignore_errors=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(rs.CAMERA_SCRIPT),
        "--device", str(device),
        "--output", str(CHECK_DIR),
        "--width", str(cfg.WIDTH), "--height", str(cfg.HEIGHT), "--fps", str(cfg.FPS),
        "--exposure", str(cfg.EXPOSURE),
        "--white-balance", str(cfg.WHITE_BALANCE), "--gain", str(cfg.GAIN),
        "--cols", str(board["cols"]), "--rows", str(board["rows"]),
        "--auto-interval", "0.2", "--max-images", "0",
        "--duration", str(SECONDS),
    ] + rs.camera_exposure_args()
    if not checker:
        cmd.append("--no-preview")

    rc = subprocess.Popen(cmd).wait()
    images = sorted((CHECK_DIR / "cam0").glob("*.png"))
    if rc != 0 and not images and checker:
        print("⚠️ 尝试无预览模式重跑……")
        subprocess.Popen(cmd + ["--no-preview"]).wait()
    return inspect_directory(CHECK_DIR / "cam0", args.camera, args.target,
                             sample=args.sample, max_pnp_px=args.max_pnp_px)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="直接对已有图片目录做标定板/PnP/参数反解体检（不开相机）")
    ap.add_argument("--preview", action="store_true",
                    help="开启实时检测与 PnP HUD 预览窗")
    ap.add_argument("--headless", action="store_true",
                    help="配合 --preview：不开 GUI 窗口，仅在终端打印实时检测统计")
    ap.add_argument("--self-test", action="store_true",
                    help="离线合成 AprilGrid 自检（无需相机硬件）")
    ap.add_argument("--target", type=Path, default=cfg.TARGET_YAML)
    ap.add_argument("--camera", type=Path, default=cfg.CAMERA_YAML)
    ap.add_argument("--sample", type=int, default=16, help="体检抽样分析张数")
    ap.add_argument("--max-pnp-px", type=float, default=None,
                    help="PnP 残差门槛（默认取 extrinsics_config.SETTINGS.max_pnp_rms_px）")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args)
    if args.preview:
        return run_live_preview(args)
    if args.dir is not None:
        return inspect_directory(args.dir, args.camera, args.target,
                                 sample=args.sample, max_pnp_px=args.max_pnp_px)
    return run_framing_capture_and_check(args)


if __name__ == "__main__":
    sys.exit(main())
