#!/usr/bin/env python3
"""tools/run_kalibr.py —— 一键用 Docker 里的 Kalibr 标定 camera-IMU 外参 + 时间偏移

数据流
------
    data/<session>/            images.csv + imu.csv + cam0/*.png
        │  tools/make_rosbag.py
        ▼
    bags/<session>.bag         ROS1 bag，/cam0/image_raw + /imu0
        │  docker run kalibr:latest
        ▼
    results/kalibr/<session>/  *-camchain-imucam.yaml / *-results-imucam.txt / *-report-imucam.pdf

用法
----
    conda activate vins
    cd extrinsic
    python tools/run_kalibr.py                       # 用最新的 session
    python tools/run_kalibr.py session17             # 指定 session
    python tools/run_kalibr.py session17 --rebuild   # 强制重打 bag
    python tools/run_kalibr.py --target-type checkerboard --cols 8 --rows 11 --square-size 0.03
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent          # extrinsic/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrinsics_config import IMU_NOISE_DEFAULTS as IMU_DEFAULTS  # noqa: E402

DATA = ROOT / "data"
BAGS = ROOT / "bags"
RES = ROOT / "results" / "kalibr"
CONF = RES / "_configs"
LOGS = RES / "_logs"

MOUNT = "/data"
IMAGE = "kalibr:latest"
TARGET_APRILGRID = dict(rows=6, cols=6, tag_size=0.088, tag_spacing=0.56)


def info(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1):
    print("\n❌ " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def container_path(p: Path) -> str:
    return MOUNT + "/" + p.relative_to(ROOT).as_posix()


def find_sessions() -> list[str]:
    if not DATA.is_dir():
        return []
    rows = [(p.stat().st_mtime, p.name) for p in DATA.iterdir()
            if p.is_dir() and (p / "images.csv").is_file() and (p / "imu.csv").is_file()]
    return [name for _, name in sorted(rows, reverse=True)]


def resolve_session(arg: str | None) -> str:
    sessions = find_sessions()
    if not sessions:
        fail(f"{DATA} 下没有同时含 images.csv 和 imu.csv 的 session")
    if arg is None:
        return sessions[0]
    if arg not in sessions:
        fail(f"没有这个 session：{arg}\n   可用的：{'  '.join(sessions[:8])}")
    return arg


def resolve_board_size(args) -> tuple[int, int]:
    if args.rows is not None and args.cols is not None:
        return args.rows, args.cols
    if args.target_type == "aprilgrid":
        r, c = TARGET_APRILGRID["rows"], TARGET_APRILGRID["cols"]
    else:
        r, c = 11, 8
    return (args.rows if args.rows is not None else r,
            args.cols if args.cols is not None else c)


def make_target_yaml(args) -> str:
    rows, cols = resolve_board_size(args)
    if args.target_type == "aprilgrid":
        return (
            "# Kalibr AprilGrid (tag36h11)\n"
            "target_type: 'aprilgrid'\n"
            f"tagRows: {rows}\n"
            f"tagCols: {cols}\n"
            f"tagSize: {args.tag_size}\n"
            f"tagSpacing: {args.tag_spacing}\n"
        )
    return (
        "# Kalibr checkerboard\n"
        "target_type: 'checkerboard'\n"
        f"targetCols: {cols}\n"
        f"targetRows: {rows}\n"
        f"rowSpacingMeters: {args.square_size}\n"
        f"colSpacingMeters: {args.square_size}\n"
    )


def make_imu_yaml() -> str:
    d = IMU_DEFAULTS
    return (
        "# Kalibr IMU continuous-time noise\n"
        "rostopic: /imu0\n"
        f"update_rate: {d['update_rate']}\n\n"
        f"accelerometer_noise_density: {d['accelerometer_noise_density']:.4e}\n"
        f"accelerometer_random_walk:   {d['accelerometer_random_walk']:.4e}\n"
        f"gyroscope_noise_density:     {d['gyroscope_noise_density']:.4e}\n"
        f"gyroscope_random_walk:       {d['gyroscope_random_walk']:.4e}\n"
    )


def make_camchain_yaml(args) -> str | None:
    src = Path(args.camchain) if args.camchain else (ROOT / "configs" / "gemini_pro_rgb.yaml")
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    if not src.is_file():
        return None
    return src.read_text(encoding="utf-8")


def ensure_configs(args) -> None:
    info("配置：")
    (CONF / "target.yaml").write_text(make_target_yaml(args), encoding="utf-8")
    (CONF / "imu.yaml").write_text(make_imu_yaml(), encoding="utf-8")
    cam = make_camchain_yaml(args)
    if cam is None:
        fail("找不到 configs/gemini_pro_rgb.yaml（相机内参），请用 --camchain 指定")
    (CONF / "camchain.yaml").write_text(cam, encoding="utf-8")
    info(f"  已生成运行配置于 {CONF.relative_to(ROOT)}/")


def _docker_ok(prefix: list[str]) -> bool:
    return subprocess.run(prefix + ["info"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def docker_prefix() -> list[str]:
    if shutil.which("docker") is None:
        fail("没有找到 docker 命令")
    if _docker_ok(["docker"]):
        return ["docker"]
    info("  当前环境没有 docker 组权限，改用 sudo docker（会提示输入密码）")
    if _docker_ok(["sudo", "docker"]):
        return ["sudo", "docker"]
    fail("docker 不可用：直接连和 sudo 连都失败，先排查 docker 服务/权限")


def run_tee(cmd: list[str], log_path: Path | None = None) -> int:
    info("  $ " + " ".join(cmd))
    fh = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if fh:
                fh.write(line)
        return proc.wait()
    finally:
        if fh:
            fh.close()


def build_bag(session: str, rebuild: bool) -> Path:
    bag = BAGS / f"{session}.bag"
    if bag.exists() and not rebuild:
        info(f"bag：复用已有 bags/{session}.bag（{bag.stat().st_size / 1e6:.0f} MB）")
        return bag
    script = ROOT / "tools" / "make_rosbag.py"
    if not script.is_file():
        fail(f"找不到 {script}")
    info(f"bag：从 data/{session}/ 打包（复用 tools/make_rosbag.py）")
    rc = run_tee([sys.executable, str(script), "--session", session, "--output", str(bag)])
    if rc != 0 or not bag.exists():
        fail("打 bag 失败，先把上面的报错解决")
    info(f"       → bags/{session}.bag（{bag.stat().st_size / 1e6:.0f} MB）")
    return bag


def bag_topic_check(bag: Path) -> tuple[str, str, int]:
    try:
        from rosbags.rosbag1 import Reader
        with Reader(bag) as reader:
            image = [c.topic for c in reader.connections if c.msgtype.endswith("/Image")]
            imu = [c.topic for c in reader.connections if c.msgtype.endswith("/Imu")]
            if image and imu:
                return image[0], imu[0], len(reader.connections)
            return "", "", len(reader.connections)
    except Exception:                                          # noqa: BLE001
        return "", "", 0


def align_topics(image_topic: str, imu_topic: str) -> None:
    imu_yaml = CONF / "imu.yaml"
    text = imu_yaml.read_text(encoding="utf-8")
    new = re.sub(r"^rostopic:.*$", f"rostopic: {imu_topic}", text, count=1, flags=re.MULTILINE)
    if new != text:
        imu_yaml.write_text(new, encoding="utf-8")

    cam_yaml = CONF / "camchain.yaml"
    text = cam_yaml.read_text(encoding="utf-8")
    new, n = re.subn(r"^(\s+)rostopic:.*$", rf"\g<1>rostopic: {image_topic}", text,
                     count=1, flags=re.MULTILINE)
    if n and new != text:
        cam_yaml.write_text(new, encoding="utf-8")


def time_stamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_kalibr(args, docker: list[str], session: str, bag: Path) -> Path:
    out_dir = RES / session
    out_dir.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{session}_{time_stamp()}.log"

    inner = (
        "source /catkin_ws/devel/setup.bash && "
        "rosrun kalibr kalibr_calibrate_imu_camera "
        f"--bag {container_path(bag)} "
        f"--cams {container_path(CONF / 'camchain.yaml')} "
        f"--imu {container_path(CONF / 'imu.yaml')} "
        f"--target {container_path(CONF / 'target.yaml')}"
    )
    if args.from_to:
        inner += f" --bag-from-to {args.from_to[0]} {args.from_to[1]}"

    cmd = docker + ["run", "--rm",
                    "-v", f"{ROOT}:{MOUNT}",
                    "-w", container_path(out_dir),
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp/mpl",
                    "-e", "MPLBACKEND=Agg", "-e", "QT_QPA_PLATFORM=offscreen",
                    IMAGE, "/bin/bash", "-c", inner]
    info(f"\n开始标定：docker（日志 {log.relative_to(ROOT)}）\n")
    rc = run_tee(cmd, log)

    stem = str(bag.with_suffix(""))
    produced = []
    for suffix in ("-camchain-imucam.yaml", "-imu.yaml",
                   "-results-imucam.txt", "-report-imucam.pdf"):
        src = Path(stem + suffix)
        if src.exists():
            produced.append(src)
            shutil.copy2(src, out_dir / src.name)

    got_camchain = any(p.name.endswith("-camchain-imucam.yaml") for p in produced)
    if not got_camchain:
        info(f"\n--- {log.relative_to(ROOT)} 最后 30 行 ---")
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        for line in tail:
            info("  " + line)
        fail(f"Kalibr 没有产出 camchain-imucam.yaml（退出码 {rc}）"
             f" —— 完整日志见 {log.relative_to(ROOT)}")
    if rc != 0:
        info(f"\n⚠️ Kalibr 退出码 {rc}，但结果文件已生成。")
    info(f"结果已复制到 {out_dir.relative_to(ROOT)}/：")
    for f in produced:
        info(f"    {f.name}")
    return log


def report(session: str) -> None:
    out_dir = RES / session
    files = sorted(p for p in out_dir.glob("*") if p.is_file())
    cam_imu = next((p for p in files if p.name.endswith("-camchain-imucam.yaml")), None)
    results = next((p for p in files if p.name.endswith("-results-imucam.txt")), None)

    info("\n" + "=" * 72)
    info(f"  结果目录：{out_dir.relative_to(ROOT)}/")
    for p in files:
        info(f"    {p.name}")
    info("=" * 72)

    if results is not None:
        info("\n--- results-imucam.txt ---")
        info(results.read_text(encoding="utf-8", errors="replace").strip()[:4000])

    if cam_imu is None:
        info("\n⚠️ 没有生成 camchain-imucam.yaml —— 标定没收敛。")
        return

    try:
        import numpy as np
        import yaml
        data = yaml.safe_load(cam_imu.read_text(encoding="utf-8"))
        cam = data["cam0"]
        T = cam["T_cam_imu"]
        ts = cam.get("timeshift_cam_imu", 0.0)
        info("\nT_cam_imu（IMU → 相机，Kalibr 约定）：")
        for row in T:
            info("    " + "  ".join(f"{v: .6f}" for v in row))
        info(f"\ntimeshift_cam_imu = {ts:+.6f} s   （相机时间戳相对 IMU 的偏移）")
        Ti = np.linalg.inv(np.asarray(T, dtype=float))
        info("\nT_imu_cam = inv(T_cam_imu)（OpenVINS 的 T_imu_cam 用这个）：")
        for row in Ti:
            info("    " + "  ".join(f"{v: .6f}" for v in row))
    except Exception as exc:                                   # noqa: BLE001
        info(f"\n（解析 {cam_imu.name} 失败：{exc}）")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用 Docker 里的 Kalibr 标定 data/<session> 的 camera-IMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python tools/run_kalibr.py session17 --rebuild")
    ap.add_argument("session", nargs="?", default=None,
                    help="session 名（默认用 data/ 下最新一个）")
    ap.add_argument("--rebuild", action="store_true", help="强制重新打 bag")
    ap.add_argument("--no-verify", action="store_true", help="跳过 bag 话题校验")
    ap.add_argument("--target-type", choices=["aprilgrid", "checkerboard"],
                    default="aprilgrid", help="标定板类型（默认 aprilgrid）")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--tag-size", type=float, default=TARGET_APRILGRID["tag_size"])
    ap.add_argument("--tag-spacing", type=float, default=TARGET_APRILGRID["tag_spacing"])
    ap.add_argument("--square-size", type=float, default=0.03)
    ap.add_argument("--camchain", type=Path, default=None)
    ap.add_argument("--from-to", nargs=2, type=float, metavar=("T0", "T1"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for d in (CONF, BAGS, RES, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    session = resolve_session(args.session)
    info(f"session：{session}" + ("（最新）" if args.session is None else ""))
    info(f"项目根：{ROOT}")

    ensure_configs(args)
    bag = build_bag(session, args.rebuild)
    docker = docker_prefix()

    if not args.no_verify:
        image_topic, imu_topic, n_topics = bag_topic_check(bag)
        if image_topic and imu_topic:
            info(f"bag 话题：image={image_topic}  imu={imu_topic}（共 {n_topics} 个）")
            align_topics(image_topic, imu_topic)

    run_kalibr(args, docker, session, bag)
    report(session)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        info("\n已中断")
        sys.exit(130)
