#!/usr/bin/env python3
"""tools/compare_results.py —— 自研标定 vs Kalibr 对照 & 独立角速度时移扫描

整合原 kalibr/compare.py、analysis/compare_all.py 与 analysis/time_offset_scan.py：
  1. 单会话对比：
       python tools/compare_results.py session16
       python tools/compare_results.py --own results/session14/camchain-kalibr.yaml --kalibr ...
  2. 全量会话盘点与跨会话一致性矩阵（默认无参数或 --all）：
       python tools/compare_results.py --all
  3. 独立角速度模长 |ω| 互相关时间偏移扫描（无需已知外参旋转）：
       python tools/compare_results.py --scan-time-offset session14
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.data import Camera, Target, make_observation, read_yaml  # noqa: E402

OWN_RES = ROOT / "results"
KALIB_DIRS = [ROOT / "results" / "kalibr", ROOT / "kalibr" / "results", ROOT / "bags", ROOT / "kalibr" / "bags"]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def find_kalibr_yaml(session: str) -> Path | None:
    for base in KALIB_DIRS:
        for d in (base / session, base):
            hits = sorted(d.glob(f"{session}*-camchain-imucam.yaml")) if d.is_dir() else []
            if hits:
                return hits[0]
    return None


def load_yaml_matrix(path: Path):
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    cam = d["cam0"]
    T = np.array(cam["T_cam_imu"], dtype=float)
    ts = float(cam.get("timeshift_cam_imu", 0.0))
    intr = np.array(cam.get("intrinsics", [0, 0, 0, 0]), dtype=float)
    dist = np.array(cam.get("distortion_coeffs", [0, 0, 0, 0]), dtype=float)
    return T, ts, intr, dist


def rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1 @ R2.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def parse_kalibr_txt(path: Path) -> dict:
    if not path.is_file():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in (
        ("reproj_px", r"Reprojection error \(cam0\) \[px\]:\s*mean\s*([-\d.eE+]+)"),
        ("gyro", r"Gyroscope error \(imu0\) \[rad/s\]:\s*mean\s*([-\d.eE+]+)"),
        ("accel", r"Accelerometer error \(imu0\) \[m/s\^2\]:\s*mean\s*([-\d.eE+]+)"),
        ("timeshift", r"timeshift cam0 to imu0: \[s\] \(t_imu = t_cam \+ shift\)\s*\n\s*([-\d.eE+]+)"),
    ):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"Gravity vector in target coords: \[m/s\^2\]\s*\n\s*\[([^\]]+)\]", txt)
    if m:
        out["gravity"] = np.array([float(x) for x in m.group(1).split()])
    return out


def own_residuals(session: str) -> dict | None:
    rep = OWN_RES / session / "report.json"
    if not rep.is_file():
        return None
    d = json.loads(rep.read_text(encoding="utf-8"))
    m = d.get("final_metrics") or {}
    return {
        "reproj": m.get("reprojection_rms_px"),
        "gyro": m.get("gyro_rms_rad_s"),
        "accel": m.get("accel_rms_m_s2"),
        "status": d.get("status"),
    }


def kalibr_residuals(path: Path) -> dict | None:
    txt = path.parent / (path.name.replace("-camchain-imucam.yaml", "-results-imucam.txt"))
    m = parse_kalibr_txt(txt)
    if not m:
        return None
    return {
        "reproj": m.get("reproj_px"),
        "gyro": m.get("gyro"),
        "accel": m.get("accel"),
        "status": "ok",
    }


def compare_pair(kalib_yaml: Path, own_yaml: Path, label: str) -> None:
    Tk, tsk, *_ = load_yaml_matrix(kalib_yaml)
    To, tso, *_ = load_yaml_matrix(own_yaml)
    dR = rot_angle_deg(Tk[:3, :3], To[:3, :3])
    dt = To[:3, 3] - Tk[:3, 3]

    print(f"\n=== {label} ===")
    print(f"  Kalibr: {rel(kalib_yaml)}")
    print(f"  自研  : {rel(own_yaml)}")
    print(f"  旋转差 : {dR:8.2f} °")
    print(f"  平移差 : {np.linalg.norm(dt) * 1000:8.1f} mm"
          f"   (dx/dy/dz = {dt[0] * 1000:+.1f} / {dt[1] * 1000:+.1f} / {dt[2] * 1000:+.1f} mm)")
    print(f"  时偏差 : {(tso - tsk) * 1000:+8.1f} ms"
          f"   (自研 {tso * 1000:+.1f} ms, Kalibr {tsk * 1000:+.1f} ms)")
    print("  T_cam_imu（自研 / Kalibr）：")
    for i in range(3):
        a = "  ".join(f"{v: .6f}" for v in To[i])
        b = "  ".join(f"{v: .6f}" for v in Tk[i])
        print(f"    [{a}]   [{b}]")


# --------------------------------------------------------------------------
# 全量会话盘点与跨会话矩阵 (原 analysis/compare_all.py)
# --------------------------------------------------------------------------

def collect_all():
    selfs = {}
    for rep in sorted(OWN_RES.glob("session*/report.json")):
        sess = rep.parent.name
        chain = rep.parent / "camchain-kalibr.yaml"
        d = json.loads(rep.read_text(encoding="utf-8"))
        if not chain.exists():
            selfs[sess] = {"status": d.get("status"), "no_chain": True,
                           "final": d.get("final_metrics", {}),
                           "counts": d.get("counts", {}),
                           "duration_s": d.get("duration_s")}
            continue
        T, ts, intr, dist = load_yaml_matrix(chain)
        selfs[sess] = {"status": d.get("status"), "T": T, "ts": ts,
                       "intr": intr, "dist": dist,
                       "final": d.get("final_metrics", {}),
                       "counts": d.get("counts", {}),
                       "duration_s": d.get("duration_s")}
    kals = {}
    for base in (ROOT / "results" / "kalibr", ROOT / "kalibr" / "results"):
        if not base.is_dir():
            continue
        for d in sorted(base.glob("session*")):
            if d.name in kals:
                continue
            chain = d / f"{d.name}-camchain-imucam.yaml"
            txt = d / f"{d.name}-results-imucam.txt"
            if not chain.exists():
                continue
            T, ts, intr, dist = load_yaml_matrix(chain)
            kals[d.name] = {"T": T, "ts": ts, "intr": intr, "dist": dist,
                            "metrics": parse_kalibr_txt(txt)}
    return selfs, kals


def run_all_inventory() -> int:
    selfs, kals = collect_all()
    print("=" * 96)
    print("PHASE 1 — 自研标定结果汇总")
    print("=" * 96)
    print(f"{'session':<16}{'status':<42}{'frames':>7}{'dur[s]':>8}{'reproj':>8}{'gyro':>9}{'accel':>8}{'tshift[ms]':>11}")
    for s, r in selfs.items():
        st = str(r.get("status", "?"))[:40]
        fm = r.get("final", {})
        c = r.get("counts", {})
        if r.get("no_chain"):
            print(f"{s:<16}{st:<42}{'-':>7}{'-':>8}{'-':>8}{'-':>9}{'-':>8}{'-':>11}")
        else:
            print(f"{s:<16}{st:<42}{c.get('frames', '-'):>7}{r.get('duration_s', 0) or 0:>8.2f}"
                  f"{fm.get('reprojection_rms_px', float('nan')):>8.2f}"
                  f"{fm.get('gyro_rms_rad_s', float('nan')):>9.4f}"
                  f"{fm.get('accel_rms_m_s2', float('nan')):>8.3f}"
                  f"{1000 * r.get('ts', 0):>11.2f}")

    print()
    print("=" * 96)
    print("PHASE 2 — Kalibr 标定结果汇总")
    print("=" * 96)
    print(f"{'session':<16}{'reproj[px]':>11}{'gyro':>9}{'accel':>9}{'tshift[ms]':>12}{'|g|':>7}")
    for s, r in kals.items():
        m = r["metrics"]
        g = m.get("gravity")
        print(f"{s:<16}{m.get('reproj_px', float('nan')):>11.2f}{m.get('gyro', float('nan')):>9.4f}"
              f"{m.get('accel', float('nan')):>9.4f}{1000 * m.get('timeshift', float('nan')):>12.2f}"
              f"{np.linalg.norm(g) if g is not None else float('nan'):>7.3f}")

    both = [s for s in selfs if s in kals and not selfs[s].get("no_chain")]
    if both:
        print()
        print("=" * 96)
        print("PHASE 3 — 同会话对比 (自研 vs Kalibr)")
        print("=" * 96)
        print(f"{'session':<10}{'dRot[deg]':>11}{'|dT|[mm]':>10}{'dx':>8}{'dy':>8}{'dz':>8}{'dShift[ms]':>12}")
        for s in both:
            a, b = selfs[s], kals[s]
            dR = rot_angle_deg(a["T"][:3, :3], b["T"][:3, :3])
            dt = (a["T"][:3, 3] - b["T"][:3, 3]) * 1000.0
            dts = (a["ts"] - b["ts"]) * 1000.0
            print(f"{s:<10}{dR:>11.3f}{np.linalg.norm(dt):>10.2f}{dt[0]:>8.2f}{dt[1]:>8.2f}{dt[2]:>8.2f}{dts:>12.2f}")

    def pairwise(name, dct):
        keys = [k for k in dct if "T" in dct[k]]
        if len(keys) < 2:
            return
        print()
        print(f"--- {name}: 跨会话两两一致性 ({len(keys)} 个有效会话) ---")
        print(f"{'pair':<26}{'dRot[deg]':>11}{'dT[mm]':>9}{'dShift[ms]':>12}")
        vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = dct[keys[i]], dct[keys[j]]
                dR = rot_angle_deg(a["T"][:3, :3], b["T"][:3, :3])
                dt = np.linalg.norm(a["T"][:3, 3] - b["T"][:3, 3]) * 1000
                dts = (a["ts"] - b["ts"]) * 1000
                vals.append((dR, dt, dts))
                print(f"{keys[i][-12:] + ' vs ' + keys[j][-12:]:<26}{dR:>11.3f}{dt:>9.2f}{dts:>12.2f}")
        if vals:
            v = np.array(vals)
            print(f"{'median':<26}{np.median(v[:, 0]):>11.3f}{np.median(v[:, 1]):>9.2f}{np.median(v[:, 2]):>12.2f}")

    pairwise("自研 (SELF)", selfs)
    pairwise("KALIBR", kals)
    return 0


# --------------------------------------------------------------------------
# 独立角速度模长 |ω| 时间偏移扫描 (原 analysis/time_offset_scan.py)
# --------------------------------------------------------------------------

def _load_session_for_scan(session: str):
    d = ROOT / "data" / session
    manifest_path = d / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    saved = [s for s in (manifest.get("saved") or []) if s.get("v4l2_buffer_timestamp_ns")]
    by_name = {Path(s["file"]).name: s for s in saved}
    rows = []
    with (d / "images.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = Path(row["filename"]).name
            rows.append((int(row["timestamp"]), d / row["filename"], by_name.get(name)))
    rows.sort(key=lambda r: r[0])
    imu = np.loadtxt(d / "imu.csv", delimiter=",", skiprows=1)
    return rows, imu


def _pnp_poses(rows, camera, target):
    t_host, t_v4l2, omegas = [], [], []
    prev = None
    checker = target.kind in ("checkerboard", "chessboard")
    for stamp, path, meta in rows:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        det = target.detect(gray)
        obs = make_observation(stamp, *det, camera, prev, checker=checker, max_points=0) if det is not None else None
        if obs is not None and obs.pnp_rms <= 3.0:
            prev = obs
            t_host.append(stamp * 1e-9)
            if meta is not None:
                off = meta["host_receive_time_ns"] - meta["host_receive_monotonic_ns"]
                t_v4l2.append((meta["v4l2_buffer_timestamp_ns"] + off) * 1e-9)
            else:
                t_v4l2.append(np.nan)
            omegas.append((obs.r_wc, obs.pnp_rms))
        else:
            t_host.append(np.nan)
            t_v4l2.append(np.nan)
            omegas.append(None)
    return np.array(t_host), np.array(t_v4l2), omegas


def _frame_rates(t, poses):
    mid, rate, rms = [], [], []
    for k in range(len(t) - 1):
        if poses[k] is None or poses[k + 1] is None:
            continue
        if not (np.isfinite(t[k]) and np.isfinite(t[k + 1])):
            continue
        dt = t[k + 1] - t[k]
        if not (0.02 < dt < 0.12):
            continue
        dR = poses[k][0].T @ poses[k + 1][0]
        theta = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        rate.append(theta / dt)
        mid.append(0.5 * (t[k] + t[k + 1]))
        rms.append(max(poses[k][1], poses[k + 1][1]))
    return np.array(mid), np.array(rate), np.array(rms)


def _scan_offset(mid, rate, t_imu, gyro, dmin=-0.12, dmax=0.12, step=0.0002, rms=None):
    deltas = np.arange(dmin, dmax + step / 2, step)
    sel = rate > 0.25
    if rms is not None:
        sel &= rms < 2.0
    m, r = mid[sel], rate[sel]
    imu_at = m[None, :] + deltas[:, None]
    g = np.linalg.norm(gyro, axis=1)
    cost = np.empty(len(deltas))
    for i in range(len(deltas)):
        tq = imu_at[i]
        idx = np.clip(np.searchsorted(t_imu, tq), 1, len(t_imu) - 1)
        t0, t1 = t_imu[idx - 1], t_imu[idx]
        w = (tq - t0) / (t1 - t0)
        gm = (1 - w) * g[idx - 1] + w * g[idx]
        cost[i] = np.sqrt(np.mean((gm - r) ** 2))
    return float(deltas[np.argmin(cost)]), int(sel.sum())


def run_time_offset_scan(sessions: list[str], max_frames: int = 0) -> int:
    import extrinsics_config as cfg
    cam = Camera.load(cfg.CAMERA_YAML)
    tgt = Target(read_yaml(cfg.TARGET_YAML))
    print(f"{'session':<12}{'valid':>8}{'pairs':>8}{'used':>7}{'delta_host[ms]':>16}{'delta_v4l2[ms]':>16}")
    for s in sessions:
        rows, imu = _load_session_for_scan(s)
        if max_frames:
            rows = rows[:max_frames]
        th, tv, poses = _pnp_poses(rows, cam, tgt)
        n_valid = sum(p is not None for p in poses)
        mid_h, rate_h, rms_h = _frame_rates(th, poses)
        d_h, used = _scan_offset(mid_h, rate_h, imu[:, 0] * 1e-9, imu[:, 4:7], rms=rms_h)
        mid_v, rate_v, rms_v = _frame_rates(tv, poses)
        d_v = _scan_offset(mid_v, rate_v, imu[:, 0] * 1e-9, imu[:, 4:7], rms=rms_v)[0] if len(mid_v) else float("nan")
        print(f"{s:<12}{n_valid:>8}{len(mid_h):>8}{used:>7}{1000 * d_h:>16.2f}{1000 * d_v:>16.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="对比指定 session（如 session16）")
    ap.add_argument("--own", type=Path, help="自研 camchain-kalibr.yaml 路径")
    ap.add_argument("--kalibr", type=Path, help="Kalibr *-camchain-imucam.yaml 路径")
    ap.add_argument("--all", action="store_true", help="盘点所有自研与 Kalibr 会话结果并输出跨会话一致性矩阵")
    ap.add_argument("--scan-time-offset", nargs="+", metavar="SESSION",
                    help="用 PnP 与陀螺仪 |ω| 互相关独立扫描验证时间偏移")
    ap.add_argument("--max-frames", type=int, default=0, help="配合 --scan-time-offset 限制最大帧数")
    args = ap.parse_args(argv)

    if args.scan_time_offset:
        return run_time_offset_scan(args.scan_time_offset, args.max_frames)

    if args.own and args.kalibr:
        compare_pair(args.kalibr, args.own, "自定义对比")
        return 0

    if args.all or not args.session:
        return run_all_inventory()

    s = args.session
    kalib_yaml = args.kalibr or find_kalibr_yaml(s)
    own_yaml = args.own or (OWN_RES / s / "camchain-kalibr.yaml")
    if kalib_yaml is None:
        sys.exit(f"没找到 {s} 的 Kalibr 结果。先跑：python tools/run_kalibr.py {s}")
    if not own_yaml.is_file():
        sys.exit(f"没找到自研结果 {own_yaml}")
    compare_pair(kalib_yaml, own_yaml, s)

    ka = kalibr_residuals(kalib_yaml)
    oa = own_residuals(s)
    print("\n  残差（两套定义/筛选不同，仅供参考）：")
    print(f"    {'':<10}{'reproj[px]':>12}{'gyro[rad/s]':>14}{'accel[m/s²]':>14}   status")
    for name, r in (("自研", oa), ("Kalibr", ka)):
        if not r:
            print(f"    {name:<10}{'—':>12}{'—':>14}{'—':>14}")
            continue
        f = lambda v, fmt: ("—" if v is None else format(v, fmt))  # noqa: E731
        print(f"    {name:<10}{f(r['reproj'], '.3f'):>12}{f(r['gyro'], '.5f'):>14}"
              f"{f(r['accel'], '.4f'):>14}   {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
