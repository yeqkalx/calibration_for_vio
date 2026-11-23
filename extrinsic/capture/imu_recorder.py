"""MAVLink 飞控 IMU → imu.csv（外参标定用）

不依赖 pyserial：用内核自带的 termios + os.read。
MAVLink 编解码使用同目录 `mavlink_min.py`。

时间戳策略：
  相机侧记的是宿主机 CLOCK_REALTIME（`time.time_ns()`）。这里必须记同一个钟。
  1) 批内：用飞控自己的 `time_usec` 算相对间隔，抖动压到亚毫秒级；
  2) 全段：录完后用 (宿主收到时间 − time_usec) 的下包络做 Theil-Sen 直线拟合，
     把每批 read 的锚点换成全段统一映射（`fit_timebase` / `anchor_timestamps`），
     使读线程积压不再进入时间戳。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import combinations
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.mavlink_min import MavlinkParser, pack  # noqa: E402

MAV_CMD_SET_MESSAGE_INTERVAL = 511
MSG_HIGHRES_IMU = 105
IMU_FIELD_BITS = 0x3F
CSV_HEADER = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]

# 下包络拟合常数
_FLOOR_QUANTILE = 0.001
_MAX_DRIFT_PPM = 500.0
_MAX_SHIFT_NS = 5_000_000_000


# --------------------------------------------------------------------------
# 飞控 time_usec -> 宿主 REALTIME 下包络时间基准拟合
# --------------------------------------------------------------------------

def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return float(s[n // 2]) if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    k = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[min(len(sorted_values) - 1, max(0, k))]


def fit_timebase(fc_usec, host_ns, *, min_samples=50, max_bins=64):
    """对 (飞控时间, 宿主收到时间) 做下包络直线拟合。"""
    pairs = [(int(f), int(h)) for f, h in zip(fc_usec, host_ns) if f > 0 and h > 0]
    if len(pairs) < min_samples:
        return {"ok": False, "reason": f"有飞控时间的样本只有 {len(pairs)} 个"}
    f0 = pairs[0][0]
    last = None
    for f, _h in pairs:
        if last is not None and f <= last:
            return {"ok": False, "reason": "飞控 time_usec 不严格递增（重启/回绕？）"}
        last = f
    span_ns = (pairs[-1][0] - f0) * 1000
    if span_ns < 1_000_000_000:
        return {"ok": False, "reason": f"跨度只有 {span_ns / 1e9:.2f} s，不足以拟合"}
    offsets = [h - f * 1000 for f, h in pairs]
    k = max(4, min(max_bins, len(pairs) // 50))
    binned = []
    for i in range(k):
        a, b = i * len(pairs) // k, (i + 1) * len(pairs) // k
        if b <= a:
            continue
        j = min(range(a, b), key=offsets.__getitem__)
        binned.append(((pairs[j][0] - f0) * 1000, offsets[j]))
    if len(binned) < 4:
        return {"ok": False, "reason": "下包络样本不足"}
    slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in combinations(binned, 2) if x2 > x1]
    drift = _median(slopes) if slopes else 0.0
    drift_ppm = drift * 1e6
    if abs(drift_ppm) > _MAX_DRIFT_PPM:
        return {"ok": False, "reason": f"飞控/宿主时钟频率差 {drift_ppm:.0f} ppm 异常"}
    corrected = sorted((h - f * 1000) - drift * (f - f0) * 1000 for f, h in pairs)
    floor_ns = _quantile(corrected, _FLOOR_QUANTILE)
    delays = sorted(d - floor_ns for d in corrected)
    return {
        "ok": True, "reason": "",
        "offset_ns": floor_ns, "drift": drift, "f0_usec": f0,
        "drift_ppm": drift_ppm, "span_s": span_ns / 1e9, "bins": len(binned),
        "delay_median_ms": _quantile(delays, 0.5) / 1e6,
        "delay_p95_ms": _quantile(delays, 0.95) / 1e6,
    }


def anchor_timestamps(fc_usec, host_ns, provisional_ns, *, min_samples=50):
    """把样本时间戳改成"飞控时间 + 下包络映射"。"""
    provisional = [int(t) for t in provisional_ns]
    fit = fit_timebase(fc_usec, host_ns, min_samples=min_samples)
    if not fit["ok"]:
        return provisional, {"ok": False, "reason": fit["reason"],
                             "fallback_samples": len(provisional), "shift_median_ms": None}
    offset, drift, f0 = fit["offset_ns"], fit["drift"], fit["f0_usec"]
    out, last, clamped, fallback = [], None, 0, 0
    for f, prov in zip(fc_usec, provisional):
        f = int(f)
        if f > 0:
            x = (f - f0) * 1000
            t = int(round(f * 1000 + offset + drift * x))
        else:
            t = prov
            fallback += 1
        if last is not None and t <= last:
            t = last + 1
            clamped += 1
        last = t
        out.append(t)
    shifts = sorted(p - t for p, t in zip(provisional, out))
    info = {**fit,
            "shift_median_ms": _quantile(shifts, 0.5) / 1e6,
            "shift_p95_ms": _quantile(shifts, 0.95) / 1e6,
            "shift_max_ms": shifts[-1] / 1e6,
            "clamped": clamped, "fallback_samples": fallback}
    if max(abs(s) for s in shifts) > _MAX_SHIFT_NS:
        return provisional, {"ok": False, "reason": "锚定前后时间戳差超过 5 s，疑似拟合异常",
                             "fallback_samples": len(provisional), "shift_median_ms": None}
    return out, info


# --------------------------------------------------------------------------
# 串口
# --------------------------------------------------------------------------

def _baud_constant(baud: int) -> int:
    name = f"B{int(baud)}"
    if not hasattr(termios, name):
        raise ValueError(f"termios 不支持波特率 {baud}；USB CDC 下随便填 115200 即可")
    return getattr(termios, name)


def open_serial(path, baud=115200):
    """打开串口并配成 8N1 原始模式。返回 fd。"""
    fd = os.open(str(path), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        a = termios.tcgetattr(fd)
        a[0] = 0
        a[1] = 0
        a[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        a[3] = 0
        a[4] = a[5] = _baud_constant(baud)
        termios.tcsetattr(fd, termios.TCSANOW, a)
        termios.tcflush(fd, termios.TCIOFLUSH)
    except Exception:
        os.close(fd)
        raise
    return fd


def list_serial_candidates():
    """列出可能的飞控串口。CUAV 7-Nano 会同时枚举出 ttyACM0 和 ttyACM1。"""
    out = []
    for pattern in ("ttyACM*", "ttyUSB*"):
        out += sorted(Path("/dev").glob(pattern))
    return out


def find_mavlink_port(candidates=None, timeout=1.5, baud=115200, log=print):
    """在候选串口里逐个探测，返回第一个真的在讲 MAVLink 的口。

    先被动监听一小段（兼容 ArduPilot 等主动发流的飞控）；若期间静默，
    则发 GCS HEARTBEAT 唤醒等待地面站心跳的飞控（如 PX4 / MicoAir USB CDC）。
    """
    candidates = list(candidates) if candidates else list_serial_candidates()
    if not candidates:
        return None, {}
    hb_v2 = pack(0, {"type": 6, "autopilot": 8, "system_status": 4, "mavlink_version": 3},
                 sysid=255, compid=190, seq=0, version=2)
    hb_v1 = pack(0, {"type": 6, "autopilot": 8, "system_status": 4, "mavlink_version": 3},
                 sysid=255, compid=190, seq=1, version=1)
    log(f"串口候选 {[str(c) for c in candidates]}，逐个探测 {timeout:g}s 找 MAVLink……")
    report = {}
    for path in candidates:
        try:
            fd = open_serial(path, baud)
        except Exception as exc:                       # noqa: BLE001
            report[str(path)] = f"打不开: {exc}"
            log(f"  {path}: 打不开（{exc}）")
            continue
        parser = MavlinkParser()
        seen = {}
        t0 = time.monotonic()
        deadline = t0 + timeout
        wake_v2_at = t0 + min(0.35, timeout * 0.25)
        wake_v1_at = t0 + min(0.85, timeout * 0.55)
        sent_v2 = False
        sent_v1 = False
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                if not seen:
                    if not sent_v2 and now >= wake_v2_at:
                        try:
                            os.write(fd, hb_v2)
                        except OSError:
                            pass
                        sent_v2 = True
                    elif not sent_v1 and now >= wake_v1_at:
                        try:
                            os.write(fd, hb_v1)
                        except OSError:
                            pass
                        sent_v1 = True
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                try:
                    data = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                for frame in parser.feed(data or b""):
                    seen[frame.name] = seen.get(frame.name, 0) + 1
        finally:
            os.close(fd)
        if seen:
            woke = "（GCS 心跳唤醒）" if (sent_v2 or sent_v1) else ""
            report[str(path)] = f"{sum(seen.values())} 帧{woke} {sorted(seen)}"
            log(f"  {path}: ✅ 收到 {sum(seen.values())} 帧{woke} {sorted(seen)}")
            return str(path), report
        report[str(path)] = "没有有效 MAVLink 帧"
        log(f"  {path}: 静默（没有有效 MAVLink 帧）")
    return None, report


# --------------------------------------------------------------------------
# 记录器
# --------------------------------------------------------------------------

@dataclass
class _Sample:
    timestamp_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    fc_usec: int = 0
    host_ns: int = 0


@dataclass
class ImuRecorder:
    """后台线程持续读串口 → 内存里的样本列表 → 一次性写成 CSV。"""

    port: str
    request_hz: float = 200.0
    baud: int = 115200
    log: object = print

    samples: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    _fd: int = field(default=0, init=False, repr=False)
    _thread: threading.Thread = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _first: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _error: str = field(default="", init=False)
    _timebase: dict = field(default_factory=dict, init=False, repr=False)
    _anchored: bool = field(default=False, init=False, repr=False)
    startup_discard_s: float = field(default=0.0, kw_only=True)

    def start(self, wait_first_s=3.0):
        self._fd = open_serial(self.port, self.baud)
        self._thread = threading.Thread(target=self._loop, name="imu-recorder", daemon=True)
        self._thread.start()
        timeout = wait_first_s + self.startup_discard_s
        if wait_first_s and not self._first.wait(timeout):
            self.stop()
            raise RuntimeError(
                f"{self.port} 打开了但 {timeout:g}s 内没收到可记录的 HIGHRES_IMU。"
                "飞控在跑吗？USB 口对吗？换另一个 ttyACM 试试。")
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._fd:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = 0
        self._anchor_timebase()
        self._finish_stats()
        return self.stats

    def _anchor_timebase(self):
        """录完后把时间戳从"每批 read 锚点"改成"全段飞控时间锚点"（幂等）。"""
        if self._anchored:
            return self._timebase
        self._anchored = True
        if not self.samples:
            self._timebase = {"ok": False, "reason": "没有样本"}
            return self._timebase
        stamps, info = anchor_timestamps(
            [s.fc_usec for s in self.samples],
            [s.host_ns for s in self.samples],
            [s.timestamp_ns for s in self.samples])
        for s, t in zip(self.samples, stamps):
            s.timestamp_ns = int(t)
        self._timebase = info
        return info

    def _loop(self):
        parser = MavlinkParser()
        target = None
        version = 2
        next_heartbeat = 0.0
        next_request = 0.0
        ack_result = None
        tx_seq = 0
        n_wake = 0
        t_open = time.monotonic()
        record_after = t_open + self.startup_discard_s
        last_time_usec = None
        last_ts = None
        instances = set()
        n_frames = 0
        n_dup = 0
        n_fields_updated_zero = 0
        n_interpolated = 0
        n_clamped = 0
        decode_fail = 0

        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if target is not None:
                    try:
                        if now >= next_heartbeat:
                            os.write(self._fd, pack(0, {"type": 6, "autopilot": 8,
                                                       "system_status": 4, "mavlink_version": 3},
                                                    version=version, seq=tx_seq))
                            tx_seq += 1
                            if next_heartbeat == 0.0:
                                next_request = now + 0.35
                            next_heartbeat = now + 1.0
                        if now >= next_request:
                            os.write(self._fd, pack(76, {
                                "param1": float(MSG_HIGHRES_IMU),
                                "param2": 1e6 / self.request_hz if self.request_hz > 0 else -1.0,
                                "command": MAV_CMD_SET_MESSAGE_INTERVAL,
                                "target_system": target[0], "target_component": target[1],
                            }, version=version, seq=tx_seq))
                            tx_seq += 1
                            next_request = now + 2.0
                    except OSError as exc:
                        self._error = f"写串口失败: {exc}"
                        return
                elif now - t_open >= 0.3 and now >= next_heartbeat:
                    # 唤醒等待 GCS HEARTBEAT 的飞控（如 PX4 / MicoAir USB CDC）
                    wake_ver = 2 if (n_wake % 2 == 0) else 1
                    try:
                        os.write(self._fd, pack(0, {"type": 6, "autopilot": 8,
                                                   "system_status": 4, "mavlink_version": 3},
                                                version=wake_ver, seq=tx_seq))
                        tx_seq += 1
                        n_wake += 1
                        next_heartbeat = now + 1.0
                    except OSError as exc:
                        self._error = f"写串口失败: {exc}"
                        return

                r, _, _ = select.select([self._fd], [], [], 0.05)
                if not r:
                    continue
                try:
                    data = os.read(self._fd, 4096)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    self._error = f"读串口失败: {exc}"
                    return
                if not data:
                    continue

                t_host = time.time_ns()
                frames = parser.feed(data)
                if not frames:
                    continue
                for frame in frames:
                    source = (frame.sysid, frame.compid)
                    if frame.name == "HEARTBEAT" and frame.compid == 1:
                        if (source, frame.version) != (target, version):
                            target, version = source, frame.version
                            next_heartbeat = 0.0
                    elif target is None and frame.name == "HIGHRES_IMU":
                        target, version = source, frame.version
                    elif (source == target and frame.name == "COMMAND_ACK"
                          and frame.fields.get("command") == MAV_CMD_SET_MESSAGE_INTERVAL):
                        ack_result = frame.fields.get("result")

                if time.monotonic() < record_after:
                    continue
                hi = [f for f in frames if f.name == "HIGHRES_IMU"]
                if not hi:
                    continue
                n_frames += len(hi)

                anchor_usec = int(hi[-1].fields.get("time_usec", 0) or 0)
                for frame in hi:
                    f = frame.fields
                    try:
                        t_usec = int(f.get("time_usec", 0) or 0)
                        if t_usec and last_time_usec is not None and t_usec == last_time_usec:
                            n_dup += 1
                            continue
                        if t_usec:
                            last_time_usec = t_usec
                        if (int(f.get("fields_updated", 0) or 0) & IMU_FIELD_BITS) == 0:
                            n_fields_updated_zero += 1
                        if t_usec and anchor_usec and anchor_usec >= t_usec:
                            ts = t_host - (anchor_usec - t_usec) * 1000
                            n_interpolated += 1
                        else:
                            ts = t_host
                        if last_ts is not None and ts <= last_ts:
                            ts = last_ts + 1
                            n_clamped += 1
                        last_ts = ts
                        inst = int(f.get("id", 0) or 0)
                        instances.add(inst)
                        self.samples.append(_Sample(
                            ts,
                            float(f["xacc"]), float(f["yacc"]), float(f["zacc"]),
                            float(f["xgyro"]), float(f["ygyro"]), float(f["zgyro"]),
                            fc_usec=t_usec or 0, host_ns=t_host))
                    except (KeyError, TypeError, ValueError):
                        decode_fail += 1
                    if not self._first.is_set():
                        self._first.set()
        finally:
            self._finish_extra = dict(
                frames=n_frames, duplicates=n_dup,
                fields_updated_zero=n_fields_updated_zero,
                interpolated=n_interpolated, decode_fail=decode_fail,
                clamped=n_clamped,
                instances=sorted(instances), parser_stats=dict(parser.stats),
                mavlink_version=version if target is not None else None,
                rate_request_ack=ack_result,
                target=target)

    def _finish_stats(self):
        extra = getattr(self, "_finish_extra", {}) or {}
        ts = [s.timestamp_ns for s in self.samples]
        gaps_ms = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
        gaps_sorted = sorted(gaps_ms)
        span = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
        instances = extra.get("instances", [])
        if self.samples:
            mags = sorted(math.degrees(math.sqrt(s.gx ** 2 + s.gy ** 2 + s.gz ** 2))
                          for s in self.samples)
            n = len(mags)
            gyro_med = mags[n // 2]
            gyro_p95 = mags[min(n - 1, int(0.95 * n))]
            def _std(values):
                mean = sum(values) / len(values)
                return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            gyro_std = math.degrees(max(_std([s.gx for s in self.samples]),
                                        _std([s.gy for s in self.samples]),
                                        _std([s.gz for s in self.samples])))
        else:
            gyro_med = gyro_p95 = gyro_std = 0.0
        stats = {
            "port": self.port,
            "requested_hz": self.request_hz,
            "mavlink_version": extra.get("mavlink_version"),
            "rate_request_ack": extra.get("rate_request_ack"),
            "samples": len(self.samples),
            "frames_received": extra.get("frames", 0),
            "duplicates_skipped": extra.get("duplicates", 0),
            "span_s": span,
            "observed_hz": (len(ts) - 1) / span if span > 0 else 0.0,
            "median_gap_ms": gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else 0.0,
            "max_gap_ms": max(gaps_ms) if gaps_ms else 0.0,
            "gaps_over_20ms": sum(1 for g in gaps_ms if g > 20.0),
            "gaps_over_50ms": sum(1 for g in gaps_ms if g > 50.0),
            "gaps_over_150ms": sum(1 for g in gaps_ms if g > 150.0),
            "instances": instances,
            "mixed_instances": len(instances) > 1,
            "gyro_med_deg_s": gyro_med,
            "gyro_p95_deg_s": gyro_p95,
            "gyro_axis_std_deg_s": gyro_std,
            "fields_updated_zero": extra.get("fields_updated_zero", 0),
            "timestamps_interpolated": extra.get("interpolated", 0),
            "timestamps_clamped": extra.get("clamped", 0),
            "timestamp_anchor": "fc_usec" if self._timebase.get("ok") else "read_time",
            "timebase": dict(self._timebase),
            "decode_failures": extra.get("decode_fail", 0),
            "parser": extra.get("parser_stats", {}),
            "error": self._error,
        }
        self.stats = stats
        return stats

    def write_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            for s in self.samples:
                writer.writerow([s.timestamp_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz])
        if self._timebase:
            info = dict(self._timebase)
            info["csv"] = str(path)
            info["samples"] = len(self.samples)
            if self.stats:
                info["span_s"] = self.stats.get("span_s")
                info["timestamp_anchor"] = self.stats.get("timestamp_anchor")
            (path.parent / "imu_timebase.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def verdict(self):
        return verdict(self.stats or self._finish_stats())


def verdict(s):
    """把统计 dict 翻译成 (level, 说明)。level ∈ {pass, warn, fail}。"""
    if s.get("error"):
        return "fail", f"串口出错：{s['error']}"
    if s["samples"] < 100:
        return "fail", f"只收到 {s['samples']} 个样本，太少了 —— 飞控没在发 HIGHRES_IMU？"
    if s["mixed_instances"]:
        return "fail", (f"记录中途 IMU 实例变了 {s['instances']} —— 标定程序会拒绝，"
                        "请固定同一个实例后再采")
    clamped = s.get("timestamps_clamped", 0)
    if clamped > max(10, 0.01 * s["samples"]):
        return "warn", (f"有 {clamped}/{s['samples']} 个样本的时间戳因乱序被强制递增"
                        f"（占 {100 * clamped / s['samples']:.1f}%）。"
                        "集中在开头是正常的积压突发；占比高说明串口读得断续，"
                        "换 USB 口/线，或把 IMU_REQUEST_HZ 降下来")
    if s.get("timestamp_anchor") == "read_time":
        reason = (s.get("timebase") or {}).get("reason") or "飞控没发 HIGHRES_IMU.time_usec"
        return "warn", (f"时间戳只能按串口读时间打（{reason}）：跨录制的相机-IMU 时移"
                        "会漂 10 ms 量级，标定时会被当成真实时移。查一下飞控固件是否在"
                        "HIGHRES_IMU 里发 time_usec")
    if s["gaps_over_150ms"]:
        return "warn", (f"有 {s['gaps_over_150ms']} 处超过 150 ms 的空洞。"
                        "用 configs/probe.yaml（max_imu_gap_s=0.15）仍可能被拒，"
                        "建议降低请求速率或换 USB 口/线")
    if s["observed_hz"] < 40:
        return "warn", (f"实测只有 {s['observed_hz']:.1f} Hz。50 Hz 能标，但平移外参精度"
                        "会从 ~3 mm 掉到 ~5 mm（实测）。想更好就提高 IMU_REQUEST_HZ "
                        "或把 SCHED_LOOP_RATE 调高")
    if s["gaps_over_50ms"]:
        return "warn", (f"实测 {s['observed_hz']:.1f} Hz，有 {s['gaps_over_50ms']} 处 >50 ms 空洞。"
                        "必须用 configs/probe.yaml（max_imu_gap_s=0.15），"
                        "否则默认的 0.05 会让标定直接拒绝")
    med = s.get("gyro_med_deg_s", 0.0)
    if med < 2.0:
        return "fail", (f"录制期间陀螺中位只有 {med:.1f}°/s —— 你基本上没动。"
                        "标定需要多轴转动（门槛是约 2.9°/s 的标准差），"
                        "静静举着是估不出外参的")
    if med > 60.0:
        return "warn", (f"录制期间陀螺中位 {med:.1f}°/s、P95 {s.get('gyro_p95_deg_s', 0):.0f}°/s "
                        "—— 转得偏快，容易运动模糊。建议压低到 10~20°/s")
    if s.get("gyro_p95_deg_s", 0) > 120.0:
        return "warn", (f"陀螺峰值 P95 = {s['gyro_p95_deg_s']:.0f}°/s —— 有猛甩动作，"
                        "那几帧会糊掉并在窗口里凿出空洞。慢一点、匀一点")
    return "pass", f"实测 {s['observed_hz']:.1f} Hz，最大空洞 {s['max_gap_ms']:.1f} ms，干净"


def _format_timebase(stats) -> str:
    tb = stats.get("timebase") or {}
    if not tb:
        return "无锚定信息（旧数据）"
    if stats.get("timestamp_anchor") == "fc_usec" and tb.get("ok"):
        return (f"飞控 time_usec 下包络锚定（{tb.get('bins', 0)} 个包络箱，"
                f"漂移 {tb.get('drift_ppm', 0):+.1f} ppm；"
                f"读延迟中位 {tb.get('delay_median_ms', 0):.1f} ms / P95 {tb.get('delay_p95_ms', 0):.1f} ms；"
                f"读时间戳被回拨中位 {tb.get('shift_median_ms', 0):.1f} ms、最大 {tb.get('shift_max_ms', 0):.1f} ms）")
    reason = tb.get("reason") or "无飞控时间"
    return f"⚠️ 回退到串口读时间（{reason}）—— 跨录制的相机-IMU 时移会漂"


def format_stats(stats) -> str:
    return (
        f"  串口            {stats['port']}\n"
        f"  请求速率        {stats['requested_hz']:g} Hz\n"
        f"  实测速率        {stats['observed_hz']:.1f} Hz（中位间隔 {stats['median_gap_ms']:.2f} ms）\n"
        f"  样本 / 时长     {stats['samples']} 个 / {stats['span_s']:.2f} s\n"
        f"  HIGHRES_IMU    收到 {stats['frames_received']} 帧，"
        f"重复丢弃 {stats['duplicates_skipped']}，解码失败 {stats['decode_failures']}\n"
        f"  时间基准        {_format_timebase(stats)}\n"
        f"  时间戳插值      {stats['timestamps_interpolated']} 个样本用了批内插值"
        f"（其中 {stats['timestamps_clamped']} 个因乱序被强制递增）\n"
        f"  最大空洞        {stats['max_gap_ms']:.1f} ms"
        f"（>50ms: {stats['gaps_over_50ms']}, >150ms: {stats['gaps_over_150ms']}）\n"
        f"  IMU 实例        {stats['instances']}"
        f"{'  ⚠️ 混用了多个实例' if stats['mixed_instances'] else ''}\n"
        f"  转动量 |ω|      中位 {stats.get('gyro_med_deg_s', 0):.1f} °/s"
        f"，P95 {stats.get('gyro_p95_deg_s', 0):.0f} °/s"
        f"，各轴标准差 {stats.get('gyro_axis_std_deg_s', 0):.1f} °/s  ← 你到底动了没有\n"
        f"  fields_updated  全零的帧 {stats['fields_updated_zero']}\n"
        f"  MAVLink         {stats['parser']}"
    )


def _main():
    import extrinsics_config as cfg

    seconds = float(getattr(cfg, "IMU_CHECK_SECONDS", 10.0))
    print("=" * 74)
    print(f"IMU 单独体检 —— 录 {seconds:g} 秒，不采图")
    print("=" * 74)

    port = cfg.IMU_PORT
    if port == "auto":
        port, report = find_mavlink_port(baud=cfg.IMU_BAUD)
        if port is None:
            print("\n❌ 没找到在讲 MAVLink 的串口。看到的候选：")
            for k, v in report.items():
                print(f"     {k}: {v}")
            print("\n  检查：飞控通电了吗？USB 线是数据线不是充电线吗？"
                  "换一个 ttyACM 试试（在 extrinsics_config.py 里写死 IMU_PORT）。")
            return 2
    rec = ImuRecorder(port, request_hz=cfg.IMU_REQUEST_HZ, baud=cfg.IMU_BAUD,
                      startup_discard_s=cfg.IMU_STARTUP_DISCARD_S)
    print(f"打开 {port} ……")
    rec.start()
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        print("\n提前结束")
    stats = rec.stop()

    print("\n采集统计")
    print(format_stats(stats))
    level, text = rec.verdict()
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[level]
    print(f"\n{icon} 结论：{text}")
    if level != "fail":
        out = rec.write_csv(HERE / "imu_check.csv")
        print(f"样本已写出：{out}")
    return 0 if level != "fail" else 2


if __name__ == "__main__":
    sys.exit(_main())
