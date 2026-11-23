"""采集链路的离线自检（原 capture/selftest_capture.py），无需插相机或飞控即可运行。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import pty
import statistics
import tempfile
import termios
import threading
import time
import unittest

import cv2
import numpy as np

from calib.data import Camera, image_manifest, read_imu_csv, read_yaml
from capture import image_health as ih
from capture import imu_recorder
from capture import record_session
from capture.mavlink_min import MavlinkParser, pack

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "capture" / "testdata" / "raw_ttyACM0.bin"


def _fake_gyro(i, hz):
    t = i / hz
    w = 2 * math.pi * 0.4
    return (0.22 * math.sin(w * t),
            0.18 * math.sin(1.3 * w * t + 1.0),
            0.14 * math.sin(0.7 * w * t + 2.0))


def _configure_raw(fd):
    a = termios.tcgetattr(fd)
    a[0] = 0
    a[1] = 0
    a[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    a[3] = 0
    a[4] = a[5] = termios.B115200
    termios.tcsetattr(fd, termios.TCSANOW, a)


def _fake_fc(master_fd, stop, *, hz, instance, repeat_every):
    time.sleep(0.15)
    period = 1.0 / hz
    t0 = 1_000_000_000
    i = 0
    seq = 0
    started = time.monotonic()
    sent_usec = t0
    try:
        os.write(master_fd, pack(0, {"type": 2, "autopilot": 3, "mavlink_version": 3},
                                 sysid=1, compid=1, seq=0))
    except BlockingIOError:
        pass
    while not stop.is_set():
        try:
            if not (repeat_every and i % repeat_every == repeat_every - 1):
                sent_usec = t0 + int((time.monotonic() - started) * 1e6)
            os.write(master_fd, pack(105, {
                "time_usec": sent_usec,
                "xacc": 0.01 * i, "yacc": -0.02 * i, "zacc": 9.80665,
                "xgyro": _fake_gyro(i, hz)[0], "ygyro": _fake_gyro(i, hz)[1],
                "zgyro": _fake_gyro(i, hz)[2],
                "fields_updated": 0x3F, "id": instance}, sysid=1, compid=1, seq=seq & 0xFF))
            seq += 1
        except BlockingIOError:
            pass
        i += 1
        time.sleep(period)


def _fake_fc_backlog(master_fd, stop, *, backlog, period_us, stream_seconds=1.2):
    time.sleep(0.15)
    usec = 1_000_000_000
    seq = 0
    blob = b""
    for _ in range(backlog):
        blob += pack(105, {
            "time_usec": usec,
            "xacc": 0.0, "yacc": 0.0, "zacc": 9.80665,
            "xgyro": _fake_gyro(seq, 200.0)[0], "ygyro": _fake_gyro(seq, 200.0)[1],
            "zgyro": _fake_gyro(seq, 200.0)[2],
            "fields_updated": 0x3F, "id": 0}, sysid=1, compid=1, seq=seq & 0xFF)
        usec += period_us
        seq += 1
    while blob and not stop.is_set():
        try:
            blob = blob[os.write(master_fd, blob):]
        except BlockingIOError:
            time.sleep(0.001)
    deadline = time.monotonic() + stream_seconds
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            os.write(master_fd, pack(105, {
                "time_usec": usec,
                "xacc": 0.0, "yacc": 0.0, "zacc": 9.80665,
                "xgyro": _fake_gyro(seq, 200.0)[0], "ygyro": _fake_gyro(seq, 200.0)[1],
                "zgyro": _fake_gyro(seq, 200.0)[2],
                "fields_updated": 0x3F, "id": 0}, sysid=1, compid=1, seq=seq & 0xFF))
            seq += 1
        except BlockingIOError:
            pass
        usec += period_us
        time.sleep(period_us / 1e6)


def _tiny_png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.full((8, 8), value, dtype="uint8"))[1].tofile(str(path))


class CapturePipelineTests(unittest.TestCase):
    def test_mavlink_parser(self):
        raw = TESTDATA.read_bytes()
        parser = MavlinkParser()
        frames = parser.feed(raw)
        names = sorted({f.name for f in frames})
        self.assertGreater(len(frames), 50)
        self.assertEqual(parser.stats["crc_bad"], 0)
        self.assertEqual(parser.stats["unknown_msgid"], 0)
        self.assertIn("HEARTBEAT", names)

        payload = {"time_usec": 123456789, "xacc": 0.5, "yacc": -1.25, "zacc": 9.81,
                   "xgyro": 0.01, "ygyro": -0.02, "zgyro": 0.03,
                   "fields_updated": 0x3F, "id": 0}
        back = MavlinkParser().feed(pack(105, payload, sysid=1, compid=1, seq=0))
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].name, "HIGHRES_IMU")
        f = back[0].fields
        self.assertAlmostEqual(f["zacc"], 9.81, places=4)
        self.assertEqual(f["fields_updated"], 0x3F)
        self.assertEqual(int(f["time_usec"]), 123456789)

        parser = MavlinkParser()
        wire = pack(105, payload, sysid=1, compid=1, seq=1, version=1)
        ack = pack(77, {"command": 511, "result": 0}, sysid=1, compid=1, seq=2, version=1)
        self.assertFalse(parser.feed(wire[:12]))
        frames = parser.feed(wire[12:] + ack + pack(0, {"mavlink_version": 3}, seq=3))
        self.assertEqual([(frame.name, frame.version) for frame in frames],
                         [("HIGHRES_IMU", 1), ("COMMAND_ACK", 1), ("HEARTBEAT", 2)])

    def test_imu_recorder_end_to_end(self):
        hz, seconds, instance, repeat_every = 200.0, 1.5, 0, 5
        master, slave = pty.openpty()
        slave_path = os.ttyname(slave)
        _configure_raw(slave)
        os.set_blocking(master, False)
        stop = threading.Event()
        writer = threading.Thread(target=_fake_fc, args=(master, stop),
                                  kwargs={"hz": hz, "instance": instance, "repeat_every": repeat_every},
                                  daemon=True)
        writer.start()
        try:
            rec = imu_recorder.ImuRecorder(slave_path, request_hz=hz)
            rec.start()
            time.sleep(seconds)
            stats = rec.stop()
        finally:
            stop.set()
            writer.join(timeout=2)
            os.close(master)
            os.close(slave)

        unique = stats["frames_received"] - stats["duplicates_skipped"]
        expected_hz = unique / stats["span_s"] if stats["span_s"] else 0.0
        self.assertGreater(stats["samples"], 80)
        self.assertEqual(stats["samples"], unique)
        self.assertLess(abs(stats["observed_hz"] - expected_hz), expected_hz * 0.05)
        self.assertGreater(stats["duplicates_skipped"], 0)
        stamps_list = [s.timestamp_ns for s in rec.samples]
        self.assertTrue(all(b > a for a, b in zip(stamps_list, stamps_list[1:])))
        self.assertEqual(stats["instances"], [instance])
        self.assertFalse(stats["mixed_instances"])
        self.assertEqual(stats.get("timestamp_anchor"), "fc_usec")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = rec.write_csv(Path(tmp) / "imu.csv")
            stamps, values = read_imu_csv(csv_path, "ns")
            self.assertEqual(len(stamps), len(rec.samples))
            self.assertLess(abs(statistics.median(abs(v[2]) for v in values) - 9.8), 0.1)

    def test_recorder_rejects_mixed_instances_and_burst_backlog(self):
        master, slave = pty.openpty()
        slave_path = os.ttyname(slave)
        _configure_raw(slave)
        os.set_blocking(master, False)
        stop = threading.Event()
        writer = threading.Thread(target=_fake_fc_backlog, args=(master, stop),
                                  kwargs={"backlog": 60, "period_us": 5000}, daemon=True)
        writer.start()
        try:
            rec = imu_recorder.ImuRecorder(slave_path, request_hz=200.0)
            rec.start()
            time.sleep(1.5)
            stats = rec.stop()
        finally:
            stop.set()
            writer.join(timeout=2)
            os.close(master)
            os.close(slave)

        ts = [s.timestamp_ns for s in rec.samples]
        self.assertGreater(stats["timestamps_clamped"], 0)
        self.assertEqual(stats.get("timestamp_anchor"), "read_time")
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))

    def test_px4_silent_until_gcs_heartbeat(self):
        """PX4 USB CDC stays silent until it receives a GCS HEARTBEAT."""
        import select as _select

        def _fake_px4_fc(master_fd, stop):
            rx = MavlinkParser()
            awake = False
            seq = 0
            t0 = 2_000_000_000
            started = time.monotonic()
            while not stop.is_set():
                r, _, _ = _select.select([master_fd], [], [], 0.02)
                if r:
                    try:
                        data = os.read(master_fd, 4096)
                    except BlockingIOError:
                        data = b""
                    for f in rx.feed(data):
                        if f.name == "HEARTBEAT":
                            awake = True
                if awake:
                    try:
                        os.write(master_fd, pack(0, {"type": 2, "autopilot": 12, "mavlink_version": 3},
                                                 sysid=1, compid=1, seq=seq & 0xFF))
                        seq += 1
                        os.write(master_fd, pack(105, {
                            "time_usec": t0 + int((time.monotonic() - started) * 1e6),
                            "xacc": 0.0, "yacc": 0.0, "zacc": -9.81,
                            "xgyro": 0.0, "ygyro": 0.0, "zgyro": 0.0,
                            "fields_updated": 0x3F, "id": 0,
                        }, sysid=1, compid=1, seq=seq & 0xFF))
                        seq += 1
                    except BlockingIOError:
                        pass
                    time.sleep(0.01)

        # 1) find_mavlink_port wakes it up
        master, slave = pty.openpty()
        slave_path = os.ttyname(slave)
        _configure_raw(slave)
        os.set_blocking(master, False)
        stop = threading.Event()
        th = threading.Thread(target=_fake_px4_fc, args=(master, stop), daemon=True)
        th.start()
        try:
            found, report = imu_recorder.find_mavlink_port(
                candidates=[Path(slave_path)], timeout=1.2, log=lambda *a: None)
            self.assertEqual(found, slave_path)
            self.assertIn("GCS 心跳唤醒", report[slave_path])
        finally:
            stop.set()
            th.join(timeout=2)
            os.close(master)
            os.close(slave)

        # 2) ImuRecorder directly on the port also wakes it up
        master, slave = pty.openpty()
        slave_path = os.ttyname(slave)
        _configure_raw(slave)
        os.set_blocking(master, False)
        stop = threading.Event()
        th = threading.Thread(target=_fake_px4_fc, args=(master, stop), daemon=True)
        th.start()
        try:
            rec = imu_recorder.ImuRecorder(slave_path, request_hz=100.0)
            rec.start(wait_first_s=2.0)
            time.sleep(0.3)
            stats = rec.stop()
            self.assertGreater(stats["samples"], 5)
        finally:
            stop.set()
            th.join(timeout=2)
            os.close(master)
            os.close(slave)

    def test_gyro_motion_verdict(self):
        base = {"error": "", "samples": 1000, "mixed_instances": False, "instances": [0],
                "timestamps_clamped": 0, "gaps_over_150ms": 0, "gaps_over_50ms": 0,
                "observed_hz": 200.0, "max_gap_ms": 5.0, "gyro_p95_deg_s": 25.0}
        self.assertEqual(imu_recorder.verdict(dict(base, gyro_med_deg_s=1.0, gyro_p95_deg_s=25.0))[0], "fail")
        self.assertEqual(imu_recorder.verdict(dict(base, gyro_med_deg_s=15.0, gyro_p95_deg_s=25.0))[0], "pass")
        self.assertEqual(imu_recorder.verdict(dict(base, gyro_med_deg_s=90.0, gyro_p95_deg_s=110.0))[0], "warn")
        self.assertEqual(imu_recorder.verdict(dict(base, gyro_med_deg_s=15.0, gyro_p95_deg_s=200.0))[0], "warn")

    def test_manifest_to_images_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "session"
            (session / "cam0").mkdir(parents=True)
            stamps = [1_700_000_000_000_000_000 + i * 33_333_333 for i in range(6)]
            saved = []
            for i, stamp in enumerate(stamps):
                rel = f"cam0/{stamp}.png"
                _tiny_png(session / rel, i * 30)
                saved.append({"file": rel, "host_receive_time_ns": stamp, "board_detected": i % 2 == 0})
            (session / "capture_manifest.json").write_text(json.dumps(
                {"mode": {"width": 1920, "height": 1080}, "termination": "duration_reached",
                 "saved": saved}), encoding="utf-8")

            stats = record_session.build_images_csv(session, log=lambda *a: None)
            self.assertTrue(Path(stats["csv"]).is_file())
            self.assertEqual(stats["images"], 6)
            self.assertEqual(stats["board_detected"], 3)
            pairs = image_manifest(stats["csv"], "ns")
            self.assertEqual([s for s, _ in pairs], stamps)

            bad = Path(tmp) / "bad"
            (bad / "cam0").mkdir(parents=True)
            (bad / "capture_manifest.json").write_text(json.dumps(
                {"saved": [{"file": "cam0/nothere.png", "host_receive_time_ns": 1}]}), encoding="utf-8")
            with self.assertRaises(record_session.ManifestError):
                record_session.build_images_csv(bad, log=lambda *a: None)

            _tiny_png(bad / "cam0" / "nothere.png", 0)
            (bad / "capture_manifest.json").write_text(json.dumps(
                {"saved": [{"file": "cam0/nothere.png", "host_receive_time_ns": 500},
                           {"file": "cam0/nothere.png", "host_receive_time_ns": 100}]}), encoding="utf-8")
            with self.assertRaises(record_session.ManifestError):
                record_session.build_images_csv(bad, log=lambda *a: None)

    def test_configs_and_image_health(self):
        cam = Camera.load(ROOT / "configs" / "gemini_pro_rgb.yaml")
        self.assertEqual(tuple(cam.size), (1920, 1080))
        target = read_yaml(ROOT / "configs" / "checkerboard.yaml")
        self.assertEqual(target.get("target_type"), "checkerboard")

        def bright(p90):
            return {"p50": p90 * 0.4, "p90": p90, "p99": p90 * 1.5, "mean": p90 * 0.5, "n": 8}

        self.assertEqual(ih.brightness_verdict(bright(41))[0], "dark")
        self.assertEqual(ih.brightness_verdict(bright(90))[0], "ok")
        self.assertEqual(ih.brightness_verdict(bright(250))[0], "bright")
        self.assertEqual(ih.brightness_verdict(None)[0], "unknown")

        silent = lambda *a, **k: None  # noqa: E731
        det_ok = {"tried": 12, "detected": 10, "rate": 10 / 12, "kinds": ["SB"],
                  "min_edge_margin_px": 152.0, "shape": (1080, 1920), "subgrid": {},
                  "sharp_median": 20.0, "sharp_hit": 21.0, "sharp_miss": 15.0}
        real = {"images": 30, "board": "7x7",
                "brightness": {"p50": 20, "p90": 43, "p99": 64, "mean": 22.5, "n": 8},
                "detection": det_ok}
        self.assertEqual(ih.report(real, log=silent)[0], "ok")

        blurry = dict(real, quality={"window": 24, "detected": 16, "accepted": 14, "rejected": 2,
                                     "effective_rate": 14 / 24, "pnp_bad_median": 2.95,
                                     "sharp_ok": 27.0, "sharp_bad": 14.0})
        lvl, acts = ih.report(blurry, log=silent)
        self.assertEqual(lvl, "warn")
        self.assertTrue(any("糊" in a for a in acts))


if __name__ == "__main__":
    unittest.main()
