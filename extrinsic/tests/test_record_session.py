import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibrate_extrinsics
from capture import record_session
from capture.vendor import capture_calibration
from calib.data import Camera, Target, read_yaml


class RecordSessionTests(unittest.TestCase):
    def test_camera_receives_purpose_specific_exposure_rate_and_limits(self):
        cfg = record_session.cfg
        camera = Camera.load(cfg.CAMERA_YAML)
        self.assertEqual(tuple(camera.size), (cfg.WIDTH, cfg.HEIGHT))
        for purpose, rate in (("calibration", cfg.CALIBRATION_AUTO_EXPOSE_EXPECTED_RATE_DPS),
                              ("vio", cfg.AUTO_EXPOSE_EXPECTED_RATE_DPS)):
            with self.subTest(purpose=purpose), patch.multiple(cfg, CAPTURE_PURPOSE=purpose,
                                                               ALLOW_DARK=False, SKIP_EXPOSURE_CHECK=False):
                command = record_session.camera_command("/dev/video4", {"cols": 6, "rows": 6})
                for flag, value in {"--width": cfg.WIDTH, "--height": cfg.HEIGHT, "--fps": cfg.FPS,
                                    "--auto-expose-expected-rate-dps": rate,
                                    "--auto-expose-blur-budget-px": cfg.AUTO_EXPOSE_BLUR_BUDGET_PX,
                                    "--auto-expose-max-gain": cfg.AUTO_EXPOSE_MAX_GAIN,
                                    "--auto-expose-min-mean": cfg.AUTO_EXPOSE_MIN_MEAN,
                                    "--focal-px": float(camera.params[0])}.items():
                    self.assertEqual(command[command.index(flag) + 1], str(value))
                self.assertEqual("--allow-below-exposure-target" in command, purpose == "calibration")
                self.assertNotIn("--allow-dark", command)
                self.assertNotIn("--skip-exposure-check", command)

    def test_exposure_target_and_hard_floor(self):
        def control_range(fd, control):
            if control == capture_calibration.CONTROLS["exposure"]:
                return {"min": 1, "max": 5000, "step": 1}
            return {"min": 0, "max": 100, "step": 1}

        for purpose, mean, expected in (("calibration", 85.6, (True, "below_target")),
                                        ("calibration", 12.6, (False, "too_dark")),
                                        ("vio", 85.6, (False, "insufficient_light"))):
            with self.subTest(purpose=purpose, mean=mean), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(record_session.cfg, "CAPTURE_PURPOSE", purpose))
                stack.enter_context(patch.object(capture_calibration, "query_ctrl", side_effect=control_range))
                stack.enter_context(patch.object(capture_calibration, "_set_raw",
                                                 side_effect=lambda fd, name, value: value))
                stack.enter_context(patch.object(capture_calibration, "_gray_mean",
                                                 side_effect=[mean - 8 + step for step in range(9)]))
                command = record_session.camera_command("/dev/video4", {"cols": 6, "rows": 6})
                args = capture_calibration.arguments(command[2:])
                ok, evidence = capture_calibration.auto_expose(None, None, args, log=lambda message: None)
                self.assertEqual((ok, evidence["status"]), expected)

    def test_solver_arguments_for_both_boards(self):
        for board in ("checkerboard", "aprilgrid"):
            with self.subTest(board=board):
                cfg = record_session.cfg
                target_yaml = cfg.ROOT / "configs" / f"{board}.yaml"
                with patch.object(cfg, "TARGET_YAML", target_yaml):
                    args = calibrate_extrinsics.parser().parse_args(
                        record_session.calibration_command()[2:])
                self.assertEqual(args.target, target_yaml)
                self.assertEqual(Target(read_yaml(args.target)).kind, board)
                self.assertEqual(args.camera, cfg.CAMERA_YAML)
                self.assertEqual(args.images, cfg.SESSION_DIR / "images.csv")
                self.assertEqual(args.imu, cfg.SESSION_DIR / "imu.csv")
                self.assertEqual(args.output, cfg.RESULTS_DIR)
                self.assertEqual((args.image_unit, args.imu_unit), ("ns", "ns"))
                self.assertIsNone(args.settings)
                self.assertIsNone(args.duration)

    def test_main_only_solves_successful_calibration_capture(self):
        for purpose, auto, cam_rc, interrupted, termination, imu_level, expected in (
            ("calibration", True, 0, False, "completed", "pass", 7),
            ("calibration", True, 0, True, "completed", "pass", 1),
            ("calibration", True, None, False, "completed", "pass", 1),
            ("calibration", True, 0, False, "failed: camera lost", "pass", 1),
            ("calibration", True, 0, False, "completed", "fail", 1),
            ("calibration", False, 0, False, "completed", "pass", 0),
            ("vio", True, 0, False, "completed", "pass", 0),
        ):
            with self.subTest(purpose=purpose, auto=auto, cam_rc=cam_rc, interrupted=interrupted,
                              termination=termination, imu_level=imu_level):
                stats = {"images": 100, "span_s": 6.0, "observed_fps": 20.0,
                         "max_gap_ms": 50.0, "gaps_over_100ms": 0, "gaps_over_350ms": 0,
                         "board_detected": 0, "manifest_termination": termination}
                with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                    session = Path(tmp)
                    (session / "capture_manifest.json").write_text("{}")
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(patch.multiple(record_session.cfg, SESSION_DIR=session,
                                                       CAPTURE_PURPOSE=purpose, RUN_CALIBRATION_AFTER=auto))
                    stack.enter_context(patch.object(record_session, "preflight", return_value=(None, None, {})))
                    stack.enter_context(patch.object(record_session, "record", return_value=({}, cam_rc, interrupted)))
                    stack.enter_context(patch.object(record_session.make_images_csv, "build", return_value=stats))
                    stack.enter_context(patch.object(record_session.imu_recorder, "format_stats", return_value=""))
                    stack.enter_context(patch.object(record_session.imu_recorder, "verdict",
                                                     return_value=(imu_level, "")))
                    solve = stack.enter_context(patch.object(record_session, "run_calibration", return_value=7))
                    self.assertEqual(record_session.main(), expected)
                    self.assertEqual(solve.call_count, int(expected == 7))


if __name__ == "__main__":
    unittest.main()
