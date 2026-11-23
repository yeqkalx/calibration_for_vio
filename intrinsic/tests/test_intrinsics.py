"""棋盘格与 AprilGrid 内参标定及采集预览离线测试。

运行方式（在 vins conda 环境中）:
    cd intrinsic
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

INTRINSIC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = INTRINSIC_DIR.parent
if str(INTRINSIC_DIR) not in sys.path:
    sys.path.insert(0, str(INTRINSIC_DIR))

import calibrate_intrinsics as calib  # noqa: E402
import capture_calibration as cap_mod  # noqa: E402
from target import CalibrationTarget, DuplicateTagError  # noqa: E402


def _draw_aruco_marker(dictionary, tag_id: int, size_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, size_px)
    return cv2.aruco.drawMarker(dictionary, tag_id, size_px)


def _render_board_texture_checkerboard(cols: int, rows: int, square_size: float, ppm: float = 2500.0):
    """生成高分辨率棋盘格纹理及其从板面 (X, Y) 到纹理像素 (tx, ty) 的映射参数。"""
    sq_px = int(round(square_size * ppm))
    margin_m = square_size * 0.8
    margin_px = int(round(margin_m * ppm))
    n_sq_x, n_sq_y = cols + 1, rows + 1
    tex_w = n_sq_x * sq_px + 2 * margin_px
    tex_h = n_sq_y * sq_px + 2 * margin_px
    tex = np.full((tex_h, tex_w), 240, dtype=np.uint8)
    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 0:
                y0 = margin_px + r * sq_px
                x0 = margin_px + c * sq_px
                tex[y0:y0 + sq_px, x0:x0 + sq_px] = 20
    # 内角点 (0, 0) 位于第 1 个格子右下角边缘（像素 margin_px + sq_px - 1 与 margin_px + sq_px 的交界处）
    origin_x = margin_px + sq_px - 0.5
    origin_y = margin_px + sq_px - 0.5
    # 棋盘格 board_points 中 X 向右、Y 向下（或平面内正交）
    return tex, (origin_x, origin_y, ppm, +1.0)


def _render_board_texture_aprilgrid(
    tag_cols: int,
    tag_rows: int,
    tag_size: float,
    tag_spacing: float,
    *,
    first_id: int = 0,
    tag_rotation_deg: int = 180,
    ppm: float = 2500.0,
):
    """生成高分辨率 Kalibr AprilGrid 纹理（board +X 向右，+Y 向上）。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # tag36h11 为 8x8 码元（含 1-bit 黑框）；选用 160 的倍数可同时让常见 tagSpacing（如 0.35）的 pitch 为精确整数
    tag_px = max(160, int(round(tag_size * ppm / 160.0)) * 160)
    ppm = tag_px / tag_size
    pitch_m = tag_size * (1.0 + tag_spacing)
    board_w_m = (tag_cols - 1) * pitch_m + tag_size
    board_h_m = (tag_rows - 1) * pitch_m + tag_size
    margin_m = tag_size * 0.6
    margin_px = int(round(margin_m * ppm))
    tex_w = int(round(board_w_m * ppm)) + 2 * margin_px
    tex_h = int(round(board_h_m * ppm)) + 2 * margin_px
    tex = np.full((tex_h, tex_w), 240, dtype=np.uint8)

    # 板面原点 (X=0, Y=0) 对应左下角 tag 的左下角外边界（像素中心坐标下边界位于 idx - 0.5）
    origin_x = margin_px - 0.5
    origin_y = margin_px + round(board_h_m * ppm) - 0.5
    k_rot = (tag_rotation_deg // 90) % 4

    for row in range(tag_rows):
        for col in range(tag_cols):
            tag_id = first_id + row * tag_cols + col
            marker = _draw_aruco_marker(dictionary, tag_id, tag_px)
            if k_rot:
                marker = np.rot90(marker, k_rot)
            x_m = col * pitch_m
            y_top_m = row * pitch_m + tag_size
            x0 = int(round(origin_x + 0.5 + x_m * ppm))
            y0 = int(round(origin_y + 0.5 - y_top_m * ppm))
            tex[y0:y0 + tag_px, x0:x0 + tag_px] = marker

    return tex, (origin_x, origin_y, ppm, -1.0)


def _render_view(
    tex: np.ndarray,
    tex_meta: tuple[float, float, float, float],
    image_size: tuple[int, int],
    k_mat: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    model: str = "pinhole",
) -> np.ndarray:
    """用精确逆投影 + cv2.remap 渲染畸变相机视图（支持 pinhole+radtan 与 fisheye）。"""
    w, h = image_size
    origin_x, origin_y, ppm, y_sign = tex_meta
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    if model == "fisheye":
        undist = cv2.fisheye.undistortPoints(pts, k_mat, dist.reshape(4, 1)).reshape(-1, 2)
    else:
        undist = cv2.undistortPoints(pts, k_mat, dist.reshape(-1)).reshape(-1, 2)

    rays_c = np.concatenate([undist, np.ones((len(undist), 1), dtype=np.float64)], axis=1)  # (HW, 3)
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    cam_b = -rot.T @ t                                   # 相机光心在板面坐标
    rays_b = rays_c @ rot                                # d_b = R^T * d_c
    denom = rays_b[:, 2]
    lam = np.where(np.abs(denom) > 1e-8, -cam_b[2] / np.where(np.abs(denom) > 1e-8, denom, 1.0), -1.0)
    hit_x = cam_b[0] + lam * rays_b[:, 0]
    hit_y = cam_b[1] + lam * rays_b[:, 1]
    map_x = np.where(lam > 0, origin_x + hit_x * ppm, -10.0).reshape(h, w).astype(np.float32)
    map_y = np.where(lam > 0, origin_y + y_sign * hit_y * ppm, -10.0).reshape(h, w).astype(np.float32)
    return cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=175)


def _pose_looking_at(board_center: np.ndarray, euler_deg: tuple[float, float, float],
                     shift_cam: tuple[float, float], dist_z: float):
    """构造相机看到板面的 (rvec, tvec)：先绕板中心旋转，再放到相机前方 dist_z 处并平移。"""
    rx, ry, rz = np.deg2rad(euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # 让板面法向 (+Z_b) 与相机光轴 (+Z_c) 一致（对 AprilGrid 还需要 Y_b 向上对应图像 -v）
    rot = rot_z @ rot_y @ rot_x
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.array([shift_cam[0], shift_cam[1], dist_z], dtype=np.float64) - rot @ board_center
    return rvec, tvec


def _aprilgrid_pose(board_center: np.ndarray, euler_deg: tuple[float, float, float],
                    shift_cam: tuple[float, float], dist_z: float):
    """AprilGrid 板面 +Y 向上，需要绕 X 轴转 180° 使 +Y_b 指向图像上方 (-v) 且 Z_c > 0。"""
    rx, ry, rz = np.deg2rad(euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    base = np.diag([1.0, -1.0, -1.0])  # X_c = X_b, Y_c = -Y_b, Z_c = -Z_b
    rot = rot_z @ rot_y @ rot_x @ base
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.array([shift_cam[0], shift_cam[1], dist_z], dtype=np.float64) - rot @ board_center
    return rvec, tvec


class IntrinsicTargetTests(unittest.TestCase):
    def test_load_existing_extrinsic_target_yamls(self):
        """可直接复用 extrinsic/configs/{aprilgrid,checkerboard}.yaml。"""
        april_yaml = REPO_ROOT / "extrinsic" / "configs" / "aprilgrid.yaml"
        checker_yaml = REPO_ROOT / "extrinsic" / "configs" / "checkerboard.yaml"

        t_april = CalibrationTarget.from_yaml(april_yaml)
        self.assertEqual(t_april.kind, "aprilgrid")
        self.assertEqual((t_april.cols, t_april.rows), (6, 6))
        self.assertAlmostEqual(t_april.tag_size_m, 0.088)
        self.assertAlmostEqual(t_april.tag_spacing, 0.56)
        self.assertEqual(t_april.tag_rotation_deg, 180)

        t_checker = CalibrationTarget.from_yaml(checker_yaml)
        self.assertEqual(t_checker.kind, "checkerboard")
        self.assertEqual((t_checker.cols, t_checker.rows), (8, 11))
        self.assertAlmostEqual(t_checker.col_spacing_m, 0.03)

        # CLI 解析两端入口都能直接接受 --target
        args_cal = calib.parse_args(["--target", str(april_yaml)])
        self.assertEqual(args_cal.target_spec.kind, "aprilgrid")
        args_cap = cap_mod.arguments(["--target", str(april_yaml), "--no-preview", "--duration", "5"])
        self.assertEqual(args_cap.target_spec.kind, "aprilgrid")

    def test_kalibr_checkerboard_yaml_and_scientific_notation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "kalibr_checker.yaml"
            p.write_text(
                "%YAML:1.0\n"
                "target_type: checkerboard\n"
                "targetCols: 7\n"
                "targetRows: 9\n"
                "colSpacingMeters: 3e-2\n"
                "rowSpacingMeters: 4e-2\n",
                encoding="utf-8",
            )
            tgt = CalibrationTarget.from_yaml(p)
            self.assertEqual(tgt.kind, "checkerboard")
            self.assertEqual((tgt.cols, tgt.rows), (7, 9))
            self.assertAlmostEqual(tgt.col_spacing_m, 0.03)
            self.assertAlmostEqual(tgt.row_spacing_m, 0.04)
            # 验证不同行列间距确实反映在 3D 模板中
            pts = tgt.object_template.reshape(9, 7, 3)
            self.assertAlmostEqual(pts[0, 1, 0], 0.03)
            self.assertAlmostEqual(pts[1, 0, 1], 0.04)

    def test_cli_conflicts_and_invalid_parameters_rejected(self):
        with self.assertRaises(SystemExit):
            calib.parse_args(["--target-type", "aprilgrid", "--cols", "8"])
        with self.assertRaises(SystemExit):
            calib.parse_args(["--target-type", "checkerboard", "--tag-cols", "6"])
        with self.assertRaises(SystemExit):
            calib.parse_args([
                "--target", str(REPO_ROOT / "extrinsic" / "configs" / "aprilgrid.yaml"),
                "--target-type", "checkerboard",
            ])
        with self.assertRaises(SystemExit):
            calib.parse_args(["--target-type", "aprilgrid", "--tag-cols", "2", "--tag-rows", "2", "--min-tags", "5"])
        with self.assertRaises(ValueError):
            CalibrationTarget.from_aprilgrid(4, 4, 0.05, 0.3, tag_rotation_deg=45)
        with self.assertRaises(ValueError):
            CalibrationTarget.from_aprilgrid(4, 4, 0.05, 0.3, first_id=580)

    def test_aprilgrid_all_four_rotations_geometry_and_checker(self):
        """验证 0/90/180/270 四种 tag_rotation_deg 下，3D 映射与方向自检全部精准匹配。"""
        k_mat = np.array([[750.0, 0.0, 400.0], [0.0, 750.0, 300.0], [0.0, 0.0, 1.0]])
        dist = np.array([-0.1, 0.02, 0.0, 0.0])
        for rot_deg in (0, 90, 180, 270):
            with self.subTest(tag_rotation_deg=rot_deg):
                tex, meta = _render_board_texture_aprilgrid(
                    4, 4, 0.05, 0.35, tag_rotation_deg=rot_deg, ppm=2200.0
                )
                center = np.array([0.5 * (3 * 0.05 * 1.35 + 0.05), 0.5 * (3 * 0.05 * 1.35 + 0.05), 0.0])
                rvec, tvec = _aprilgrid_pose(center, (18.0, -14.0, 25.0), (0.01, -0.01), 0.55)
                img = _render_view(tex, meta, (800, 600), k_mat, dist, rvec, tvec, model="pinhole")

                tgt_ok = CalibrationTarget.from_aprilgrid(
                    4, 4, 0.05, 0.35, tag_rotation_deg=rot_deg, min_tags=4
                )
                det = tgt_ok.detect(img)
                self.assertIsNotNone(det)
                self.assertGreaterEqual(det.num_tags, 12)
                check = tgt_ok.check_aprilgrid_rotation([det])
                self.assertEqual(check["best_rotation_deg"], rot_deg)
                self.assertGreater(check["configured_alignment_cos"], 0.95)

                # 用真实的投影验证 3D-2D 对应误差在亚像素级别
                proj, _ = cv2.projectPoints(det.object_points, rvec, tvec, k_mat, dist)
                rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det.image_points.reshape(-1, 2)) ** 2, axis=1))))
                self.assertLess(rms, 0.5)

                # 填错方向必须被 check_aprilgrid_rotation 拦下并指出正确角度
                wrong_rot = (rot_deg + 180) % 360
                tgt_bad = CalibrationTarget.from_aprilgrid(
                    4, 4, 0.05, 0.35, tag_rotation_deg=wrong_rot, min_tags=4
                )
                det_bad = tgt_bad.detect(img)
                with self.assertRaises(ValueError) as ctx:
                    tgt_bad.check_aprilgrid_rotation([det_bad])
                self.assertIn(f"{rot_deg}°", str(ctx.exception))

    def test_aprilgrid_pixel_center_origin_alignment(self):
        """验证 AprilGrid 角点严格对齐 OpenCV 像素中心原点（与棋盘格/Kalibr/OpenVINS 一致，无 +0.5 px 主点偏差）。"""
        tgt = CalibrationTarget.from_aprilgrid(2, 2, 0.05, 0.5, tag_rotation_deg=0, min_tags=1)
        img = np.full((300, 300), 240, dtype=np.uint8)
        img[100:180, 100:180] = _draw_aruco_marker(tgt.dictionary, 0, 80)
        det = tgt.detect(img)
        self.assertIsNotNone(det)
        expected = np.array([[99.5, 99.5], [179.5, 99.5], [179.5, 179.5], [99.5, 179.5]])
        np.testing.assert_allclose(det.image_points.reshape(4, 2), expected, atol=0.15)

    def test_duplicate_tag_id_detected_and_handled_safely(self):
        tgt = CalibrationTarget.from_aprilgrid(3, 3, 0.05, 0.3, tag_rotation_deg=0, min_tags=2)
        img = np.full((420, 640), 235, dtype=np.uint8)
        m0 = _draw_aruco_marker(tgt.dictionary, 0, 90)
        m1 = _draw_aruco_marker(tgt.dictionary, 1, 90)
        # 在同一帧放置两个 ID=0 的 tag
        img[60:150, 60:150] = m0
        img[60:150, 220:310] = m1
        img[60:150, 380:470] = m0
        with self.assertRaises(DuplicateTagError):
            tgt.detect(img)
        det, warn = cap_mod._detect_target(tgt, img)
        self.assertIsNone(det)
        self.assertEqual(warn, "DUP TAG ID!")

    def test_preview_image_rendering_both_boards(self):
        # 棋盘格预览渲染
        tex_cb, meta_cb = _render_board_texture_checkerboard(6, 4, 0.04, ppm=2000.0)
        k = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
        d = np.zeros(4)
        center_cb = np.array([2.5 * 0.04, 1.5 * 0.04, 0.0])
        rvec, tvec = _pose_looking_at(center_cb, (5.0, -5.0, 0.0), (0.0, 0.0), 0.6)
        gray_cb = _render_view(tex_cb, meta_cb, (640, 480), k, d, rvec, tvec)
        tgt_cb = CalibrationTarget.from_checkerboard(6, 4, 0.04)
        det_cb = tgt_cb.detect(gray_cb)
        self.assertIsNotNone(det_cb)
        bgr_cb = cv2.cvtColor(gray_cb, cv2.COLOR_GRAY2BGR)
        out_cb = cap_mod.preview_image(bgr_cb, det_cb, ["line1", "line2", "line3"], {(1, 1)}, target=tgt_cb)
        self.assertEqual(out_cb.shape, bgr_cb.shape)
        # 兼容旧调用：直接传 corners ndarray 且不传 target
        out_legacy = cap_mod.preview_image(bgr_cb, det_cb.image_points, ["line1"], {(1, 1)})
        self.assertEqual(out_legacy.shape, bgr_cb.shape)

        # AprilGrid 预览渲染
        tex_ap, meta_ap = _render_board_texture_aprilgrid(4, 4, 0.05, 0.35, tag_rotation_deg=180, ppm=2000.0)
        center_ap = np.array([0.5 * (3 * 0.05 * 1.35 + 0.05)] * 2 + [0.0])
        rvec_a, tvec_a = _aprilgrid_pose(center_ap, (8.0, -8.0, 0.0), (0.0, 0.0), 0.65)
        gray_ap = _render_view(tex_ap, meta_ap, (640, 480), k, d, rvec_a, tvec_a)
        tgt_ap = CalibrationTarget.from_aprilgrid(4, 4, 0.05, 0.35, tag_rotation_deg=180, min_tags=4)
        det_ap = tgt_ap.detect(gray_ap)
        self.assertIsNotNone(det_ap)
        bgr_ap = cv2.cvtColor(gray_ap, cv2.COLOR_GRAY2BGR)
        out_ap = cap_mod.preview_image(bgr_ap, det_ap, ["line1", "line2", "line3"], {(0, 0), (1, 1)}, target=tgt_ap)
        self.assertEqual(out_ap.shape, bgr_ap.shape)

    def test_end_to_end_checkerboard_pinhole(self):
        """端到端合成测试：棋盘格 + pinhole radtan。"""
        image_size = (960, 720)
        k_true = np.array([[780.0, 0.0, 485.0], [0.0, 784.0, 355.0], [0.0, 0.0, 1.0]])
        d_true = np.array([-0.16, 0.045, 0.0015, -0.001])
        cols, rows, sq = 7, 5, 0.04
        tex, meta = _render_board_texture_checkerboard(cols, rows, sq, ppm=2400.0)
        center = np.array([(cols - 1) * sq * 0.5, (rows - 1) * sq * 0.5, 0.0])

        poses = [
            ((0, 0, 0), (0.0, 0.0), 0.62),
            ((24, -18, 8), (-0.16, -0.10), 0.68),
            ((-22, 20, -10), (0.16, -0.10), 0.68),
            ((22, 18, 12), (-0.16, 0.10), 0.68),
            ((-20, -22, -12), (0.16, 0.10), 0.68),
            ((30, 0, 5), (0.0, -0.12), 0.65),
            ((-30, 0, -5), (0.0, 0.12), 0.65),
            ((0, 30, 6), (-0.18, 0.0), 0.65),
            ((0, -30, -6), (0.18, 0.0), 0.65),
            ((18, 25, 25), (-0.12, -0.08), 0.75),
            ((-18, -25, -25), (0.12, 0.08), 0.75),
            ((25, -20, 35), (0.10, -0.08), 0.72),
            ((-25, 20, -35), (-0.10, 0.08), 0.72),
            ((12, -28, 15), (0.04, 0.05), 0.58),
            ((-12, 28, -15), (-0.04, -0.05), 0.58),
            ((28, 12, -20), (-0.08, 0.09), 0.64),
            ((-28, -12, 20), (0.08, -0.09), 0.64),
            ((15, 15, 45), (0.0, 0.0), 0.70),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "cam0"
            out_dir = root / "intrinsics"
            img_dir.mkdir()
            for idx, (euler, shift, dist_z) in enumerate(poses):
                rvec, tvec = _pose_looking_at(center, euler, shift, dist_z)
                img = _render_view(tex, meta, image_size, k_true, d_true, rvec, tvec, model="pinhole")
                cv2.imwrite(str(img_dir / f"frame_{idx:03d}.png"), img)

            report = calib.main([
                "--images", str(img_dir),
                "--target-type", "checkerboard",
                "--cols", str(cols),
                "--rows", str(rows),
                "--square-size", str(sq),
                "--min-views", "12",
                "--output", str(out_dir),
            ])
            self.assertIn(report["status"], ("passed", "passed_with_warnings"))
            self.assertEqual(report["target_type"], "checkerboard")
            self.assertEqual(report["board_inner_corners"], [cols, rows])
            self.assertLess(report["reprojection_rms_px"], 0.25)

            camchain = yaml.safe_load((out_dir / "camchain_intrinsics.yaml").read_text(encoding="utf-8"))
            intr = camchain["cam0"]["intrinsics"]
            dist_est = camchain["cam0"]["distortion_coeffs"]
            self.assertAlmostEqual(intr[0], k_true[0, 0], delta=3.0)
            self.assertAlmostEqual(intr[1], k_true[1, 1], delta=3.0)
            self.assertAlmostEqual(intr[2], k_true[0, 2], delta=3.0)
            self.assertAlmostEqual(intr[3], k_true[1, 2], delta=3.0)
            self.assertAlmostEqual(dist_est[0], d_true[0], delta=0.01)
            self.assertAlmostEqual(dist_est[1], d_true[1], delta=0.02)
            self.assertTrue((out_dir / "detections").is_dir())

    def test_end_to_end_aprilgrid_pinhole_rot180_partial_views_and_wrong_rot_rejection(self):
        """端到端合成测试：Kalibr AprilGrid (rot=180) + 部分出画边缘视角 + 错配旋转拦截。"""
        image_size = (960, 720)
        k_true = np.array([[760.0, 0.0, 475.0], [0.0, 763.0, 365.0], [0.0, 0.0, 1.0]])
        d_true = np.array([-0.15, 0.04, -0.001, 0.0015])
        tag_cols, tag_rows, tag_size, tag_spacing = 5, 5, 0.055, 0.35
        tex, meta = _render_board_texture_aprilgrid(
            tag_cols, tag_rows, tag_size, tag_spacing, tag_rotation_deg=180, ppm=2400.0
        )
        pitch = tag_size * (1.0 + tag_spacing)
        center = np.array([0.5 * ((tag_cols - 1) * pitch + tag_size),
                           0.5 * ((tag_rows - 1) * pitch + tag_size), 0.0])

        # 刻意包含推到四角部分出画的视角（只看到部分 tag）和完整视角
        poses = [
            ((0, 0, 0), (0.0, 0.0), 0.72),
            ((22, -18, 10), (-0.28, -0.20), 0.65),   # 左上角部分出画
            ((-22, 18, -10), (0.28, -0.20), 0.65),   # 右上角部分出画
            ((20, 22, 14), (-0.28, 0.20), 0.65),     # 左下角部分出画
            ((-20, -22, -14), (0.28, 0.20), 0.65),   # 右下角部分出画
            ((28, 0, 5), (0.0, -0.18), 0.68),
            ((-28, 0, -5), (0.0, 0.18), 0.68),
            ((0, 28, 8), (-0.24, 0.0), 0.68),
            ((0, -28, -8), (0.24, 0.0), 0.68),
            ((18, 24, 30), (-0.14, -0.10), 0.76),
            ((-18, -24, -30), (0.14, 0.10), 0.76),
            ((24, -20, 40), (0.14, -0.10), 0.74),
            ((-24, 20, -40), (-0.14, 0.10), 0.74),
            ((15, -25, 180), (0.0, 0.0), 0.72),      # 整板倒置 180° 拍摄
            ((-15, 25, 90), (0.05, -0.05), 0.72),    # 整板横转 90° 拍摄
            ((26, 12, -25), (-0.10, 0.12), 0.70),
            ((-26, -12, 25), (0.10, -0.12), 0.70),
            ((12, 16, -60), (0.0, 0.0), 0.66),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "cam0"
            out_dir = root / "intrinsics_ok"
            out_bad = root / "intrinsics_bad_rot"
            img_dir.mkdir()
            target_yaml = root / "aprilgrid.yaml"
            target_yaml.write_text(yaml.safe_dump({
                "target_type": "aprilgrid",
                "tagCols": tag_cols,
                "tagRows": tag_rows,
                "tagSize": tag_size,
                "tagSpacing": tag_spacing,
                "first_id": 0,
                "tag_rotation_deg": 180,
            }), encoding="utf-8")

            for idx, (euler, shift, dist_z) in enumerate(poses):
                rvec, tvec = _aprilgrid_pose(center, euler, shift, dist_z)
                img = _render_view(tex, meta, image_size, k_true, d_true, rvec, tvec, model="pinhole")
                cv2.imwrite(str(img_dir / f"frame_{idx:03d}.png"), img)

            # 1) 正确配置应高精度收敛，并包含部分出画视图
            report = calib.main([
                "--images", str(img_dir),
                "--target", str(target_yaml),
                "--min-tags", "4",
                "--min-views", "12",
                "--strict-coverage",
                "--output", str(out_dir),
            ])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["target_type"], "aprilgrid")
            self.assertLess(report["reprojection_rms_px"], 0.30)
            self.assertLess(report["aprilgrid_stats"]["tags_per_view_min"], tag_cols * tag_rows)
            self.assertEqual(report["aprilgrid_rotation_check"]["best_rotation_deg"], 180)
            self.assertGreater(report["aprilgrid_rotation_check"]["configured_alignment_cos"], 0.95)
            self.assertEqual(report["coverage"]["covered_cells"], 9)

            camchain = yaml.safe_load((out_dir / "camchain_intrinsics.yaml").read_text(encoding="utf-8"))
            intr = camchain["cam0"]["intrinsics"]
            dist_est = camchain["cam0"]["distortion_coeffs"]
            self.assertAlmostEqual(intr[0], k_true[0, 0], delta=3.5)
            self.assertAlmostEqual(intr[1], k_true[1, 1], delta=3.5)
            self.assertAlmostEqual(intr[2], k_true[0, 2], delta=3.5)
            self.assertAlmostEqual(intr[3], k_true[1, 2], delta=3.5)
            self.assertAlmostEqual(dist_est[0], d_true[0], delta=0.015)
            self.assertAlmostEqual(dist_est[1], d_true[1], delta=0.025)

            # 2) 用错误的 --tag-rotation-deg 0 覆盖时，必须在优化前明确拦截且不输出 YAML
            with self.assertRaises(ValueError) as ctx:
                calib.main([
                    "--images", str(img_dir),
                    "--target", str(target_yaml),
                    "--tag-rotation-deg", "0",
                    "--min-tags", "4",
                    "--min-views", "12",
                    "--no-save-detections",
                    "--output", str(out_bad),
                ])
            self.assertIn("180°", str(ctx.exception))
            self.assertFalse((out_bad / "camchain_intrinsics.yaml").exists())
            bad_report = json.loads((out_bad / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(bad_report["status"], "failed")

    def test_end_to_end_aprilgrid_fisheye_rot0_nonzero_first_id(self):
        """端到端合成测试：AprilGrid (rot=0, first_id=12) + fisheye 模型 + 重复 ID 脏帧跳过。"""
        image_size = (800, 600)
        k_true = np.array([[430.0, 0.0, 405.0], [0.0, 432.0, 298.0], [0.0, 0.0, 1.0]])
        d_true = np.array([-0.045, 0.012, -0.003, 0.0008])
        tag_cols, tag_rows, tag_size, tag_spacing, first_id = 4, 4, 0.06, 0.35, 12
        tex, meta = _render_board_texture_aprilgrid(
            tag_cols, tag_rows, tag_size, tag_spacing,
            first_id=first_id, tag_rotation_deg=0, ppm=2200.0,
        )
        pitch = tag_size * (1.0 + tag_spacing)
        center = np.array([0.5 * ((tag_cols - 1) * pitch + tag_size),
                           0.5 * ((tag_rows - 1) * pitch + tag_size), 0.0])

        poses = [
            ((0, 0, 0), (0.0, 0.0), 0.45),
            ((24, -18, 12), (-0.20, -0.14), 0.42),
            ((-24, 18, -12), (0.20, -0.14), 0.42),
            ((22, 22, 16), (-0.20, 0.14), 0.42),
            ((-22, -22, -16), (0.20, 0.14), 0.42),
            ((30, 0, 8), (0.0, -0.15), 0.44),
            ((-30, 0, -8), (0.0, 0.15), 0.44),
            ((0, 30, 6), (-0.20, 0.0), 0.44),
            ((0, -30, -6), (0.20, 0.0), 0.44),
            ((20, 25, 35), (-0.10, -0.08), 0.48),
            ((-20, -25, -35), (0.10, 0.08), 0.48),
            ((25, -20, 45), (0.10, -0.08), 0.46),
            ((-25, 20, -45), (-0.10, 0.08), 0.46),
            ((16, -24, 130), (0.04, 0.04), 0.46),
            ((-16, 24, -130), (-0.04, -0.04), 0.46),
            ((28, 14, -20), (-0.08, 0.08), 0.45),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "cam0"
            out_dir = root / "intrinsics_fisheye"
            img_dir.mkdir()
            for idx, (euler, shift, dist_z) in enumerate(poses):
                rvec, tvec = _aprilgrid_pose(center, euler, shift, dist_z)
                img = _render_view(tex, meta, image_size, k_true, d_true, rvec, tvec, model="fisheye")
                cv2.imwrite(str(img_dir / f"frame_{idx:03d}.png"), img)

            # 混入一张含重复 tag ID 的脏帧，确认会被安全跳过而不污染标定
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            dup_img = np.full((image_size[1], image_size[0]), 235, dtype=np.uint8)
            m_dup = _draw_aruco_marker(dictionary, first_id, 100)
            dup_img[80:180, 80:180] = m_dup
            dup_img[80:180, 260:360] = m_dup
            cv2.imwrite(str(img_dir / "frame_999_dup.png"), dup_img)

            report = calib.main([
                "--images", str(img_dir),
                "--target-type", "aprilgrid",
                "--tag-cols", str(tag_cols),
                "--tag-rows", str(tag_rows),
                "--tag-size", str(tag_size),
                "--tag-spacing", str(tag_spacing),
                "--first-id", str(first_id),
                "--tag-rotation-deg", "0",
                "--min-tags", "4",
                "--model", "fisheye",
                "--min-views", "12",
                "--no-save-detections",
                "--output", str(out_dir),
            ])
            self.assertIn(report["status"], ("passed", "passed_with_warnings"))
            self.assertEqual(report["target_type"], "aprilgrid")
            self.assertLess(report["reprojection_rms_px"], 0.35)
            self.assertTrue(any("重复" in item["reason"] for item in report["views_skipped"]))

            camchain = yaml.safe_load((out_dir / "camchain_intrinsics.yaml").read_text(encoding="utf-8"))
            self.assertEqual(camchain["cam0"]["distortion_model"], "equidistant")
            intr = camchain["cam0"]["intrinsics"]
            self.assertAlmostEqual(intr[0], k_true[0, 0], delta=4.0)
            self.assertAlmostEqual(intr[1], k_true[1, 1], delta=4.0)
            self.assertAlmostEqual(intr[2], k_true[0, 2], delta=4.0)
            self.assertAlmostEqual(intr[3], k_true[1, 2], delta=4.0)

    def test_end_to_end_checkerboard_fisheye(self):
        """端到端合成测试：棋盘格 + fisheye 模型。"""
        image_size = (800, 600)
        k_true = np.array([[440.0, 0.0, 398.0], [0.0, 442.0, 302.0], [0.0, 0.0, 1.0]])
        d_true = np.array([-0.04, 0.01, -0.002, 0.0005])
        cols, rows, sq = 7, 5, 0.04
        tex, meta = _render_board_texture_checkerboard(cols, rows, sq, ppm=2400.0)
        center = np.array([(cols - 1) * sq * 0.5, (rows - 1) * sq * 0.5, 0.0])

        poses = [
            ((0, 0, 0), (0.0, 0.0), 0.36),
            ((24, -18, 8), (-0.14, -0.10), 0.40),
            ((-22, 20, -10), (0.14, -0.10), 0.40),
            ((22, 18, 12), (-0.14, 0.10), 0.40),
            ((-20, -22, -12), (0.14, 0.10), 0.40),
            ((30, 0, 5), (0.0, -0.11), 0.38),
            ((-30, 0, -5), (0.0, 0.11), 0.38),
            ((0, 30, 6), (-0.15, 0.0), 0.38),
            ((0, -30, -6), (0.15, 0.0), 0.38),
            ((18, 25, 25), (-0.09, -0.07), 0.44),
            ((-18, -25, -25), (0.09, 0.07), 0.44),
            ((25, -20, 35), (0.08, -0.07), 0.42),
            ((-25, 20, -35), (-0.08, 0.07), 0.42),
            ((15, -25, 18), (0.04, 0.04), 0.35),
            ((-15, 25, -18), (-0.04, -0.04), 0.35),
            ((26, 12, -22), (-0.07, 0.07), 0.38),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "cam0"
            out_dir = root / "intrinsics_cb_fisheye"
            img_dir.mkdir()
            for idx, (euler, shift, dist_z) in enumerate(poses):
                rvec, tvec = _pose_looking_at(center, euler, shift, dist_z)
                img = _render_view(tex, meta, image_size, k_true, d_true, rvec, tvec, model="fisheye")
                cv2.imwrite(str(img_dir / f"frame_{idx:03d}.png"), img)

            report = calib.main([
                "--images", str(img_dir),
                "--target-type", "checkerboard",
                "--cols", str(cols),
                "--rows", str(rows),
                "--square-size", str(sq),
                "--model", "fisheye",
                "--min-views", "12",
                "--no-save-detections",
                "--output", str(out_dir),
            ])
            self.assertIn(report["status"], ("passed", "passed_with_warnings"))
            self.assertLess(report["reprojection_rms_px"], 0.20)
            camchain = yaml.safe_load((out_dir / "camchain_intrinsics.yaml").read_text(encoding="utf-8"))
            self.assertEqual(camchain["cam0"]["distortion_model"], "equidistant")
            intr = camchain["cam0"]["intrinsics"]
            self.assertAlmostEqual(intr[0], k_true[0, 0], delta=3.0)
            self.assertAlmostEqual(intr[1], k_true[1, 1], delta=3.0)
            self.assertAlmostEqual(intr[2], k_true[0, 2], delta=3.0)
            self.assertAlmostEqual(intr[3], k_true[1, 2], delta=3.0)

    def test_aprilgrid_outlier_view_filtered(self):
        """验证 AprilGrid 下迭代异常视角剔除能自动剔除一块几何变形的坏帧。"""
        image_size = (800, 600)
        k_true = np.array([[720.0, 0.0, 400.0], [0.0, 720.0, 300.0], [0.0, 0.0, 1.0]])
        d_true = np.array([-0.12, 0.03, 0.0, 0.0])
        tag_cols, tag_rows, tag_size, tag_spacing = 4, 4, 0.055, 0.35
        tex, meta = _render_board_texture_aprilgrid(
            tag_cols, tag_rows, tag_size, tag_spacing, tag_rotation_deg=180, ppm=2400.0
        )
        # 用不同 tagSpacing 生成一张几何不一致的坏帧模拟弯曲/变形板面
        tex_bad, meta_bad = _render_board_texture_aprilgrid(
            tag_cols, tag_rows, tag_size, tag_spacing + 0.12, tag_rotation_deg=180, ppm=2400.0
        )
        pitch = tag_size * (1.0 + tag_spacing)
        center = np.array([0.5 * ((tag_cols - 1) * pitch + tag_size)] * 2 + [0.0])

        poses = [
            ((0, 0, 0), (0.0, 0.0), 0.62),
            ((22, -16, 10), (-0.18, -0.12), 0.60),
            ((-22, 16, -10), (0.18, -0.12), 0.60),
            ((20, 20, 14), (-0.18, 0.12), 0.60),
            ((-20, -20, -14), (0.18, 0.12), 0.60),
            ((28, 0, 6), (0.0, -0.14), 0.62),
            ((-28, 0, -6), (0.0, 0.14), 0.62),
            ((0, 28, 6), (-0.16, 0.0), 0.62),
            ((0, -28, -6), (0.16, 0.0), 0.62),
            ((18, 22, 30), (-0.10, -0.08), 0.66),
            ((-18, -22, -30), (0.10, 0.08), 0.66),
            ((24, -18, 40), (0.10, -0.08), 0.64),
            ((-24, 18, -40), (-0.10, 0.08), 0.64),
            ((14, -24, -20), (0.0, 0.0), 0.58),
            ((-14, 24, 20), (0.0, 0.0), 0.58),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "cam0"
            out_dir = root / "intrinsics_outlier"
            img_dir.mkdir()
            for idx, (euler, shift, dist_z) in enumerate(poses):
                rvec, tvec = _aprilgrid_pose(center, euler, shift, dist_z)
                img = _render_view(tex, meta, image_size, k_true, d_true, rvec, tvec, model="pinhole")
                cv2.imwrite(str(img_dir / f"frame_{idx:03d}.png"), img)
            # 注入坏帧
            rvec_b, tvec_b = _aprilgrid_pose(center, (10, -10, 5), (0.0, 0.0), 0.65)
            bad_img = _render_view(tex_bad, meta_bad, image_size, k_true, d_true, rvec_b, tvec_b, model="pinhole")
            cv2.imwrite(str(img_dir / "frame_999_warped.png"), bad_img)

            report = calib.main([
                "--images", str(img_dir),
                "--target-type", "aprilgrid",
                "--tag-cols", str(tag_cols),
                "--tag-rows", str(tag_rows),
                "--tag-size", str(tag_size),
                "--tag-spacing", str(tag_spacing),
                "--tag-rotation-deg", "180",
                "--min-tags", "4",
                "--min-views", "12",
                "--no-save-detections",
                "--output", str(out_dir),
            ])
            self.assertIn(report["status"], ("passed", "passed_with_warnings"))
            removed_names = [Path(item["file"]).name for item in report["removed_views"]]
            self.assertIn("frame_999_warped.png", removed_names)
            self.assertLess(report["reprojection_rms_px"], 0.40)
            self.assertLess(report["reprojection_rms_px"], report["initial_rms_px"] * 0.5)

    def test_config_defaults_switch_between_boards(self):
        """验证 PyCharm 直接改 calibration_config.py 顶部的 TARGET_TYPE / TARGET_YAML 生效。"""
        import calibration_config as cfg

        orig_type, orig_yaml = cfg.TARGET_TYPE, cfg.TARGET_YAML
        try:
            cfg.TARGET_YAML = None
            cfg.TARGET_TYPE = "aprilgrid"
            args_cal = calib.parse_args([])
            self.assertEqual(args_cal.target_spec.kind, "aprilgrid")
            self.assertEqual((args_cal.target_spec.cols, args_cal.target_spec.rows), (cfg.TAG_COLS, cfg.TAG_ROWS))

            args_cap = cap_mod.arguments(["--no-preview", "--duration", "5"])
            self.assertEqual(args_cap.target_spec.kind, "aprilgrid")

            # CLI 显式覆写 --target-type checkerboard 可覆盖配置文件中的 TARGET_YAML
            cfg.TARGET_YAML = REPO_ROOT / "extrinsic" / "configs" / "aprilgrid.yaml"
            args_override = calib.parse_args(["--target-type", "checkerboard", "--cols", "7", "--rows", "9"])
            self.assertEqual(args_override.target_spec.kind, "checkerboard")
            self.assertEqual((args_override.target_spec.cols, args_override.target_spec.rows), (7, 9))
        finally:
            cfg.TARGET_TYPE = orig_type
            cfg.TARGET_YAML = orig_yaml


if __name__ == "__main__":
    unittest.main()
