"""标定板几何定义、检测与可视化（支持棋盘格与 Kalibr tag36h11 AprilGrid）。

被 capture_calibration.py（实时预览/采图）与 calibrate_intrinsics.py（离线内参标定）
共同使用，确保采集端预览所见与离线标定使用完全一致的检测策略与坐标约定。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml


CHECKERBOARD_CLI_FIELDS = ("cols", "rows", "square_size")
APRILGRID_CLI_FIELDS = (
    "tag_cols",
    "tag_rows",
    "tag_size",
    "tag_spacing",
    "first_id",
    "tag_rotation_deg",
    "min_tags",
    "april_corner_window_px",
)


class DuplicateTagError(ValueError):
    """单帧中检测到重复的 AprilGrid tag ID（例如画面混入第二块板子或反光重影）。"""


@dataclass
class Detection:
    """单帧标定板检测结果。"""

    object_points: np.ndarray          # shape (N, 1, 3), dtype float64, 单位米
    image_points: np.ndarray           # shape (N, 1, 2), dtype float64, 像素坐标
    tag_ids: np.ndarray | None = None  # AprilGrid: shape (M,), dtype int32；棋盘格为 None
    tag_px: float | None = None        # AprilGrid: 单个 tag 边长中位数 (px)；棋盘格为 None

    @property
    def num_points(self) -> int:
        return int(len(self.image_points))

    @property
    def num_tags(self) -> int | None:
        return None if self.tag_ids is None else int(len(self.tag_ids))


def _check_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，当前为 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须是至少为 {minimum} 的整数，当前为 {value}")
    return int(value)


def _check_float(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} 必须是有限数值，当前为 {value!r}")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} 必须是有限数值，当前为 {value!r}") from exc
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        cond = "非负" if allow_zero else "正的"
        raise ValueError(f"{name} 必须是{cond}有限数，当前为 {value!r}")
    return result


def normalize_target_type(value) -> str:
    if not isinstance(value, str):
        raise ValueError(f"标定板类型必须是字符串，当前为 {value!r}")
    norm = value.strip().lower()
    if norm in ("checkerboard", "chessboard"):
        return "checkerboard"
    if norm == "aprilgrid":
        return "aprilgrid"
    raise ValueError(f'不支持的标定板类型 {value!r}；仅支持 "checkerboard" 或 "aprilgrid"')


def read_target_yaml(path: Path) -> dict:
    """读取标定板 YAML（兼容 Kalibr 与 extrinsic/configs/*.yaml，自动跳过 %YAML 指令）。"""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ValueError(f"标定板 YAML 文件不存在: {resolved}")
    try:
        text = resolved.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"无法读取标定板 YAML ({resolved}): {exc}") from exc
    cleaned = "\n".join(line for line in text.splitlines() if not line.startswith("%YAML:"))
    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        raise ValueError(f"标定板 YAML 解析失败 ({resolved}): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"标定板 YAML 顶层必须是键值映射: {resolved}")
    return data


def _require_mapping_key(mapping: dict, keys: tuple[str, ...], label: str):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    raise ValueError(f"标定板 YAML 缺少必填字段 {label}（可接受键名: {', '.join(keys)}）")


class CalibrationTarget:
    """棋盘格与 Kalibr tag36h11 AprilGrid 统一检测器与 3D 几何模型。"""

    def __init__(
        self,
        *,
        kind: str,
        cols: int,
        rows: int,
        col_spacing_m: float = 0.0,
        row_spacing_m: float = 0.0,
        tag_size_m: float = 0.0,
        tag_spacing: float = 0.0,
        first_id: int = 0,
        tag_rotation_deg: int = 180,
        min_tags: int = 4,
        april_corner_window_px: int = 4,
        source_yaml: Path | None = None,
    ):
        self.kind = normalize_target_type(kind)
        self.source_yaml = None if source_yaml is None else Path(source_yaml).expanduser().resolve()

        if self.kind == "checkerboard":
            self.cols = _check_int(cols, "COLS", 2)
            self.rows = _check_int(rows, "ROWS", 2)
            self.col_spacing_m = _check_float(col_spacing_m, "SQUARE_SIZE_M")
            self.row_spacing_m = _check_float(row_spacing_m, "SQUARE_SIZE_M")
            obj = np.zeros((self.rows * self.cols, 1, 3), dtype=np.float64)
            grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
            obj[:, 0, 0] = grid[:, 0] * self.col_spacing_m
            obj[:, 0, 1] = grid[:, 1] * self.row_spacing_m
            self.object_template = obj
            self.tag_size_m = None
            self.tag_spacing = None
            self.first_id = None
            self.tag_rotation_deg = None
            self.roll = 0
            self.min_tags = None
            self.april_corner_window_px = None
            self.dictionary = None
            self.detector_params = None
            self.detector = None
            self.fallback_detector_params = None
            self.fallback_detector = None
        else:
            self.cols = _check_int(cols, "TAG_COLS", 1)
            self.rows = _check_int(rows, "TAG_ROWS", 1)
            self.tag_size_m = _check_float(tag_size_m, "TAG_SIZE_M")
            self.tag_spacing = _check_float(tag_spacing, "TAG_SPACING", allow_zero=True)
            self.first_id = _check_int(first_id, "FIRST_ID", 0)
            if isinstance(tag_rotation_deg, bool) or not isinstance(tag_rotation_deg, int) or tag_rotation_deg not in (0, 90, 180, 270):
                raise ValueError(f"TAG_ROTATION_DEG 只能是 0 / 90 / 180 / 270，当前为 {tag_rotation_deg!r}")
            self.tag_rotation_deg = int(tag_rotation_deg)
            self.roll = (self.tag_rotation_deg // 90) % 4
            self.min_tags = _check_int(min_tags, "APRILGRID_MIN_TAGS", 1)
            total_tags = self.cols * self.rows
            if self.min_tags > total_tags:
                raise ValueError(f"APRILGRID_MIN_TAGS ({self.min_tags}) 不能大于总 tag 数 ({total_tags})")
            self.april_corner_window_px = _check_int(april_corner_window_px, "APRIL_CORNER_WINDOW_PX", 1)
            if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
                raise ValueError("AprilGrid 需要 OpenCV aruco 模块（请安装 opencv-contrib-python>=4.7）")
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            if self.first_id + total_tags > len(dictionary.bytesList):
                raise ValueError(
                    f"AprilGrid tag ID 范围 [{self.first_id}, {self.first_id + total_tags - 1}] "
                    f"超出 tag36h11 字典上限 ({len(dictionary.bytesList) - 1})"
                )
            self.dictionary = dictionary

            def _make_params():
                return (
                    cv2.aruco.DetectorParameters()
                    if hasattr(cv2.aruco, "DetectorParameters")
                    else cv2.aruco.DetectorParameters_create()
                )

            # 回退检测器（SUBPIX）：低对比度打印板在默认 (min=3, step=10) 下易漏检，
            # 改为 min=5, step=4 (win 5,9,13,17,21) 后可稳定检出。
            subpix_params = _make_params()
            subpix_params.adaptiveThreshWinSizeMin = 5
            subpix_params.adaptiveThreshWinSizeStep = 4
            subpix_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            subpix_params.cornerRefinementWinSize = int(self.april_corner_window_px)
            subpix_params.cornerRefinementMaxIterations = 40
            subpix_params.cornerRefinementMinAccuracy = 0.01

            # 主检测器优先使用 CORNER_REFINE_APRILTAG（AprilTag2 四边直线拟合求交）：
            # 相比通用 cornerSubPix，在非鞍点的 L 型黑色 tag 角点上无内缩/半像素系统偏差，
            # 实测重投影误差可从 ~0.5 px 降至 ~0.10-0.17 px，更适合内参 0.5 px 门槛。
            # aprilTagQuadSigma=0.8 仅在四边分割阶段抑制噪声碎片，兼顾精度与速度。
            if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
                april_params = _make_params()
                april_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
                april_params.aprilTagQuadDecimate = 0.0
                april_params.aprilTagQuadSigma = 0.8
                self.detector_params = april_params
                self.fallback_detector_params = subpix_params
            else:
                self.detector_params = subpix_params
                self.fallback_detector_params = None

            self.detector = (
                cv2.aruco.ArucoDetector(dictionary, self.detector_params)
                if hasattr(cv2.aruco, "ArucoDetector")
                else None
            )
            self.fallback_detector = (
                cv2.aruco.ArucoDetector(dictionary, self.fallback_detector_params)
                if (self.fallback_detector_params is not None and hasattr(cv2.aruco, "ArucoDetector"))
                else None
            )
            self.col_spacing_m = None
            self.row_spacing_m = None
            self.object_template = None

    @classmethod
    def from_checkerboard(
        cls,
        cols: int,
        rows: int,
        square_size_m: float | None = None,
        *,
        col_spacing_m: float | None = None,
        row_spacing_m: float | None = None,
        source_yaml: Path | None = None,
    ) -> "CalibrationTarget":
        sx = square_size_m if col_spacing_m is None else col_spacing_m
        sy = square_size_m if row_spacing_m is None else row_spacing_m
        return cls(
            kind="checkerboard",
            cols=cols,
            rows=rows,
            col_spacing_m=sx,
            row_spacing_m=sy,
            source_yaml=source_yaml,
        )

    @classmethod
    def from_aprilgrid(
        cls,
        tag_cols: int,
        tag_rows: int,
        tag_size_m: float,
        tag_spacing: float,
        *,
        first_id: int = 0,
        tag_rotation_deg: int = 180,
        min_tags: int = 4,
        april_corner_window_px: int = 4,
        source_yaml: Path | None = None,
    ) -> "CalibrationTarget":
        return cls(
            kind="aprilgrid",
            cols=tag_cols,
            rows=tag_rows,
            tag_size_m=tag_size_m,
            tag_spacing=tag_spacing,
            first_id=first_id,
            tag_rotation_deg=tag_rotation_deg,
            min_tags=min_tags,
            april_corner_window_px=april_corner_window_px,
            source_yaml=source_yaml,
        )

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        default_min_tags: int = 4,
        default_corner_window_px: int = 4,
        overrides: dict | None = None,
    ) -> "CalibrationTarget":
        entry = read_target_yaml(path)
        kind = normalize_target_type(_require_mapping_key(entry, ("target_type",), "target_type"))
        ov = overrides or {}
        if kind == "checkerboard":
            cols = ov["cols"] if ov.get("cols") is not None else _require_mapping_key(entry, ("targetCols", "cols"), "cols/targetCols")
            rows = ov["rows"] if ov.get("rows") is not None else _require_mapping_key(entry, ("targetRows", "rows"), "rows/targetRows")
            if ov.get("square_size") is not None:
                sx = sy = ov["square_size"]
            else:
                sx = _require_mapping_key(entry, ("colSpacingMeters", "square_size"), "square_size/colSpacingMeters")
                sy = _require_mapping_key(entry, ("rowSpacingMeters", "square_size"), "square_size/rowSpacingMeters")
            return cls.from_checkerboard(
                cols=cols,
                rows=rows,
                col_spacing_m=sx,
                row_spacing_m=sy,
                source_yaml=path,
            )
        tag_cols = ov["tag_cols"] if ov.get("tag_cols") is not None else _require_mapping_key(entry, ("tagCols",), "tagCols")
        tag_rows = ov["tag_rows"] if ov.get("tag_rows") is not None else _require_mapping_key(entry, ("tagRows",), "tagRows")
        tag_size = ov["tag_size"] if ov.get("tag_size") is not None else _require_mapping_key(entry, ("tagSize",), "tagSize")
        tag_spacing = ov["tag_spacing"] if ov.get("tag_spacing") is not None else _require_mapping_key(entry, ("tagSpacing",), "tagSpacing")
        first_id = ov["first_id"] if ov.get("first_id") is not None else entry.get("first_id", 0)
        tag_rotation_deg = ov["tag_rotation_deg"] if ov.get("tag_rotation_deg") is not None else entry.get("tag_rotation_deg", 180)
        min_tags = ov["min_tags"] if ov.get("min_tags") is not None else entry.get("min_tags", default_min_tags)
        corner_win = (
            ov["april_corner_window_px"]
            if ov.get("april_corner_window_px") is not None
            else entry.get("april_corner_window_px", default_corner_window_px)
        )
        return cls.from_aprilgrid(
            tag_cols=tag_cols,
            tag_rows=tag_rows,
            tag_size_m=tag_size,
            tag_spacing=tag_spacing,
            first_id=first_id,
            tag_rotation_deg=tag_rotation_deg,
            min_tags=min_tags,
            april_corner_window_px=corner_win,
            source_yaml=path,
        )

    def april_points(self, tag_id: int, *, rotation_deg: int | None = None) -> np.ndarray:
        """返回单个 AprilTag 4 个角点在板面坐标系中的 3D 坐标 (4, 3)。

        Kalibr 约定：ID 从左到右、从下到上递增；板面 +X 向右，+Y 向上，Z=0。
        OpenCV 检出的 4 个角点按 marker 自身 [TL, TR, BR, BL] 顺时针排列。
        """
        if self.kind != "aprilgrid":
            raise ValueError("仅 AprilGrid 标定板支持 april_points")
        idx = int(tag_id) - self.first_id
        x = (idx % self.cols) * self.tag_size_m * (1.0 + self.tag_spacing)
        y = (idx // self.cols) * self.tag_size_m * (1.0 + self.tag_spacing)
        s = self.tag_size_m
        quad = np.array(
            [[x, y + s, 0.0], [x + s, y + s, 0.0], [x + s, y, 0.0], [x, y, 0.0]],
            dtype=np.float64,
        )
        roll = self.roll if rotation_deg is None else (int(rotation_deg) // 90) % 4
        return np.roll(quad, roll, axis=0) if roll else quad

    def _run_aruco_detector(self, gray: np.ndarray, detector, params) -> Detection | None:
        if detector is None:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=params
            )
        else:
            corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return None
        # OpenCV 的 CORNER_REFINE_APRILTAG 沿用 AprilTag2 像素角点原点（像素 (i,j) 中心在 (i+0.5, j+0.5)），
        # 而 OpenCV calibrateCamera / projectPoints / 棋盘格检测器 / SUBPIX / Kalibr / OpenVINS
        # 均采用像素中心原点（像素 (i,j) 中心在 (i, j)）。此处减 0.5 对齐坐标原点，消除主点 (cx, cy) +0.5 px 系统偏差。
        april_origin_shift = (
            0.5
            if (
                hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG")
                and params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG
            )
            else 0.0
        )
        upper = self.first_id + self.rows * self.cols
        valid = [
            (int(tag_id), c.reshape(4, 2).astype(np.float64) - april_origin_shift)
            for tag_id, c in zip(ids.ravel(), corners)
            if self.first_id <= int(tag_id) < upper
        ]
        seen_ids = set()
        duplicate_ids = []
        for tag_id, _ in valid:
            if tag_id in seen_ids:
                duplicate_ids.append(tag_id)
            seen_ids.add(tag_id)
        if duplicate_ids:
            dups = sorted(set(duplicate_ids))
            raise DuplicateTagError(f"检测到重复 AprilGrid tag ID {dups}；请移走画面中的额外标定板或反光")
        if len(valid) < self.min_tags:
            return None
        valid.sort(key=lambda item: item[0])
        obj = np.concatenate([self.april_points(i) for i, _ in valid], axis=0).reshape(-1, 1, 3)
        img = np.concatenate([c for _, c in valid], axis=0).reshape(-1, 1, 2).astype(np.float64)
        tag_ids = np.array([i for i, _ in valid], dtype=np.int32)
        quads = img.reshape(-1, 4, 2)
        tag_px = float(np.median(np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)))
        return Detection(object_points=obj, image_points=img, tag_ids=tag_ids, tag_px=tag_px)

    def detect(self, gray: np.ndarray) -> Detection | None:
        """在灰度图上检测标定板角点，返回 Detection 或 None。"""
        if self.kind == "checkerboard":
            pattern = (self.cols, self.rows)
            corners = None
            if hasattr(cv2, "findChessboardCornersSB"):
                # 一致的 180° 翻转会被单帧外参吸收，跳过 EXHAUSTIVE 以兼顾速度。
                found, sb_corners = cv2.findChessboardCornersSB(
                    gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if found:
                    corners = sb_corners.astype(np.float64)
            if corners is None:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found, classic = cv2.findChessboardCorners(gray, pattern, flags)
                if not found:
                    return None
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
                corners = cv2.cornerSubPix(gray, classic, (11, 11), (-1, -1), criteria).astype(np.float64)
            return Detection(
                object_points=self.object_template.copy(),
                image_points=corners.reshape(-1, 1, 2),
            )

        det = self._run_aruco_detector(gray, self.detector, self.detector_params)
        if det is not None or self.fallback_detector_params is None:
            return det
        return self._run_aruco_detector(gray, self.fallback_detector, self.fallback_detector_params)

    def check_aprilgrid_rotation(self, detections: list[Detection]) -> dict | None:
        """无需相机内参，校验 AprilGrid 单个 tag 局部坐标轴是否与整板网格方向一致。

        对每帧内每个 tag，用其 4 个角点求出局部 2x2 雅可比矩阵 J（将板面 (dX, dY)
        映射到像素位移），再与相邻 tag 中心之间的实际像素位移比对余弦相似度。
        即使在强鱼眼畸变或大倾角下，正确旋转的最近邻一致性仍接近 +1.0，
        而错 180° 时接近 -1.0、错 90°/270° 时接近 0.0。
        """
        if self.kind != "aprilgrid":
            return None
        s2 = self.tag_size_m * self.tag_size_m
        scores_by_rot: dict[int, list[float]] = {0: [], 90: [], 180: [], 270: []}

        for det in detections:
            if det.tag_ids is None or len(det.tag_ids) < 2:
                continue
            ids = det.tag_ids
            img_quads = det.image_points.reshape(-1, 4, 2)
            # 先算与旋转无关的板面中心和图像中心
            base_quads = np.stack([self.april_points(int(i), rotation_deg=0)[:, :2] for i in ids], axis=0)
            obj_centers = base_quads.mean(axis=1)   # (M, 2)
            img_centers = img_quads.mean(axis=1)    # (M, 2)
            # 每个 tag 找板面距离最近的邻居，最大化局部线性近似精度
            diff_mat = obj_centers[None, :, :] - obj_centers[:, None, :]  # (M, M, 2)
            dist2 = np.sum(diff_mat * diff_mat, axis=2)
            np.fill_diagonal(dist2, np.inf)
            nearest = np.argmin(dist2, axis=1)
            d_obj = obj_centers[nearest] - obj_centers  # (M, 2)
            d_img = img_centers[nearest] - img_centers  # (M, 2)
            d_img_norm = np.linalg.norm(d_img, axis=1)

            for rot in (0, 90, 180, 270):
                roll = (rot // 90) % 4
                obj_quads = np.roll(base_quads, roll, axis=1) if roll else base_quads
                obj_c = obj_quads - obj_centers[:, None, :]      # (M, 4, 2)
                img_c = img_quads - img_centers[:, None, :]      # (M, 4, 2)
                # 正方形角点满足 obj_c^T @ obj_c = s^2 * I_2，闭式最小二乘雅可比
                jac = np.einsum("mki,mkj->mij", obj_c, img_c) / s2  # (M, 2, 2)
                d_pred = np.einsum("mi,mij->mj", d_obj, jac)        # (M, 2)
                d_pred_norm = np.linalg.norm(d_pred, axis=1)
                denom = d_img_norm * d_pred_norm
                valid_mask = denom > 1e-6
                if np.any(valid_mask):
                    cosines = np.sum(d_img[valid_mask] * d_pred[valid_mask], axis=1) / denom[valid_mask]
                    scores_by_rot[rot].extend(cosines.tolist())

        if not scores_by_rot[self.tag_rotation_deg]:
            return None
        medians = {rot: float(np.median(vals)) for rot, vals in scores_by_rot.items() if vals}
        best_rot = max(medians, key=medians.get)
        current_score = medians[self.tag_rotation_deg]
        best_score = medians[best_rot]
        result = {
            "configured_rotation_deg": self.tag_rotation_deg,
            "configured_alignment_cos": round(current_score, 4),
            "best_rotation_deg": best_rot,
            "best_alignment_cos": round(best_score, 4),
            "all_rotation_cos": {str(k): round(v, 4) for k, v in medians.items()},
        }
        if best_rot != self.tag_rotation_deg and current_score < 0.5 and best_score > 0.85:
            raise ValueError(
                f"AprilGrid tag 方向配置不匹配：当前 tag_rotation_deg={self.tag_rotation_deg}"
                f"（局部与整板方向一致性 {current_score:.2f}），"
                f"图像几何指向 {best_rot}°（一致性 {best_score:.2f}）。"
                "Kalibr PDF 生成的板子请设 TAG_ROTATION_DEG = 180，OpenCV 自拼板子请设 0。"
            )
        return result

    def draw_detection(self, canvas: np.ndarray, detection: Detection, *, scale: float = 1.0,
                       for_preview: bool = False) -> None:
        """在 BGR 图像上绘制检测结果（原地绘制）。"""
        if self.kind == "checkerboard":
            pts = detection.image_points.reshape(-1, 2) * scale
            if for_preview:
                for x, y in pts:
                    cv2.circle(canvas, (round(x), round(y)), 2, (0, 255, 0), -1)
            else:
                cv2.drawChessboardCorners(
                    canvas, (self.cols, self.rows), pts.reshape(-1, 1, 2).astype(np.float32), True
                )
            return

        quads = detection.image_points.reshape(-1, 4, 2) * scale
        ids = detection.tag_ids if detection.tag_ids is not None else range(len(quads))
        thickness = 2 if not for_preview else 1
        font_scale = 0.45 if not for_preview else 0.38
        for tag_id, quad in zip(ids, quads):
            pts = np.rint(quad).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (0, 230, 0), thickness, cv2.LINE_AA)
            # 红点标出每个 tag 的 0 号角点，便于肉眼核对 tag_rotation_deg 方向
            cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), 3, (0, 0, 255), -1)
            center = np.rint(quad.mean(axis=0)).astype(int)
            cv2.putText(
                canvas,
                str(int(tag_id)),
                (int(center[0]) - 8, int(center[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def short_label(self) -> str:
        if self.kind == "checkerboard":
            return f"chessboard {self.cols}x{self.rows}"
        return f"aprilgrid {self.cols}x{self.rows}"

    def describe(self) -> str:
        if self.kind == "checkerboard":
            if abs(self.col_spacing_m - self.row_spacing_m) < 1e-12:
                size_str = f"square={self.col_spacing_m:g}m"
            else:
                size_str = f"spacing={self.col_spacing_m:g}x{self.row_spacing_m:g}m"
            return f"棋盘格 {self.cols}x{self.rows} ({size_str})"
        return (
            f"AprilGrid {self.cols}x{self.rows} "
            f"(tagSize={self.tag_size_m:g}m, tagSpacing={self.tag_spacing:g}, "
            f"first_id={self.first_id}, rot={self.tag_rotation_deg}°, min_tags={self.min_tags})"
        )

    def summary_dict(self) -> dict:
        if self.kind == "checkerboard":
            out = {
                "target_type": "checkerboard",
                "cols": self.cols,
                "rows": self.rows,
                "square_size_m": self.col_spacing_m,
            }
            if abs(self.col_spacing_m - self.row_spacing_m) >= 1e-12:
                out["col_spacing_m"] = self.col_spacing_m
                out["row_spacing_m"] = self.row_spacing_m
        else:
            out = {
                "target_type": "aprilgrid",
                "tagCols": self.cols,
                "tagRows": self.rows,
                "tagSize": self.tag_size_m,
                "tagSpacing": self.tag_spacing,
                "first_id": self.first_id,
                "tag_rotation_deg": self.tag_rotation_deg,
                "min_tags": self.min_tags,
                "april_corner_window_px": self.april_corner_window_px,
            }
        if self.source_yaml is not None:
            out["source_yaml"] = str(self.source_yaml)
        return out


def _int_at_least(minimum: int):
    def _parse(value: str) -> int:
        result = int(value)
        if result < minimum:
            raise argparse.ArgumentTypeError(f"必须是至少为 {minimum} 的整数")
        return result
    return _parse


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("必须是正的有限数")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("必须是非负有限数")
    return result


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器添加棋盘格与 AprilGrid 共用的标定板参数。"""
    group = parser.add_argument_group("标定板参数（支持 checkerboard 与 aprilgrid）")
    group.add_argument(
        "--target",
        type=Path,
        default=None,
        help="标定板 YAML 路径（兼容 Kalibr 及 extrinsic/configs/{aprilgrid,checkerboard}.yaml）",
    )
    group.add_argument(
        "--target-type",
        choices=("checkerboard", "chessboard", "aprilgrid"),
        default=None,
        help="标定板类型：checkerboard（棋盘格）或 aprilgrid（Kalibr tag36h11）",
    )
    # 棋盘格参数（默认 None 以便区分命令行显式覆写与配置文件默认值）
    group.add_argument("--cols", type=_int_at_least(2), default=None, help="[棋盘格] 横向内角点数")
    group.add_argument("--rows", type=_int_at_least(2), default=None, help="[棋盘格] 纵向内角点数")
    group.add_argument("--square-size", type=_positive_float, default=None,
                       help="[棋盘格] 相邻角点距离（米）")
    # AprilGrid 参数
    group.add_argument("--tag-cols", type=_int_at_least(1), default=None, help="[AprilGrid] 横向 tag 数")
    group.add_argument("--tag-rows", type=_int_at_least(1), default=None, help="[AprilGrid] 纵向 tag 数")
    group.add_argument("--tag-size", type=_positive_float, default=None,
                       help="[AprilGrid] 单个 tag 黑色外框边长（米）")
    group.add_argument("--tag-spacing", type=_nonnegative_float, default=None,
                       help="[AprilGrid] tag 间白色间隙 / tagSize（无量纲）")
    group.add_argument("--first-id", type=_int_at_least(0), default=None,
                       help="[AprilGrid] 左下角起始 tag ID（默认 0）")
    group.add_argument("--tag-rotation-deg", type=int, choices=(0, 90, 180, 270), default=None,
                       help="[AprilGrid] 单 tag 旋转角：Kalibr PDF 板填 180，OpenCV 自拼板填 0")
    group.add_argument("--min-tags", type=_int_at_least(1), default=None,
                       help="[AprilGrid] 单帧最少检出 tag 数")
    group.add_argument("--april-corner-window-px", type=_int_at_least(1), default=None,
                       help="[AprilGrid] 亚像素角点细化半窗（像素）")


def build_target(
    args: argparse.Namespace,
    config,
    *,
    error_fn: Callable[[str], None] | None = None,
) -> CalibrationTarget:
    """按 CLI > TARGET_YAML > calibration_config.py 的优先级构造并校验标定板对象。"""

    def fail(msg: str):
        if error_fn is not None:
            error_fn(msg)
        raise ValueError(msg)

    try:
        cli_checker = [f for f in CHECKERBOARD_CLI_FIELDS if getattr(args, f, None) is not None]
        cli_april = [f for f in APRILGRID_CLI_FIELDS if getattr(args, f, None) is not None]
        if cli_checker and cli_april:
            fail(f"不能同时传入棋盘格参数 ({', '.join(cli_checker)}) 与 AprilGrid 参数 ({', '.join(cli_april)})")

        default_min_tags = getattr(config, "APRILGRID_MIN_TAGS", 4)
        default_corner_win = getattr(config, "APRIL_CORNER_WINDOW_PX", 4)
        config_yaml = getattr(config, "TARGET_YAML", None)
        explicit_yaml = getattr(args, "target", None)
        explicit_type = getattr(args, "target_type", None)

        overrides = {
            name: getattr(args, name, None)
            for name in CHECKERBOARD_CLI_FIELDS + APRILGRID_CLI_FIELDS
        }

        yaml_to_use = None
        if explicit_yaml is not None:
            yaml_to_use = Path(explicit_yaml)
            entry = read_target_yaml(yaml_to_use)
            yaml_kind = normalize_target_type(_require_mapping_key(entry, ("target_type",), "target_type"))
            if explicit_type is not None and normalize_target_type(explicit_type) != yaml_kind:
                fail(f"--target-type={explicit_type} 与 --target YAML ({yaml_kind}) 不一致")
            resolved_kind = yaml_kind
        elif explicit_type is not None:
            resolved_kind = normalize_target_type(explicit_type)
            if config_yaml is not None:
                entry = read_target_yaml(Path(config_yaml))
                yaml_kind = normalize_target_type(_require_mapping_key(entry, ("target_type",), "target_type"))
                if yaml_kind == resolved_kind:
                    yaml_to_use = Path(config_yaml)
        elif config_yaml is not None:
            yaml_to_use = Path(config_yaml)
            entry = read_target_yaml(yaml_to_use)
            resolved_kind = normalize_target_type(_require_mapping_key(entry, ("target_type",), "target_type"))
        else:
            resolved_kind = normalize_target_type(getattr(config, "TARGET_TYPE", "checkerboard"))

        if resolved_kind == "checkerboard" and cli_april:
            fail(
                f"当前标定板为 checkerboard，不能传入 AprilGrid 参数 ({', '.join(cli_april)})；"
                "如需标定 AprilGrid，请加 --target-type aprilgrid 或 --target <aprilgrid.yaml>"
            )
        if resolved_kind == "aprilgrid" and cli_checker:
            fail(
                f"当前标定板为 aprilgrid，不能传入棋盘格参数 ({', '.join(cli_checker)})；"
                "如需标定棋盘格，请加 --target-type checkerboard 或 --target <checkerboard.yaml>"
            )

        if yaml_to_use is not None:
            return CalibrationTarget.from_yaml(
                yaml_to_use,
                default_min_tags=default_min_tags,
                default_corner_window_px=default_corner_win,
                overrides=overrides,
            )

        if resolved_kind == "checkerboard":
            cols = overrides["cols"] if overrides["cols"] is not None else getattr(config, "COLS", 8)
            rows = overrides["rows"] if overrides["rows"] is not None else getattr(config, "ROWS", 11)
            square_size = (
                overrides["square_size"]
                if overrides["square_size"] is not None
                else getattr(config, "SQUARE_SIZE_M", 0.03)
            )
            return CalibrationTarget.from_checkerboard(cols, rows, square_size)

        return CalibrationTarget.from_aprilgrid(
            tag_cols=overrides["tag_cols"] if overrides["tag_cols"] is not None else getattr(config, "TAG_COLS", 6),
            tag_rows=overrides["tag_rows"] if overrides["tag_rows"] is not None else getattr(config, "TAG_ROWS", 6),
            tag_size_m=overrides["tag_size"] if overrides["tag_size"] is not None else getattr(config, "TAG_SIZE_M", 0.088),
            tag_spacing=overrides["tag_spacing"] if overrides["tag_spacing"] is not None else getattr(config, "TAG_SPACING", 0.56),
            first_id=overrides["first_id"] if overrides["first_id"] is not None else getattr(config, "FIRST_ID", 0),
            tag_rotation_deg=(
                overrides["tag_rotation_deg"]
                if overrides["tag_rotation_deg"] is not None
                else getattr(config, "TAG_ROTATION_DEG", 180)
            ),
            min_tags=overrides["min_tags"] if overrides["min_tags"] is not None else default_min_tags,
            april_corner_window_px=(
                overrides["april_corner_window_px"]
                if overrides["april_corner_window_px"] is not None
                else default_corner_win
            ),
        )
    except ValueError as exc:
        fail(str(exc))
        raise
