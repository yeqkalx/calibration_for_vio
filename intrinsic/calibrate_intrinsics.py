#!/usr/bin/env python3
"""相机离线内参标定（支持棋盘格与 Kalibr tag36h11 AprilGrid）。

默认参数集中在同目录 calibration_config.py，也支持通过 --target 复用外参/Kalibr 标定板 YAML。
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calibration_config as config
from target import (
    CalibrationTarget,
    DuplicateTagError,
    add_target_arguments,
    build_target,
)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def positive_int(value):
    result = int(value)
    if result < 2:
        raise argparse.ArgumentTypeError("必须是至少为 2 的整数")
    return result


def positive_float(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("必须是正的有限数")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=config.IMAGES_DIR, help="原始图片目录")
    add_target_arguments(parser)
    parser.add_argument("--model", choices=("pinhole", "fisheye"), default=config.MODEL)
    parser.add_argument("--output", type=Path, default=config.RESULT_DIR)
    parser.add_argument("--topic", default=config.TOPIC)
    parser.add_argument("--min-views", type=positive_int, default=config.MIN_VIEWS,
                        help="最少有效视角，默认 20；建议拍摄 25~40 张")
    parser.add_argument("--save-detections", action=argparse.BooleanOptionalAction,
                        default=config.SAVE_DETECTIONS, help="保存角点检查图")
    parser.add_argument("--outlier-factor", type=positive_float, default=config.OUTLIER_FACTOR)
    parser.add_argument("--outlier-floor-px", type=positive_float, default=config.OUTLIER_FLOOR_PX)
    parser.add_argument("--max-rms-px", type=positive_float, default=config.MAX_RMS_PX)
    parser.add_argument("--strict-coverage", action=argparse.BooleanOptionalAction,
                        default=config.STRICT_COVERAGE)
    args = parser.parse_args(argv)
    # argparse does not validate non-string defaults loaded from the config file.
    if isinstance(args.min_views, bool) or not isinstance(args.min_views, int) or args.min_views < 2:
        parser.error("最少视角数必须是至少为 2 的整数")
    for value in (args.outlier_factor, args.outlier_floor_px, args.max_rms_px):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            parser.error("误差阈值必须是正的有限数")
    if args.model not in ("pinhole", "fisheye"):
        parser.error("MODEL 只能是 pinhole 或 fisheye")
    args.target_spec = build_target(args, config, error_fn=parser.error)
    return args


def board_points(cols, rows, square_size):
    """兼容旧接口：返回棋盘格 (rows*cols, 1, 3) 3D 内角点模板。"""
    return CalibrationTarget.from_checkerboard(cols, rows, square_size).object_template.copy()


def read_grayscale(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error):
        return None


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"PNG 编码失败: {path}")
    encoded.tofile(str(path))


def find_corners(gray, pattern):
    """兼容旧接口：检测棋盘格角点并返回 (N, 1, 2) float64 或 None。"""
    det = CalibrationTarget.from_checkerboard(pattern[0], pattern[1], 1.0).detect(gray)
    return None if det is None else det.image_points


def calibrate(object_points, image_points, image_size, model):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                config.CALIBRATION_ITERATIONS, 1e-9)
    if model == "fisheye":
        focal = config.FISHEYE_FOCAL_RATIO * max(image_size)
        k_init = np.array([[focal, 0, image_size[0] / 2],
                           [0, focal, image_size[1] / 2], [0, 0, 1]], dtype=np.float64)
        flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                 | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            [p.reshape(-1, 1, 3).astype(np.float64) for p in object_points],
            [p.reshape(-1, 1, 2).astype(np.float64) for p in image_points],
            image_size, k_init,
            np.zeros((4, 1), dtype=np.float64), flags=flags, criteria=criteria)
    else:
        # Fit the four radtan parameters actually exported; do not drop fitted k3.
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
            [p.reshape(-1, 1, 3).astype(np.float32) for p in object_points],
            [p.reshape(-1, 1, 2).astype(np.float32) for p in image_points],
            image_size, None, None,
            flags=cv2.CALIB_FIX_K3, criteria=criteria)
    return rms, k, d.ravel()[:4].tolist(), rvecs, tvecs


def reprojection_errors(object_points, image_points, k, d, rvecs, tvecs, model):
    errors, all_squared, total_points = [], 0.0, 0
    dist = np.asarray(d, dtype=np.float64)
    for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
        if model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(
                obj.reshape(-1, 1, 3).astype(np.float64), rv, tv, k, dist)
        else:
            projected, _ = cv2.projectPoints(
                obj.reshape(-1, 1, 3).astype(np.float64), rv, tv, k, dist)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        squared = float(np.sum(delta * delta))
        errors.append(float(np.sqrt(squared / len(delta))))
        all_squared += squared
        total_points += len(delta)
    return errors, float(np.sqrt(all_squared / total_points))


def fit_with_filter(object_points, image_points, files, image_size, args, history):
    active = list(range(len(files)))
    while True:
        obj = [object_points[i] for i in active]
        img = [image_points[i] for i in active]
        rms, k, d, rvecs, tvecs = calibrate(obj, img, image_size, args.model)
        errors, pixel_rms = reprojection_errors(obj, img, k, d, rvecs, tvecs, args.model)
        if not np.isfinite(errors).all() or not np.isfinite(pixel_rms):
            raise ValueError("优化发散产生非有限误差；检查板型、视角和模型")
        threshold = max(args.outlier_factor * float(np.median(errors)), args.outlier_floor_px)
        worst = int(np.argmax(errors))
        record = {"round": len(history) + 1, "views_used": len(active),
                  "opencv_rms_px": float(rms), "reprojection_rms_px": pixel_rms,
                  "threshold_px": threshold, "removed": [],
                  "views": [{"file": str(files[i]), "reprojection_rms_px": err}
                            for i, err in zip(active, errors)]}
        history.append(record)
        if errors[worst] <= threshold:
            return active, rms, k, d, errors, pixel_rms
        if len(active) <= args.min_views:
            record["blocked_removal"] = str(files[active[worst]])
            raise ValueError(f"仍有异常视角，但再剔除会少于 {args.min_views} 张；请补拍，不强行输出内参")
        record["removed"] = [{"file": str(files[active[worst]]),
                              "reprojection_rms_px": errors[worst]}]
        print(f"第 {len(history)} 轮 RMS {pixel_rms:.4f} px；剔除 {files[active[worst]].name} "
              f"({errors[worst]:.3f} px > {threshold:.3f} px)", flush=True)
        del active[worst]


def validate_solution(k, d, rms, image_size, model, target_kind="checkerboard"):
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("标定发散：结果含非有限值")
    w, h = image_size
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    issues = []
    if not all(config.FOCAL_MIN_RATIO * max(w, h) < f < config.FOCAL_MAX_RATIO * max(w, h)
               for f in (fx, fy)):
        issues.append("焦距超出当前相机的经验范围")
    if fx <= 0 or fy <= 0 or abs(fx - fy) / max(fx, fy) >= config.MAX_FOCAL_ASPECT_ERROR:
        issues.append("fx/fy 比例异常")
    if not (0 < cx < w and 0 < cy < h):
        issues.append("主点不在图像内部")
    if model == "pinhole":
        for name, value, limit in zip(("k1", "k2", "p1", "p2"), d, config.RADTAN_LIMITS):
            if abs(value) >= limit:
                issues.append(f"{name}={value:.5g} 超出经验防发散护栏 {limit}")
    if issues:
        extra = (
            " 若使用 AprilGrid，请同时核对 TAG_SPACING、TAG_ROTATION_DEG（Kalibr PDF 为 180，自拼板为 0）与 TAG_COLS/TAG_ROWS。"
            if target_kind == "aprilgrid" else ""
        )
        raise ValueError("；".join(issues) + "。检查倾斜角、边缘覆盖、平整度和镜头模型；"
                         "这些阈值是当前相机的护栏，不是通用物理定律。" + extra)


def coverage_report(points, image_size):
    w, h = image_size
    counts = np.zeros((3, 3), dtype=int)
    for corners in points:
        cells = np.floor(corners.reshape(-1, 2) * [3 / w, 3 / h]).astype(int)
        for col, row in cells:
            if 0 <= col < 3 and 0 <= row < 3:
                counts[row, col] += 1
    missing = [name for row, col, name in ((0, 0, "左上"), (0, 2, "右上"),
                                          (2, 0, "左下"), (2, 2, "右下")) if counts[row, col] == 0]
    covered = int(np.count_nonzero(counts))
    warnings = []
    if covered < config.MIN_COVERED_CELLS or missing:
        warnings.append(f"九宫格仅覆盖 {covered}/9 格；未覆盖角格: {','.join(missing) or '无'}。"
                        "边缘/四角样本不足，畸变不可信，应补拍。")
    return {"corner_counts_3x3": counts.tolist(), "covered_cells": covered,
            "missing_corner_cells": missing, "warnings": warnings}


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main(argv=None):
    args = parse_args(argv)
    target = args.target_spec
    image_dir = args.images.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"图片目录不存在: {image_dir}；先采集，或修改 calibration_config.py")
    if image_dir == output or image_dir.is_relative_to(output):
        raise ValueError("结果目录不能等于或包含原始图片目录")
    if (output / "camchain_intrinsics.yaml").exists() or (output / "report.json").exists():
        raise ValueError(f"结果目录已有标定文件: {output}；请换新的 RESULT_DIR，保留历史结果")
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in EXTENSIONS
                   and p.is_file() and not p.resolve().is_relative_to(output))
    if not files:
        raise ValueError(f"未找到原始图片: {image_dir}")

    print(f"标定板: {target.describe()} | 模型: {args.model}", flush=True)
    object_points, image_points, view_detections, accepted, detections, skipped = [], [], [], [], [], []
    image_size = None
    report = {
        "status": "failed",
        "note": "Only intrinsics; no T_cam_imu or timeshift_cam_imu.",
        "model": args.model,
        "target_type": target.kind,
        "target": target.summary_dict(),
        "min_views": args.min_views,
        "views_skipped": skipped,
        "filter_rounds": [],
        "warnings": [],
    }
    if target.kind == "checkerboard":
        report["board_inner_corners"] = [target.cols, target.rows]
        report["square_size_m"] = target.col_spacing_m

    try:
        for index, path in enumerate(files):
            print(f"检测 {index + 1}/{len(files)}: {path.name}", flush=True)
            gray = read_grayscale(path)
            if gray is None:
                skipped.append({"file": str(path), "reason": "cannot read"})
                continue
            current_size = (gray.shape[1], gray.shape[0])
            if image_size is None:
                image_size = current_size
            if image_size != current_size:
                skipped.append({"file": str(path), "reason": f"size {current_size} != {image_size}"})
                continue
            try:
                detection = target.detect(gray)
            except DuplicateTagError as exc:
                print(f"警告: {path.name}: {exc}", file=sys.stderr)
                skipped.append({"file": str(path), "reason": str(exc)})
                continue
            if detection is None:
                reason = (
                    "chessboard not detected"
                    if target.kind == "checkerboard"
                    else f"aprilgrid not detected (< {target.min_tags} tags)"
                )
                skipped.append({"file": str(path), "reason": reason})
                continue
            object_points.append(detection.object_points)
            image_points.append(detection.image_points)
            view_detections.append(detection)
            accepted.append(path)
            dest = None
            if args.save_detections:
                drawing = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                target.draw_detection(drawing, detection)
                for col in (1, 2):
                    cv2.line(drawing, (gray.shape[1] * col // 3, 0),
                             (gray.shape[1] * col // 3, gray.shape[0] - 1), (0, 180, 255), 1)
                for row in (1, 2):
                    cv2.line(drawing, (0, gray.shape[0] * row // 3),
                             (gray.shape[1] - 1, gray.shape[0] * row // 3), (0, 180, 255), 1)
                relative = path.relative_to(image_dir)
                dest = output / "detections" / relative.parent / f"{relative.name}_detect.png"
                write_png(dest, drawing)
            detections.append(str(dest) if dest else None)

        report.update(image_size=list(image_size) if image_size else None,
                      views_detected=len(accepted), detected_files=[str(p) for p in accepted])
        if image_points:
            report["coverage_before_filter"] = coverage_report(image_points, image_size)
        if target.kind == "aprilgrid" and view_detections:
            tag_counts = [d.num_tags for d in view_detections if d.num_tags is not None]
            tag_sizes = [d.tag_px for d in view_detections if d.tag_px is not None]
            report["aprilgrid_stats"] = {
                "tags_per_view_min": int(min(tag_counts)) if tag_counts else 0,
                "tags_per_view_median": float(np.median(tag_counts)) if tag_counts else 0.0,
                "tags_per_view_max": int(max(tag_counts)) if tag_counts else 0,
                "median_tag_px": round(float(np.median(tag_sizes)), 2) if tag_sizes else None,
            }

        if len(accepted) < args.min_views:
            report["warnings"] = report.get("coverage_before_filter", {}).get("warnings", [])
            for warning in report["warnings"]:
                print(f"警告: {warning}", file=sys.stderr)
            hint = (
                "请检查 TAG_COLS/TAG_ROWS、FIRST_ID、APRILGRID_MIN_TAGS、补光与清晰度，"
                if target.kind == "aprilgrid"
                else "请补拍清晰且角度多样的图片，"
            )
            raise ValueError(f"只有 {len(accepted)} 张有效图片，需要至少 {args.min_views} 张；"
                             f"{hint}不要为正式标定降低下限。")

        if target.kind == "aprilgrid":
            rot_check = target.check_aprilgrid_rotation(view_detections)
            if rot_check is not None:
                report["aprilgrid_rotation_check"] = rot_check

        active, rms, k, d, errors, pixel_rms = fit_with_filter(
            object_points, image_points, accepted, image_size, args, report["filter_rounds"])
        coverage = coverage_report([image_points[i] for i in active], image_size)
        views_out = []
        for i, err in zip(active, errors):
            item = {
                "file": str(accepted[i]),
                "reprojection_rms_px": err,
                "points_used": view_detections[i].num_points,
                "detection_file": detections[i],
            }
            if target.kind == "aprilgrid":
                item["tags_detected"] = view_detections[i].num_tags
                item["median_tag_px"] = (
                    None if view_detections[i].tag_px is None
                    else round(view_detections[i].tag_px, 2)
                )
            views_out.append(item)
        report.update(views_used=len(active), opencv_rms_px=float(rms), reprojection_rms_px=pixel_rms,
                      initial_rms_px=report["filter_rounds"][0]["reprojection_rms_px"],
                      camera_matrix=k.tolist(), distortion_coeffs=d, coverage=coverage,
                      warnings=coverage["warnings"],
                      removed_views=[v for round_ in report["filter_rounds"] for v in round_["removed"]],
                      views=views_out)
        for warning in report["warnings"]:
            print(f"警告: {warning}", file=sys.stderr)
        validate_solution(k, d, pixel_rms, image_size, args.model, target_kind=target.kind)
        if pixel_rms > args.max_rms_px:
            extra = (
                "（AprilGrid 请重点核对 TAG_SPACING、TAG_ROTATION_DEG 与 TAG_SIZE_M）"
                if target.kind == "aprilgrid" else ""
            )
            raise ValueError(f"总 RMS {pixel_rms:.4f} px > 门槛 {args.max_rms_px:.3f} px；"
                             f"检查报告、板面平整度、模糊和模型{extra}，补拍后重新标定")
        if args.strict_coverage and coverage["warnings"]:
            raise ValueError("覆盖检查未通过；严格模式不输出内参")
        intrinsics = [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]
        distortion_model = "radtan" if args.model == "pinhole" else "equidistant"
        camchain = {"cam0": {"camera_model": "pinhole", "intrinsics": intrinsics,
                             "distortion_model": distortion_model, "distortion_coeffs": d,
                             "resolution": list(image_size), "rostopic": args.topic}}
        report["status"] = "passed_with_warnings" if report["warnings"] else "passed"
        report["quality_note"] = "Low RMS alone does not prove calibration accuracy or VIO readiness."
        write_report(output, report)
        with (output / "camchain_intrinsics.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(camchain, stream, sort_keys=False, allow_unicode=True)
        print(f"有效图片 {len(active)}/{len(files)}；内参 {intrinsics}；RMS {pixel_rms:.4f} px")
        print(f"结果: {output}")
        return report
    except (ValueError, cv2.error, OSError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        write_report(output, report)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, cv2.error, OSError) as exc:
        print(f"标定失败: {exc}", file=sys.stderr)
        sys.exit(1)
