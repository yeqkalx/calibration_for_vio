"""相机–IMU 外参标定 · 唯一总配置文件

本文件集中管理「现场采集 + 标定板与相机设置 + 离线标定求解」的所有默认参数。
在 PyCharm 中只需修改本文件顶部的会话名与标定板类型，即可直接运行：
  - capture/check_camera.py     （采前取景/曝光/标定板体检，或 --preview 实时预览）
  - capture/record_session.py   （一键采集相机 + IMU，并在采集后自动标定）
  - calibrate_extrinsics.py     （对已有 session 或 ROS bag 单独运行离线标定）
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

# ==========================================================================
# 1. 当前会话与标定板（现场最常改的几项）
# ==========================================================================
SESSION_NAME = "session6"          # 每次重录或换板请改新的名字，避免覆盖旧数据
CAPTURE_PURPOSE = "calibration"     # "calibration" = 标定采集；"vio" = 跑场景 VIO 数据采集
RUN_CALIBRATION_AFTER = True        # True = 采集完成后自动调用离线标定求解
REUSE_EXISTING_SESSION = False      # False = 目录非空时拒绝覆盖

# 标定板类型：
#   "aprilgrid"    -> configs/aprilgrid.yaml（tagCols/tagRows/tagSize/tagSpacing）
#   "checkerboard" -> configs/checkerboard.yaml（cols/rows 内角点数 / square_size）
TARGET_TYPE = "aprilgrid"
if TARGET_TYPE not in ("checkerboard", "aprilgrid"):
    raise ValueError('TARGET_TYPE 只能是 "checkerboard" 或 "aprilgrid"')
TARGET_YAML = ROOT / "configs" / f"{TARGET_TYPE}.yaml"

SESSION_DIR = ROOT / "data" / SESSION_NAME
RESULTS_DIR = ROOT / "results" / SESSION_NAME

# ==========================================================================
# 2. 相机与自动测光参数（Orbbec Gemini Pro RGB UVC）
# ==========================================================================
CAMERA_USB_ID = "2bc5:0511"         # USB Vendor:Product，拒绝误用笔记本内置摄像头
DEVICE = Path("/dev/video4")        # 默认 V4L2 节点（record_session 会按 USB ID 自动扫描）
BACKEND = "auto"                    # "auto" 优先 V4L2 直采，失败回退 OpenCV；"v4l2" 禁止回退

# 分辨率与相机内参严格绑定：
#   "1080p"   -> configs/gemini_pro_rgb.yaml (1920x1080)
#   "720p"    -> configs/gemini_pro_rgb_720p.yaml (1280x720)
#   "640x480" -> configs/gemini_pro_rgb_640x480.yaml (640x480)
RESOLUTION = "1080p"
_RESOLUTION_TABLE = {
    "1080p": (1920, 1080, ROOT / "configs" / "gemini_pro_rgb.yaml"),
    "720p": (1280, 720, ROOT / "configs" / "gemini_pro_rgb_720p.yaml"),
    "640x480": (640, 480, ROOT / "configs" / "gemini_pro_rgb_640x480.yaml"),
}
if RESOLUTION not in _RESOLUTION_TABLE:
    raise ValueError(f"RESOLUTION 只能是 {sorted(_RESOLUTION_TABLE)} 之一，当前为 {RESOLUTION!r}")
WIDTH, HEIGHT, CAMERA_YAML = _RESOLUTION_TABLE[RESOLUTION]

FPS = 30
EXPOSURE = 100                      # 单位 100 微秒（100 = 10 ms）
WHITE_BALANCE = 4600                # 固定色温 (K)
GAIN = 30
BRIGHTNESS = 0
CONTRAST = 32
SHARPNESS = 3
POWER_LINE_FREQUENCY = 1            # 0=关, 1=50Hz, 2=60Hz
BACKLIGHT_COMPENSATION = 0          # 必须关（0），防止 AE 漏网时二次改测光

# 进场自动测光与手动曝光步进验证
SKIP_EXPOSURE_CHECK = False
AUTO_EXPOSE = True
AUTO_EXPOSE_TARGET = 100.0          # 目标灰度均值 (0..255)
AUTO_EXPOSE_MIN_MEAN = 45.0         # 最低可用灰度均值，低于此值拒绝采集
AUTO_EXPOSE_MAX_EXPOSURE_MS = 18.0  # 曝光绝对上限 (ms)
AUTO_EXPOSE_BLUR_BUDGET_PX = 3.0    # 运动模糊预算 (px)：t_exp <= blur / (fx * ω)
AUTO_EXPOSE_EXPECTED_RATE_DPS = 100.0               # VIO 模式预期角速度 (°/s)
CALIBRATION_AUTO_EXPOSE_EXPECTED_RATE_DPS = 12.0    # 标定模式预期角速度 (°/s)
AUTO_EXPOSE_MAX_GAIN = 40           # 测光增益上限
ALLOW_DARK = False
SHOW_PREVIEW = False                # 正式采集关预览以保 30fps；需要看画面请跑 capture/check_camera.py

# 单机采图底层默认参数（供 capture_calibration.py 独立命令行调用时作为默认值）
COLS = 8
ROWS = 11
AUTO_INTERVAL = 2.0
DURATION = 90.0
MAX_IMAGES = 40
NO_PREVIEW = False
READ_RETRIES = 10
RETRY_INTERVAL = 0.5
WARMUP_FRAMES = 10

# ==========================================================================
# 3. 飞控 IMU 采集参数（MAVLink HIGHRES_IMU）
# ==========================================================================
IMU_PORT = "auto"                   # "auto" 自动探测在发 MAVLink 的 ttyACM*/ttyUSB*
IMU_REQUEST_HZ = 200                # 请求 HIGHRES_IMU 频率 (Hz)
IMU_STARTUP_DISCARD_S = 2.0         # 开流初期丢弃秒数
IMU_BAUD = 115200
IMU_PRIME_S = 2.0                   # 相机开始前提前录 IMU 秒数
IMU_TAIL_S = 0.5                    # 相机结束后顺延录 IMU 秒数
IMU_CHECK_SECONDS = 10.0            # 单独运行 capture/imu_recorder.py 体检时的录制时长
DURATION_S = 40.0                   # 正式采集时长 (s)

# Kalibr 连续时间 IMU 噪声默认值（2026-11-23 MicoAir743v2 0.5h 静置实测，供 tools/run_kalibr.py 使用）
IMU_NOISE_DEFAULTS = {
    "update_rate": 191.3,
    "gyroscope_noise_density": 9.7223e-05,
    "accelerometer_noise_density": 1.0694e-03,
    "gyroscope_random_walk": 1.0000e-05,
    "accelerometer_random_walk": 9.2162e-05,
}

# ==========================================================================
# 4. 离线标定求解器设置
# ==========================================================================
@dataclass
class Settings:
    knot_spacing_s: float = 0.04        # 均匀三次样条结点间距 (s)
    max_duration_s: float = 30.0        # 单次标定连续窗口上限 (s)
    start_s: float = 0.0                # 相对于第一张输入图像的起始秒数
    detect_rate_hz: float = 0.0         # 检测抽样率 Hz（0 = 逐帧检测，避免预抽帧切短连续段）
    camera_rate_hz: float = 15.0        # 进入求解器的图像率上限 Hz（应 <= 相机帧率的一半）
    auto_window: int = 1                # 1 = 检出有空洞时自动选最长合格连续段；0 = 整段拒绝
    imu_rate_hz: float = 300.0          # 进入优化的最高原始 IMU 采样率 Hz（阈值 3.3ms，保留飞控 3.73/7.46ms 全部 ~192Hz 原始样本）
    max_offset_s: float = 0.10          # t_imu = t_cam + offset 搜索边界 (s)
    max_translation_m: float = 0.50     # T_cam_imu 各轴平移边界 (m)
    image_sigma_px: float = 0.5         # 单坐标像素标准差（白化权重）
    gyro_sigma_rad_s: float = 0.01      # 单个 IMU 陀螺样本标准差（匹配手持样条残差尺度，避免 soft_l1 大面积饱和）
    accel_sigma_m_s2: float = 0.10      # 单个 IMU 加计样本标准差（匹配手持样条残差尺度，避免 soft_l1 大面积饱和）
    gravity_m_s2: float = 9.80665       # 当地重力模长 (m/s²)
    gravity_norm_sigma: float = 0.005   # 重力模长约束标准差 (m/s²)
    max_gyro_bias_rad_s: float = 0.5
    max_accel_bias_m_s2: float = 2.0
    min_frames: int = 35
    min_duration_s: float = 5.0
    max_camera_gap_s: float = 0.35      # 相邻有效检测帧最大允许间隔 (s)
    max_imu_gap_s: float = 0.05
    min_rotation_std_rad_s: float = 0.05
    min_lever_singular: float = 0.15
    min_target_depth_m: float = 0.01
    depth_sigma_m: float = 0.001
    max_pnp_rms_px: float = 2.0
    max_reprojection_rms_px: float = 2.0
    max_gyro_rms_rad_s: float = 0.04
    max_accel_rms_m_s2: float = 0.40
    min_tags: int = 8                   # AprilGrid 单帧最少 tag 数
    april_corner_window_px: int = 4     # AprilTag 亚像素细化半窗 (px)
    max_points_per_frame: int = 150
    init_max_nfev: int = 80
    joint_max_nfev: int = 300
    optimizer_ftol: float = 1e-5
    optimizer_xtol: float = 1e-7
    optimizer_gtol: float = 1e-6
    robust_scale: float = 2.5
    seed: int = 7
    save_detection_every: int = 20
    verbose: int = 1

    def __post_init__(self):
        """Normalize numeric YAML strings; reject bools, NaN and fractional counts."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(f"配置项 {field.name}: 需要数值，收到 {value!r}")
            try:
                number = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"配置项 {field.name}: 无效数值 {value!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"配置项 {field.name}: 必须为有限数值")
            if field.type is int or field.type == "int":
                if not number.is_integer():
                    raise ValueError(f"配置项 {field.name}: 必须为整数")
                number = int(number)
            setattr(self, field.name, number)
        nonnegative = {"start_s", "detect_rate_hz"}
        for field in fields(self):
            value = getattr(self, field.name)
            is_float = field.type is float or field.type == "float"
            if is_float and (value < 0 if field.name in nonnegative else value <= 0):
                raise ValueError(f"配置项 {field.name}: 必须 "
                                 f"{'>=0' if field.name in nonnegative else '>0'}")
        minima = {"min_frames": 3, "min_tags": 1, "april_corner_window_px": 1,
                  "max_points_per_frame": 0, "init_max_nfev": 1, "joint_max_nfev": 1,
                  "seed": 0, "save_detection_every": 0, "verbose": 0}
        for key, minimum in minima.items():
            if getattr(self, key) < minimum:
                raise ValueError(f"配置项 {key}: 必须 >= {minimum}")
        if self.verbose > 2 or self.seed >= 2**32:
            raise ValueError("配置项 verbose 必须为 0/1/2；seed 必须小于 2**32")
        if self.auto_window not in (0, 1):
            raise ValueError("配置项 auto_window 只能是 0 或 1")
        for key in ("optimizer_ftol", "optimizer_xtol", "optimizer_gtol"):
            if getattr(self, key) <= sys.float_info.epsilon:
                raise ValueError(f"配置项 {key}: 必须大于机器精度")


# 离线标定默认路径（自动跟随顶部的 SESSION_NAME / RESOLUTION / TARGET_TYPE，也可通过 CLI 覆盖）
PATHS = {
    "camera": CAMERA_YAML,
    "target": TARGET_YAML,
    "images": SESSION_DIR / "images.csv",
    "imu": SESSION_DIR / "imu.csv",
    "bag": None,
    "output": RESULTS_DIR,
}
CAM_TOPIC = "/cam0/image_raw"
IMU_TOPIC = "/imu0"
IMAGE_TIMESTAMP_UNIT = "ns"
IMU_TIMESTAMP_UNIT = "ns"
SETTINGS = Settings(camera_rate_hz=15, max_pnp_rms_px=3.5)
