"""相机内参标定 · 唯一配置文件

现场只需修改本文件顶部的标定板类型、规格和会话名，即可在 PyCharm 或终端运行：
  1. capture_calibration.py   （锁曝光/自动测光并采集标定板图像，支持棋盘格与 AprilGrid）
  2. calibrate_intrinsics.py  （离线求解内参、迭代剔除异常视角、九宫格覆盖检查）
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ==================== 1. 现场必改：标定板和保存位置 ====================
# 标定板类型：
#   "checkerboard" -> 棋盘格（使用下方 COLS / ROWS / SQUARE_SIZE_M）
#   "aprilgrid"    -> Kalibr tag36h11 AprilGrid（使用下方 TAG_COLS / TAG_ROWS / TAG_SIZE_M / TAG_SPACING 等）
TARGET_TYPE = "checkerboard"    # 可选: "checkerboard" 或 "aprilgrid"
# 可选：直接复用外参/Kalibr 的标定板 YAML（如 ROOT.parent / "extrinsic" / "configs" / "aprilgrid.yaml"）；
# 非 None 时优先按该 YAML 读取板型与尺寸，设为 None 时使用下方参数。
TARGET_YAML = None

# --- 棋盘格参数（TARGET_TYPE == "checkerboard" 且 TARGET_YAML is None 时生效） ---
COLS = 8                        # 横向内角点数（不是格子数）
ROWS = 11                       # 纵向内角点数
SQUARE_SIZE_M = 0.03            # 相邻角点距离（米），请用卡尺实测

# --- AprilGrid 参数（TARGET_TYPE == "aprilgrid" 且 TARGET_YAML is None 时生效） ---
TAG_COLS = 6                    # 横向 tag 数量
TAG_ROWS = 6                    # 纵向 tag 数量
TAG_SIZE_M = 0.088              # 单个 tag 黑色外框边长（米），请用卡尺实测
TAG_SPACING = 0.56              # tag 间白色间隙 / TAG_SIZE_M（无量纲），卡尺量白隙再除以 TAG_SIZE_M
FIRST_ID = 0                    # 左下角起始 tag ID（ID 从左到右、从下到上递增）
# Kalibr 的 kalibr_create_target_pdf 画 tag 时写死 rotation=2（每张 tag 旋转了 180°），填 180；
# 自己用 cv2.aruco.generateImageMarker 拼的板子填 0。填错会导致几何方向反转、标定失败。
TAG_ROTATION_DEG = 180          # 只能是 0 / 90 / 180 / 270
APRILGRID_MIN_TAGS = 4          # 单帧最少检出 tag 数（允许板子部分出画以覆盖四角，建议 4~8）
APRIL_CORNER_WINDOW_PX = 4      # AprilTag 亚像素角点细化半窗（像素；远距离小 tag 可降至 2）

SESSION_NAME = "session01"      # 每轮采集改用新的会话名，避免混用不同参数

SESSION_DIR = ROOT / "data" / SESSION_NAME
IMAGES_DIR = SESSION_DIR / "cam0"
RESULT_DIR = SESSION_DIR / "intrinsics"

# ==================== 2. 相机与测光设置（与 VIO 模式一致） ====================
DEVICE = Path("/dev/video4")        # RGB 采集节点；会核对 USB 身份 2bc5:0511
WIDTH = 1920
HEIGHT = 1080
FPS = 30
MODEL = "pinhole"                   # 普通镜头用 "pinhole"；鱼眼镜头用 "fisheye"
BACKEND = "auto"                    # "auto" 优先 V4L2 直采，失败退回 OpenCV
EXPOSURE = 100                      # 单位 100 微秒（100 = 10 ms）
WHITE_BALANCE = 4600                # 固定白平衡 (K)
GAIN = 30
BRIGHTNESS = 0
CONTRAST = 32
SHARPNESS = 3
POWER_LINE_FREQUENCY = 1            # 0=关闭，1=50 Hz，2=60 Hz
BACKLIGHT_COMPENSATION = 0          # 必须关（0），防止 AE 漏网时二次改测光

SKIP_EXPOSURE_CHECK = False         # 锁定后执行手动曝光步进验证
AUTO_EXPOSE = True                  # 进场先自动测光，选定后全程锁死
AUTO_EXPOSE_TARGET = 100.0          # 目标灰度均值 (0..255)
AUTO_EXPOSE_MIN_MEAN = 45.0         # 低于此灰度拒绝采集
AUTO_EXPOSE_MAX_EXPOSURE_MS = 18.0
AUTO_EXPOSE_BLUR_BUDGET_PX = 3.0
AUTO_EXPOSE_EXPECTED_RATE_DPS = 12.0
AUTO_EXPOSE_MAX_GAIN = 40
ALLOW_DARK = False

# ==================== 3. 采集方式 ====================
AUTO_INTERVAL = 2.0         # 自动存图间隔（秒）；0 表示只按空格键存图
DURATION = 90.0             # 最长运行时间（秒）；0 表示手动退出
MAX_IMAGES = 40             # 最多保存张数；建议拍摄 25~40 个有效不同视角
NO_PREVIEW = False          # 现场桌面采集建议 False（看绿点与九宫格覆盖）
READ_RETRIES = 10
RETRY_INTERVAL = 0.5
WARMUP_FRAMES = 10

# ==================== 4. 质量检查阈值 ====================
MIN_VIEWS = 20              # 过滤后最少有效视角
SAVE_DETECTIONS = True      # 保存带角点与九宫格辅助线的检查图
OUTLIER_FACTOR = 2.0        # 剔除阈值 = max(OUTLIER_FACTOR * 误差中位数, OUTLIER_FLOOR_PX)
OUTLIER_FLOOR_PX = 1.0
MAX_RMS_PX = 0.5            # 总重投影 RMS 门槛（px）
MIN_COVERED_CELLS = 7       # 九宫格最少覆盖格数
STRICT_COVERAGE = False
FOCAL_MIN_RATIO = 0.2
FOCAL_MAX_RATIO = 5.0
MAX_FOCAL_ASPECT_ERROR = 0.15
RADTAN_LIMITS = (1.5, 1.5, 0.1, 0.1)
FISHEYE_FOCAL_RATIO = 0.5
CALIBRATION_ITERATIONS = 200
TOPIC = "/cam0/image_raw"
