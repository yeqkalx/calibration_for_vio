"""相机–IMU 外参标定 · 现场采集配置

PyCharm 用法：选 TARGET_TYPE，核对对应的标定板 YAML，然后运行 `record_session.py`。

`record_session.py` 会自动：
  1) 检查相机（USB 2bc5:0511）和飞控串口到底在不在
  2) 一边采图、一边录 IMU —— 两路都用宿主机的 CLOCK_REALTIME，
     所以天然落在同一个时钟域，不需要额外同步
  3) 收尾生成 `images.csv` / `imu.csv`，正好是 `calibrate_extrinsics.py` 要的格式
  4) 采集通过后自动用本次数据求解外参，结果写到 results/<SESSION_NAME>/

不必修改 `extrinsics_config.py` 的 PATHS；入口会显式传入本次采集的文件。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # extrinsic/

# ==========================================================================
# 1. 这次录到哪儿
# ==========================================================================
# 每次重录或换板都用一个新的会话名，不覆盖旧数据或标定结果。
SESSION_NAME = "session01"

SESSION_DIR = ROOT / "data" / SESSION_NAME
RESULTS_DIR = ROOT / "results" / SESSION_NAME

# 目标目录里已经有上一次的采集结果时，是否允许覆盖。
# False = 直接报错停下（防止把上一轮的图和 IMU 混在一起）。
REUSE_EXISTING_SESSION = False

# ==========================================================================
# 2. 相机
# ==========================================================================
CAMERA_USB_ID = "2bc5:0511"        # Orbbec Gemini Pro 的 RGB（UVC）。别的值会拒绝运行

# 分辨率决定用哪份内参 —— 两者必须一致，工具会比对并报错，不会自动缩放。
#   "1080p"   → configs/gemini_pro_rgb.yaml（1920x1080）
#   "720p"    → configs/gemini_pro_rgb_720p.yaml（1280x720）
#   "640x480" → configs/gemini_pro_rgb_640x480.yaml
#
# 【2026-10-15 实测】1080p 的保存循环约 12 fps，720p 在关掉预览后可到 ~30 fps。
# 两种都能标定，但必须和内参文件严格对应；640x480 的内参来自已有 SDK 配置、
# 畸变沿用 1080p 系数，未用棋盘独立验证，只作为可用选项保留。
RESOLUTION = "1080p"

_RESOLUTION_TABLE = {
    "1080p": (1920, 1080, ROOT / "configs" / "gemini_pro_rgb.yaml"),
    "720p": (1280, 720, ROOT / "configs" / "gemini_pro_rgb_720p.yaml"),
    "640x480": (640, 480, ROOT / "configs" / "gemini_pro_rgb_640x480.yaml"),
}

if RESOLUTION not in _RESOLUTION_TABLE:
    raise ValueError(f"RESOLUTION 只能是 {sorted(_RESOLUTION_TABLE)} 之一，现在是 {RESOLUTION!r}")

# 这三项由 RESOLUTION 推导出来，不要在别处手改
WIDTH, HEIGHT, CAMERA_YAML = _RESOLUTION_TABLE[RESOLUTION]

FPS = 30                # 请求帧率。真机 30 fps 时，标定侧的 camera_rate_hz 要设成 30
EXPOSURE = 100          # 单位 100 微秒；100 = 10 ms。对着 iPad 屏幕如果发白就往下调（如 40）
WHITE_BALANCE = 4600    # 固定白平衡（K）
GAIN = 30               # 增益，大了噪声大

# 【2026-10-20 统一口径】下面这几项以前只写在 vendor/calibration_config.py 里，
# 而预览/采集/vio 又各有各的默认值 —— "同一个相机换个入口就换个亮度"有一部分就是这里。
# 现在这份文件是唯一真源：record_session 会显式把它们传给采集脚本，预览也读这里。
BRIGHTNESS = 0          # 亮度偏置，一般不改
CONTRAST = 32           # 对比度；避免黑白细节截断
SHARPNESS = 3           # 锐化；过强会影响角点定位
POWER_LINE_FREQUENCY = 1        # 0=关，1=50 Hz，2=60 Hz
BACKLIGHT_COMPENSATION = 0      # 逆光补偿默认是 1，会在 AE 漏网时二次改测光，必须关

# 【2026-10-20】采集脚本锁定控件后会做一次"曝光步进测试"：把曝光翻倍，看图像亮度
# 是不是也跟着翻倍 —— G_CTRL 读回只证明驱动缓存，证明不了固件真的照做了。
# 不通过就拒绝采集。只有确认是误判（比如场景在测试中变了）时才把这里改成 True。
SKIP_EXPOSURE_CHECK = False

# 【2026-10-20】以前 EXPOSURE/GAIN 是写死的，那组数字只照"有日光的那次"定的，
# 换到只有室内灯的场地就不够。现在进场先自动测光（把曝光顶到模糊预算、还不够再加增益），
# 选定后整段采集锁死不动 —— 可复现靠"锁定"，不是靠"写死"。
AUTO_EXPOSE = True

# 【2026-10-20】这次录的东西是给谁用的：
#   "vio"         —— 拿着相机走场景跑 VIO。画面里没有标定板，
#                    所以"AprilGrid 检出 0/20"是正常的，报告里会跳过这一段。
#   "calibration" —— 对着棋盘格或 AprilGrid 采集外参标定数据。
CAPTURE_PURPOSE = "calibration"
AUTO_EXPOSE_TARGET = 100.0        # 目标灰度均值
AUTO_EXPOSE_MIN_MEAN = 45.0       # 达不到就拒绝采集，别浪费时间
# 曝光上限 = 运动模糊预算。走 0.65 m/s、2 m 外的特征：1 ms ≈ 1.4 px 模糊，
# 所以 18 ms ≈ 25 px 已经很多了；可接受的上限大约 12~15 ms。
# 静态拍标定板时模糊不重要，可以调到 30。
AUTO_EXPOSE_MAX_EXPOSURE_MS = 18.0   # 绝对上限（不再当模糊预算用，见下面三项）

# 【2026-10-21】曝光上限必须由「运动模糊预算」反推，不能写死：
#     blur_px = f * ω * t_exp
# session28/29/30 就是被写死的 18 ms 顶满录的（增益也顶到 100），实测
# KLT 两帧存活率 67%（模糊 >20 px 时 38%）、成熟特征 62 个/帧，
# 三角化失败 55%、MSCKF 更新率 26%，最后尺度偏 5 倍。
# 对比 session24（10 ms / 增益 10）：存活率 77%、成熟特征 217 个、更新率 51%，能跑对。
AUTO_EXPOSE_BLUR_BUDGET_PX = 3.0      # 允许的运动模糊（像素）；>12 px 跟踪就崩
AUTO_EXPOSE_EXPECTED_RATE_DPS = 100.0  # 预期最大角速度（度/秒）——按你的机体填
CALIBRATION_AUTO_EXPOSE_EXPECTED_RATE_DPS = 12.0  # 对着静态标定板转动的预期角速度
AUTO_EXPOSE_MAX_GAIN = 40             # 增益上限：实测噪声 σ≥8 时存活率从 78% 掉到 43%
ALLOW_DARK = False
SHOW_PREVIEW = False

# ==========================================================================
# 3. 标定板 —— 必须和标定那一步用同一份，行列数要和你实际摆的板子完全一致
# ==========================================================================
# "checkerboard": configs/checkerboard.yaml，cols/rows 是内角点数，square_size 是格边长（米）。
# "aprilgrid": configs/aprilgrid.yaml，tagCols/tagRows 是 tag 个数，tagSize 是单 tag 黑框外沿（米），
# tagSpacing 是白隙 / tagSize。两份文件里的尺寸都必须按实物核对。
TARGET_TYPE = "aprilgrid"
if TARGET_TYPE not in ("checkerboard", "aprilgrid"):
    raise ValueError('TARGET_TYPE 只能是 "checkerboard" 或 "aprilgrid"')
TARGET_YAML = ROOT / "configs" / f"{TARGET_TYPE}.yaml"

# ==========================================================================
# 4. 飞控 IMU
# ==========================================================================
IMU_PORT = "auto"       # "auto" = 在 ttyACM*/ttyUSB* 里自动找会讲 MAVLink 的那个
                        # 也可以写死，例如 "/dev/ttyACM0"
IMU_REQUEST_HZ = 200    # 请求 HIGHRES_IMU。ArduPilot 上限约 0.8×SCHED_LOOP_RATE，
                        # 请求 200 实际可能只有 50 —— 以采集后打印的实测速率为准
IMU_STARTUP_DISCARD_S = 2.0   # 打开串口后先丢弃这么多秒，避开开流瞬间的不稳定样本
IMU_BAUD = 115200       # USB CDC 下这个值不生效（走 USB 全速），真实串口链路才需要

# 相机开始前后各多录一点 IMU，保证两路时间轴有重叠（标定工具要按重叠区取窗口）
IMU_PRIME_S = 2.0
IMU_TAIL_S = 0.5
IMU_CHECK_SECONDS = 10.0   # 单独点 imu_recorder.py 做体检时录多久

# ==========================================================================
# 5. 采集时长
# ==========================================================================
# 标定侧硬门槛：至少 35 帧有效检出、且时间跨度 >= 5 秒，所以别低于 8 秒。
# 第一次冒烟测试建议 8~10 秒；正式采集 15~30 秒，并且另录 2~3 个独立窗口做复核。
DURATION_S = 40.0

# ==========================================================================
# 6. 录完要不要顺手把标定也跑了
# ==========================================================================
# 只在 CAPTURE_PURPOSE="calibration" 时生效。
# True  = 采集 → 标定 一次点击全跑完（适合冒烟测试）
# False = 只采集，标定另外点（适合正式采集，方便分开排错）
RUN_CALIBRATION_AFTER = True
