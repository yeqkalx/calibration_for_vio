# 相机–IMU 外参标定

这是一套自包含的 **单目相机–IMU 外参与时间偏移标定** 工程，包含：
1. **连续时间联合优化核心（`calib/`）**：无需 ROS / Ceres / Kalibr 运行时，支持 `pinhole + radtan`、棋盘格与 `tag36h11` AprilGrid、自动最长连续窗口选择与模糊/暗光诊断。
2. **现场硬件同步采集与体检（`capture/`）**：Orbbec RGB UVC（V4L2 直采 + 自动测光 + 手动曝光步进验证）+ MAVLink 飞控 IMU（`time_usec` 全段下包络时间锚定）。
3. **周边导出与对照工具（`tools/`）**：OpenVINS 配置导出、ROS1 bag 打包、Docker Kalibr 一键标定、自研 vs Kalibr 对照与独立角速度时移扫描、合成数据闭环自检。

---

## 1. 目录结构

```text
extrinsic/
├── extrinsics_config.py         # ★ 唯一总配置（会话名、相机/IMU采集、标定板、求解器参数）
├── calibrate_extrinsics.py      # ★ 离线标定主入口（支持直接运行 / --session / CSV / ROS bag）
│
├── calib/                       # 【核心算法包】纯算法实现，无硬件依赖
│   ├── spline.py                # 均匀三次平移 B-spline + SO(3) 累积三次旋转样条与解析导数
│   ├── data.py                  # 相机模型、棋盘/AprilGrid 检测、平面 PnP、CSV 与 ROS1/2 bag 读取
│   ├── windows.py               # 检出空洞诊断、清晰度/白格亮度判据、自动连续选窗
│   ├── solver.py                # 稀疏非线性最小二乘联合求解与 Kalibr/OpenVINS YAML 导出
│   └── acquisition.py           # 采集链路声明预检
│
├── capture/                     # 【现场采集包】相机 + 飞控 IMU 采集与体检
│   ├── check_camera.py          # ★ 采前体检 & 实时预览（10s 取景体检 / PnP 反解板参数 / --preview）
│   ├── record_session.py        # ★ 一键采集（相机+IMU 同步录制 → 生成 CSV → 自动求解外参）
│   ├── imu_recorder.py          # MAVLink IMU 录制 + time_usec 下包络时间基准拟合（可独立跑 10s 体检）
│   ├── capture_calibration.py   # V4L2 相机直采、自动测光与手动曝光步进验证
│   ├── image_health.py          # 画面亮度、检出率、切边与 PnP 质量诊断
│   └── mavlink_min.py           # 零依赖 MAVLink v1/v2 编解码器
│
├── tools/                       # 【工具与对照】导出、打包、Kalibr 对比与合成验证
│   ├── prepare_openvins.py      # 将标定结果导出为 OpenVINS 单目配置目录
│   ├── make_rosbag.py           # 将 data/<session> 打包为 ROS1 .bag（供 OpenVINS 或 Kalibr 使用）
│   ├── run_kalibr.py            # 一键打包并调用 Docker Kalibr 标定
│   ├── compare_results.py       # 自研 vs Kalibr 单次/全量跨会话对比 + 独立 |ω| 互相关时移扫描
│   ├── remap_imu_clock.py       # 基于实测同步锚点的跨时钟域仿射重映射
│   └── verify_synthetic.py      # 合成图像+IMU 端到端闭环自检（支持 --review 负对照回归）
│
├── configs/                     # 相机内参、标定板与预设参数 YAML
├── results/final/               # 最终标定结果与报告归档
└── tests/                       # 自动化单元测试套件（涵盖核心算法与采集全链路）
```

---

## 2. 最简四步工作流（PyCharm / 终端通用）

### 第一步：运行离线自检（无需插硬件）
```bash
cd extrinsic
python -m unittest discover -s tests -v
```

### 第二步：修改唯一配置文件 [`extrinsics_config.py`](extrinsics_config.py)
只需核对文件顶部的第 1 节：
- `SESSION_NAME = "session01"`（每次新采集换一个名字）
- `TARGET_TYPE = "aprilgrid"` 或 `"checkerboard"`（对应修改 [`configs/aprilgrid.yaml`](configs/aprilgrid.yaml) 或 [`configs/checkerboard.yaml`](configs/checkerboard.yaml) 的实测尺寸）
- `RESOLUTION = "1080p"`（或 `"720p"` / `"640x480"`，自动匹配对应的相机内参 YAML）

### 第三步：采前相机/标定板体检
在正式录制前，先对准标定板确认亮度、距离与标定板参数：
```bash
# 方式 A（推荐，PyCharm 直接运行）：录 10 秒并自动诊断亮度、检出率、PnP 残差，自动反解最优 tagSpacing / tag_rotation_deg
python capture/check_camera.py

# 方式 B：开启实时检测与 PnP HUD 预览窗口（按 Q 退出，S 存图）
python capture/check_camera.py --preview

# 方式 C：对已有图片目录做离线标定板几何体检
python capture/check_camera.py --dir data/session01/cam0
```

### 第四步：正式采集 + 自动求解外参
在 PyCharm 直接运行 [`capture/record_session.py`](capture/record_session.py)，或终端执行：
```bash
python capture/record_session.py
```
程序会自动：
1. 校验 Orbbec 相机 USB 身份（`2bc5:0511`）并自动探测正在发送 MAVLink 的飞控串口；
2. 先启动 IMU 录制，再锁定相机曝光并录制图像，统一锚定到宿主机 `CLOCK_REALTIME`；
3. 生成 `data/<SESSION_NAME>/images.csv` 与 `imu.csv` 并输出采集体检报告；
4. 若 `RUN_CALIBRATION_AFTER = True`，自动调用求解器将标定结果输出到 `results/<SESSION_NAME>/`。

---

## 3. 离线标定与常用工具命令

### 单独运行离线标定
```bash
# 直接按 extrinsics_config.py 当前配置的 SESSION_NAME 运行
python calibrate_extrinsics.py

# 指定已有会话名（或 --session latest 自动取最新会话；--list 查看所有会话）
python calibrate_extrinsics.py --session session14
python calibrate_extrinsics.py --list

# 从 ROS1 .bag 或 ROS2 bag 目录直接标定
python calibrate_extrinsics.py \
  --camera configs/gemini_pro_rgb.yaml \
  --target configs/aprilgrid.yaml \
  --bag bags/session14.bag \
  --cam-topic /cam0/image_raw --imu-topic /imu0 \
  --output results/session14_bag
```

### 导出 OpenVINS 配置与打包 ROS1 Bag
```bash
# 1. 将采集会话打包为 ROS1 bag（供 OpenVINS serial 模式使用）
python tools/make_rosbag.py --session session14

# 2. 基于模板导出冻结离线外参的 OpenVINS 配置目录
python tools/prepare_openvins.py \
  --calibration results/session14 \
  --template-config /path/to/open_vins/config/template \
  --imu-topic /imu0 --cam-topic /cam0/image_raw \
  --timeoffset-mode frozen \
  --output results/openvins_session14
```

### Docker Kalibr 标定与结果对比
```bash
# 一键打包 bag 并调用 Docker 中的 Kalibr 标定
python tools/run_kalibr.py session14

# 对比自研求解器与 Kalibr 在同一会话或全量会话上的外参差异
python tools/compare_results.py session14
python tools/compare_results.py --all

# 独立使用 PnP 与陀螺仪 |ω| 互相关扫描验证时间偏移
python tools/compare_results.py --scan-time-offset session14
```

---

## 4. 输出文件与坐标/时间约定

标定通过残差与激励检查后，在 `results/<session>/` 下生成：
- `camchain-kalibr.yaml`：包含 `T_cam_imu` 与 `timeshift_cam_imu`（Kalibr 格式）
- `camchain-openvins.yaml`：包含互逆的 `T_imu_cam`（OpenCV `FileStorage` 兼容 YAML）
- `report.json`：完整残差、激励指标、选窗诊断、零偏、重力方向与变换矩阵
- `extraction.json`、`detections/`、`residuals.npz`、`trajectory.npz`：逐帧检出与样条轨迹证据

**坐标与时间定义**：
```text
p_C = R_CI * p_I + p_CI = T_cam_imu * p_I
p_I = T_imu_cam * p_C   = inverse(T_cam_imu) * p_C
t_imu = t_cam + timeshift_cam_imu
```
`p_CI` 是 IMU 原点在相机坐标系中的位置；`timeshift_cam_imu > 0` 表示同一物理时刻的 IMU 时间戳数值比相机时间戳大。
