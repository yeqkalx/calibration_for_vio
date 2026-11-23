# calibration_for_vio

`pure-uav` 项目 VIO 传感器标定工具集（Orbbec Gemini Pro RGB + CUAV 7 Nano）。

## 推荐标定顺序与模块导航

| 顺序 | 目录 | 功能说明 | 核心入口 |
|---|---|---|---|
| **1** | [`intrinsic/`](intrinsic/README.md) | **相机内参标定**（棋盘格 / AprilGrid，`pinhole + radtan` / `fisheye`） | `capture_calibration.py` → `calibrate_intrinsics.py` |
| **2** | [`imu_noise/`](imu_noise/README.md) | **IMU 静态随机噪声标定**（重叠式 Allan 方差：噪声密度与随机游走） | `record_static_imu.py` → `analyze_static_noise.py` |
| **3** | [`imu_accel/`](imu_accel/README.md) | **IMU 加速度计确定性标定**（六面法 / 椭球拟合：零偏、尺度因子与三轴非正交 $T_a$） | `record_accel_calib.py` → `accel_calib.py` |
| **4** | [`extrinsic/`](extrinsic/README.md) | **相机–IMU 外参与时间偏移标定**（连续时间样条联合优化、现场同步采集、Kalibr 对照、OpenVINS 导出） | `capture/check_camera.py` → `capture/record_session.py` → `calibrate_extrinsics.py` |
