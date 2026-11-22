# calibration_for_vio

pure-uav 项目 VIO 部分的标定代码。

| 目录 | 内容 |
|---|---|
| `extrinsic/` | 相机–IMU 外参与时间偏移标定（核心算法、现场采集链路、Kalibr 对照、OpenVINS 导出） |
| `intrinsic/` | 相机内参标定（棋盘格，pinhole / fisheye） |
| `imu_noise/` | IMU 静态噪声采集与 Allan 方差分析 |
| `imu_accel/` | IMU 加计零偏/尺度标定 |
