# calibration_for_vio

pure-uav 项目 VIO 部分的标定代码（独立仓库，长期维护）。

| 目录 | 内容 |
|---|---|
| `extrinsic/` | 相机–IMU 外参与时间偏移标定（核心算法、现场采集链路、Kalibr 对照、OpenVINS 导出） |
| `intrinsic/` | 相机内参标定（棋盘格，pinhole / fisheye） |
| `imu_noise/` | IMU 静态噪声采集与 Allan 方差分析 |
| `imu_accel/` | IMU 加计零偏/尺度标定 |

## 快速开始

```bash
conda activate vins
cd extrinsic
python -m unittest discover -s tests -v    # 标定核心回归
python capture/selftest_capture.py         # 采集链路自检，不需要硬件
```

外参标定的完整流程见 [`extrinsic/README.md`](extrinsic/README.md) 与
[`extrinsic/capture/README.md`](extrinsic/capture/README.md)。

## 来源与维护

本仓库由 `pure-uav` 的 `src/vio/{extrinsic,intrinsic,imu_noise,imu_accel}` 拆分而来，
原始标定工作区为 `/home/kxy/calib`。`extrinsic/capture/` 采集链路已与标定工作区
2026-11-22 的最新版本同步（MAVLink 1/2、IMU 启动丢弃、标定测光门槛、采集后自动求解）。
