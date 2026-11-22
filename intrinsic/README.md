# 相机内参标定（`intrinsic/`）

提供 Orbbec Gemini Pro RGB（UVC）相机的棋盘格图像采集与离线内参标定（支持 `pinhole + radtan` 与 `fisheye` 模型，直接导出 Kalibr / OpenVINS 兼容的 `camchain_intrinsics.yaml`）。

---

## 文件结构

| 文件 | 作用 |
|---|---|
| [`calibration_config.py`](calibration_config.py) | **唯一配置文件**：棋盘内角点数、格边长、分辨率、曝光/自动测光及质量门槛 |
| [`capture_calibration.py`](capture_calibration.py) | **采集脚本**：校验 USB 身份（`2bc5:0511`）、自动测光 + 手动曝光步进验证、九宫格覆盖提示 |
| [`calibrate_intrinsics.py`](calibrate_intrinsics.py) | **离线标定脚本**：棋盘角点检测、逐轮异常视角剔除、九宫格覆盖检查与 YAML 导出 |

---

## 两步使用方法（PyCharm / 终端通用）

### 第一步：修改配置并采集棋盘图像
在 [`calibration_config.py`](calibration_config.py) 中核对：
- `COLS`、`ROWS`（棋盘**内角点数**，不是黑白格子数）
- `SQUARE_SIZE_M`（卡尺实测单格边长，单位米）
- `SESSION_NAME`（每次采集使用新名称，如 `"session01"`）
- `WIDTH`、`HEIGHT`（须与后续外参标定及 VIO 运行分辨率一致）

然后运行采集：
```bash
cd intrinsic
python capture_calibration.py
```
- 预览窗中出现**绿点**表示当前帧检出棋盘，画面九宫格圆点变绿表示该区域已覆盖。
- 建议从不同距离、不同倾斜角度拍摄 **25~40 张**，并确保**画面四角与边缘**均有角点覆盖。

### 第二步：运行离线内参标定
```bash
python calibrate_intrinsics.py
```
也支持命令行覆盖路径或参数：
```bash
python calibrate_intrinsics.py \
  --images data/session01/cam0 \
  --cols 8 --rows 11 --square-size 0.03 \
  --output data/session01/intrinsics
```

---

## 输出说明

标定通过后在 `data/<SESSION_NAME>/intrinsics/` 下生成：
- `camchain_intrinsics.yaml`：包含 `cam0` 的 `intrinsics: [fx, fy, cx, cy]`、`distortion_coeffs` 与 `resolution`，可直接复制到 `extrinsic/configs/` 作为外参标定的相机内参输入。
- `report.json`：重投影 RMS、每轮剔除的异常视角、九宫格覆盖统计及各张图片的重投影误差。
- `detections/`：叠加了检出角点与九宫格辅助线的检查图。
