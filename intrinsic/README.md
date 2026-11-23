# 相机内参标定（`intrinsic/`）

提供 Orbbec Gemini Pro RGB（UVC）相机的标定板图像采集与离线内参标定：
- **支持两种标定板**：棋盘格（`checkerboard`）与 Kalibr 标准 `tag36h11` AprilGrid（`aprilgrid`，支持部分遮挡/出画以充分覆盖画面边缘与四角）
- **支持两种镜头模型**：`pinhole + radtan` 与 `fisheye`（`equidistant`），直接导出 Kalibr / OpenVINS 兼容的 `camchain_intrinsics.yaml`
- **支持复用外参标定板 YAML**：可通过 `TARGET_YAML` 或 `--target` 直接读取 `extrinsic/configs/{aprilgrid,checkerboard}.yaml` 或 Kalibr 标定板 YAML

---

## 文件结构

| 文件 | 作用 |
|---|---|
| [`calibration_config.py`](calibration_config.py) | **唯一配置文件**：标定板类型与规格（棋盘格 / AprilGrid）、分辨率、曝光/自动测光及质量门槛 |
| [`target.py`](target.py) | **标定板共用模块**：棋盘格与 `tag36h11` AprilGrid 几何定义、检测、方向一致性校验与可视化（采集与离线标定共用） |
| [`capture_calibration.py`](capture_calibration.py) | **采集脚本**：校验 USB 身份（`2bc5:0511`）、自动测光 + 手动曝光步进验证、实时标定板检测与九宫格覆盖提示 |
| [`calibrate_intrinsics.py`](calibrate_intrinsics.py) | **离线标定脚本**：棋盘格/AprilGrid 检测、AprilGrid tag 方向自检、逐轮异常视角剔除、九宫格覆盖检查与 YAML 导出 |
| [`tests/test_intrinsics.py`](tests/test_intrinsics.py) | **离线单元测试**：涵盖棋盘格与 AprilGrid（`pinhole` 与 `fisheye`、部分可见、旋转校验、YAML/CLI 配置）合成闭环验证 |

---

## 两步使用方法（PyCharm / 终端通用）

### 第一步：修改配置并采集标定板图像
在 [`calibration_config.py`](calibration_config.py) 顶部选择标定板类型并核对实测尺寸：
- `TARGET_TYPE = "checkerboard"` 或 `"aprilgrid"`（也可将 `TARGET_YAML` 设为 `ROOT.parent / "extrinsic" / "configs" / "aprilgrid.yaml"` 直接复用外参配置）
- **棋盘格**：核对 `COLS`、`ROWS`（**内角点数**，不是黑白格子数）与 `SQUARE_SIZE_M`（卡尺实测单格边长，米）
- **AprilGrid**：核对 `TAG_COLS`、`TAG_ROWS`、`TAG_SIZE_M`（单个 tag 黑色外框边长，米）、`TAG_SPACING`（白隙 / `TAG_SIZE_M`）、`FIRST_ID`、`TAG_ROTATION_DEG`（**Kalibr PDF 打印板填 `180`，OpenCV `generateImageMarker` 自拼板填 `0`**）、`BLACK_BORDER`（**Kalibr 标准板填 `2`，OpenCV 自拼板填 `1`，`None` 为自动先试 `2` 再试 `1`**）与 `APRILGRID_MIN_TAGS`
- `SESSION_NAME`（每次采集使用新名称，如 `"session01"`）
- `WIDTH`、`HEIGHT`（须与后续外参标定及 VIO 运行分辨率一致）

然后运行采集：
```bash
cd intrinsic
python capture_calibration.py
```
- 棋盘格模式：预览窗出现**绿点**表示当前帧检出棋盘。
- AprilGrid 模式：预览窗绘制**绿色 tag 框 + 红色 0 号角点 + tag ID**，状态栏显示检出 tag 数与中位像素边长。
- 画面九宫格圆点变绿表示该区域已覆盖；建议从不同距离、不同倾斜角度拍摄 **25~40 张**，确保**画面四角与边缘**均有充分覆盖（AprilGrid 可让部分板面移出画面以贴近四角）。

### 第二步：运行离线内参标定
直接按 `calibration_config.py` 运行：
```bash
python calibrate_intrinsics.py
```

也支持命令行覆盖路径或标定板参数：
```bash
# 方式 A：直接复用外参/Kalibr 的 AprilGrid YAML
python calibrate_intrinsics.py \
  --images data/session01/cam0 \
  --target ../extrinsic/configs/aprilgrid.yaml \
  --output data/session01/intrinsics

# 方式 B：命令行指定 AprilGrid 参数
python calibrate_intrinsics.py \
  --images data/session01/cam0 \
  --target-type aprilgrid \
  --tag-cols 6 --tag-rows 6 --tag-size 0.088 --tag-spacing 0.3 \
  --tag-rotation-deg 180 --black-border 2 --min-tags 4 \
  --output data/session01/intrinsics

# 方式 C：命令行指定棋盘格参数
python calibrate_intrinsics.py \
  --images data/session01/cam0 \
  --target-type checkerboard \
  --cols 8 --rows 11 --square-size 0.03 \
  --output data/session01/intrinsics
```

---

## 离线单元测试（无需硬件）

```bash
cd intrinsic
python -m unittest discover -s tests -v
```

---

## 输出说明

标定通过后在 `data/<SESSION_NAME>/intrinsics/` 下生成：
- `camchain_intrinsics.yaml`：包含 `cam0` 的 `intrinsics: [fx, fy, cx, cy]`、`distortion_coeffs` 与 `resolution`，可直接复制到 `extrinsic/configs/` 作为外参标定的相机内参输入。
- `report.json`：标定板配置、重投影 RMS、AprilGrid tag 数量/尺寸/方向校验结果、每轮剔除的异常视角、九宫格覆盖统计及各张图片的重投影误差。
- `detections/`：叠加了检出角点（或 AprilGrid tag 框/0 号角点/ID）与九宫格辅助线的检查图。
