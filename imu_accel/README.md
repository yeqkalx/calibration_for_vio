# IMU 加速度计零偏 / 尺度 / 三轴不正交标定（`imu_accel/`）

估计加速度计确定性误差矩阵 $T_a = F^{-1}$（尺度因子与三轴非正交）及零偏 $b$，直接生成可填入 OpenVINS `kalibr_imu_chain.yaml` 的标定块：
$$a_{\text{corrected}} = F \cdot (a_{\text{raw}} - b), \quad \|a_{\text{corrected}}\| = g$$

---

## 文件结构

| 文件 | 作用 |
|---|---|
| [`record_accel_calib.py`](record_accel_calib.py) | **六面节奏引导采集**：按顺序倒计时提示翻面（肚子→背→机头→机尾→右→左），录制 `HIGHRES_IMU` 到 `imu_accel/data/`，完成后自动调用 `accel_calib.py` |
| [`accel_calib.py`](accel_calib.py) | **离线求解器**：支持 6 面正交法（`--mode ferraris`）与 $\ge 10$ 姿态椭球拟合法（`--mode ellipsoid`），内置 `--self-test` 合成数据验证 |

---

## 快速开始（PyCharm / 终端通用）

```bash
cd imu_accel

# ① 离线自检（无需硬件，验证 Ferraris 六面法与 Ellipsoid 椭球拟合还原精度）
python accel_calib.py --self-test

# ② 空跑一遍翻面提示节奏（不录音，约 20 秒，熟悉翻面顺序）
python record_accel_calib.py --dry-run

# ③ 正式采集 + 自动标定（约 125 秒，数据保存在 imu_accel/data/accelcal_<时间戳>.csv）
python record_accel_calib.py

# ④ 对最新采集的 CSV 单独重新求解（无需手填文件名，自动取最新 accelcal_*.csv）
python accel_calib.py
```

---

## 常用参数

### `record_accel_calib.py`（采集）
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hold S` | `15` | 每个面保持静止的秒数（求解要求静止 $\ge 4\,\text{s}$） |
| `--trans S` | `6` | 两次静止之间的换面倒计时秒数 |
| `--prep S` | `5` | 开始前准备倒计时秒数 |
| `--dry-run` | 关 | 不连接飞控，仅快速走一遍提示节奏 |
| `--no-calibrate` | 关 | 采集结束后不自动运行求解 |
| `--port P` | `auto` | 飞控串口（默认自动探测发送 MAVLink 的端口） |

### `accel_calib.py`（求解）
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--csv PATH` | 自动取最新 | 指定 IMU CSV 文件（不传则自动读取 `imu_accel/data/` 下最新 `accelcal_*.csv`） |
| `--session DIR` | — | 直接读取采集会话目录下的 `imu.csv` |
| `--mode` | `auto` | `ferraris`（6 个正交面）、`ellipsoid`（$\ge 10$ 个含倾斜姿态）、`auto`（按检测到的姿态数自动选择） |
| `--order` | `belly,back,nose,tail,right,left` | `ferraris` 模式下 6 段静止对应的摆放顺序 |
| `--g` | `9.81` | 当地重力加速度模长（需与 OpenVINS `estimator_config.yaml` 中 `gravity_mag` 一致） |
| `--save PATH` | 关 | 将结果矩阵保存为 `.npz`（包含 `F`、`b`、`Ta`） |
