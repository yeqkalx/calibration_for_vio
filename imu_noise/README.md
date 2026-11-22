# IMU 静态噪声标定（Allan 方差 · `imu_noise/`）

通过采集飞控 MAVLink `HIGHRES_IMU` 200 Hz 静置数据并计算重叠式 Allan 偏差，直接为 Kalibr / OpenVINS 提供连续时间噪声参数：
- `gyroscope_noise_density` ($\sigma_g$, $\text{rad/s/\sqrt{Hz}}$)
- `gyroscope_random_walk` ($K_g$, $\text{rad/s}^2/\sqrt{\text{Hz}}$)
- `accelerometer_noise_density` ($\sigma_a$, $\text{m/s}^2/\sqrt{\text{Hz}}$)
- `accelerometer_random_walk` ($K_a$, $\text{m/s}^3/\sqrt{\text{Hz}}$)

---

## 文件结构

| 文件 | 作用 |
|---|---|
| [`record_static_imu.py`](record_static_imu.py) | **静置采集脚本**：流式写盘（防长时内存溢出）、双时间轴记录（`fc_time_us` + `host_ns`）、断线重连 |
| [`analyze_static_noise.py`](analyze_static_noise.py) | **Allan 方差分析脚本**：运行前先过合成自检，自动识别白噪声 $-1/2$ 斜率区与随机游走 $+1/2$ 斜率区，导出图表与 `kalibr_imu_chain.yaml` |
| [`serial_port.py`](serial_port.py) | 跨平台串口抽象（Linux `termios` 零依赖 / Windows `pyserial`）与 MAVLink 端口自动探测 |
| [`mavlink_min.py`](mavlink_min.py) | 零依赖 MAVLink v1/v2 编解码器 |
| [`logparse.py`](logparse.py) | ArduPilot DataFlash `.bin` 日志读取器（可选交叉验证用） |

---

## 使用方法（PyCharm / 终端通用）

### 第一步：录制静置 IMU 数据
将飞控平放在无振动桌面上（USB 供电，不接电机不装桨）：
```bash
cd imu_noise

# 1. 先探测 5 秒确认串口与 HIGHRES_IMU 链路正常
python record_static_imu.py --probe

# 2. 短时冒烟测试（0.1 小时 = 6 分钟，足够测出白噪声密度）
python record_static_imu.py 0.1

# 3. 正式长时录制（默认 8 小时，用于同时测出随机游走）
python record_static_imu.py
```
输出保存在 `imu_noise/data/static_<时间戳>.csv`。

### 第二步：运行 Allan 方差分析
```bash
# 默认自动读取 imu_noise/data/ 下最新的 CSV（每次分析前自动运行合成数据反解自检）
python analyze_static_noise.py

# 或显式指定 CSV / ArduPilot .bin 日志文件
python analyze_static_noise.py data/static_20261017_085103.csv
```
输出保存在 `imu_noise/results/<时间戳>/`：
- `allan_gyro.png` / `allan_accel.png`：双对数 Allan 偏差曲线与拟合窗口
- `noise_results.json`：完整拟合数值、$R^2$、时间基准抖动与判据信息
- `kalibr_imu_chain.yaml`：可直接用于 OpenVINS / Kalibr 的 IMU 噪声配置

---

## 核心参数约定（避免 $\sqrt{f_s}$ 倍常见错误）

Kalibr 与 OpenVINS 中的 `*_noise_density` 定义为**连续时间单边功率谱密度平方根**（单位 $\text{[unit]}/\sqrt{\text{Hz}}$），离散采样标准差与之满足：
$$\sigma_{\text{discrete}} = \text{noise\_density} \cdot \sqrt{f_s}$$
对簇长 $m = \tau \cdot f_s$ 求重叠式 Allan 偏差：
$$\text{AD}(\tau) = \frac{\sigma_{\text{discrete}}}{\sqrt{\tau \cdot f_s}} = \frac{\text{noise\_density}}{\sqrt{\tau}}$$
因此：
- **白噪声区（斜率 $-1/2$）**：$\text{noise\_density} = \text{AD}(\tau)\sqrt{\tau} = \text{AD}(1\,\text{s})$（**不能**再乘 $\sqrt{f_s}$；`imu_utils` 输出的 `gyr_n` 乘了 $\sqrt{f_s}$，不可直接填入 Kalibr/OpenVINS）。
- **零偏随机游走区（斜率 $+1/2$）**：$\text{random\_walk} = \text{AD}(\tau)\sqrt{3/\tau}$。
