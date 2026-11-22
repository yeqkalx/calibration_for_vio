# capture/ —— 外参标定的采集链路


在 PyCharm 里打开 `calibration_for_vio/extrinsic` 作为项目（解释器示例：
`/home/kxy/miniforge3/envs/vins/bin/python`），然后：

```
                      ┌─────────────────────────────┐
   第一次先点这个  →  │ selftest_capture.py         │  离线自检，不用插任何硬件
                      └─────────────────────────────┘
                      ┌─────────────────────────────┐
   采数据之前点这个 → │ framing_check.py            │  ★ 10 秒取景/曝光体检，带预览窗
                      └─────────────────────────────┘
                      ┌─────────────────────────────┐
   画面调对后点这个 → │ record_session.py           │  ★ 采集 → 生成 CSV → 自动标定
                      └─────────────────────────────┘
                      ┌─────────────────────────────┐
   只想验证 IMU   →  │ imu_recorder.py             │  录 10 秒 IMU，给体检结论，不采图
                      └─────────────────────────────┘
                      ┌─────────────────────────────┐
   已有图想重转   →  │ make_images_csv.py          │  manifest → images.csv
                      └─────────────────────────────┘
```

采集参数在 [`capture_config.py`](capture_config.py)，标定板尺寸在对应的 YAML。

> ⚠️ **为什么要有 `framing_check.py`**：采集时的预览窗被关掉了 —— 那个 Tk/PIL 预览每帧要
> ~306 ms，开着它帧率只有 1.8 fps。代价是**采集变成盲拍**。2026-10-15 就因此白采了两轮
> （第一轮 2.6 fps 的 25 张、第二轮 155 张**全部零检出**，都是跑完标定才知道）。
> `framing_check.py` 把预览加回来，而且只录 10 秒就当场给判读。

---

## PyCharm 一键外参标定

1. 在 `capture_config.py` 选 `TARGET_TYPE`：
   - `"checkerboard"`：改 [`configs/checkerboard.yaml`](../configs/checkerboard.yaml) 的 `cols`、`rows`（横纵内角点数）和 `square_size`（实测格边长，米）。
   - `"aprilgrid"`：改 [`configs/aprilgrid.yaml`](../configs/aprilgrid.yaml) 的 `tagCols`、`tagRows`、`tagSize`（单个 tag 黑框外沿边长，米）、`tagSpacing`（白隙 ÷ tagSize）。`first_id` 和 `tag_rotation_deg` 需与板子的 ID 排布和印版一致。
2. 用一个新的 `SESSION_NAME`，保留 `CAPTURE_PURPOSE = "calibration"`、`RUN_CALIBRATION_AFTER = True`。两种板都运行同一个 `record_session.py`，不用修改 `extrinsics_config.py` 的 `PATHS`。
3. 看到控制台提示「计时从现在开始」，让标定板保持可见，转动整套设备（绕三个轴，并做一些平移）。采集完成后不再占用设备，程序自动用本次图像、IMU 和对应的标定板 YAML 求解。先看 `results/<SESSION_NAME>/report.json`；通过后再用 `camchain-kalibr.yaml` 或 `camchain-openvins.yaml`。

一次只用一种板。换板或重录时换一个 `SESSION_NAME`。如果只想录不求解，设 `RUN_CALIBRATION_AFTER = False`；`CAPTURE_PURPOSE = "vio"` 下也不会自动标定。

默认采集 1920×1080 MJPG、请求 30 fps；可在 `capture_config.py` 的 `RESOLUTION` 切到 `720p` / `640x480`，分别对应 `configs/gemini_pro_rgb_720p.yaml` / `configs/gemini_pro_rgb_640x480.yaml`。切换分辨率必须同步换内参，工具会比对并报错，不会自动缩放。

先对准板子并均匀补光，再开始采集。标定按当前内参的 `fx` 和 `CALIBRATION_AUTO_EXPOSE_EXPECTED_RATE_DPS`（默认 12°/s）计算曝光预算，灰度达到最低门槛但没到目标时仍会继续做曝光验证和标定板检测。`framing_check.py` 使用同一测光配置，可先跑它排查亮度和检出。

---

## `record_session.py` 具体做什么

```
[1/4] 找相机        扫 /dev/video*，按 USB 身份 2bc5:0511 认人（拒绝误用笔记本摄像头）
[2/4] 找飞控串口    在 ttyACM*/ttyUSB* 里逐个"只监听不发送"，挑出真在讲 MAVLink 的那个
                    （7-Nano 会同时枚举出 ttyACM0 和 ttyACM1，只有一个在跑 MAVLink）
[3/4] 查配置        内参分辨率必须 == 采集分辨率；标定板行列数会被打印出来让你核对
[4/4] 查目录        会话目录非空就拒绝，防止把两轮数据混在一起
────────────────────────────────────────────────────────────
起 IMU 录制线程 → 等 2 秒 → 拉起相机子进程 → 等它跑完 → 收 IMU → 写两份 CSV
────────────────────────────────────────────────────────────
打印体检报告 → 标定模式下自动求解
```

相机采完会在控制台看到它的实时输出（保存进度、实测帧率）。

---

## 时间戳是怎么对齐的（这部分决定标定能不能做）

两路都是**宿主机的 `CLOCK_REALTIME`**（`time.time_ns()`）：

- 相机：`capture_calibration.py` 把 `host_receive_time_ns` 写进文件名和 manifest
- IMU：`imu_recorder.py` 在每次 `read` 到串口数据时取 `time.time_ns()`

**两个进程调的是同一个系统时钟**，时间戳都是绝对时刻，所以不需要任何额外同步手段，
两个进程谁先启动、差几秒，都不影响。

真正剩下的偏差是**系统性**的：相机那一侧的时间戳是"宿主机收到整帧"的时刻
（曝光 + 读出 + USB 传输 + 驱动缓冲），IMU 那一侧是"宿主机读到串口字节"的时刻
（只有 USB 传输）。两者差几十毫秒，正落在标定程序 `max_offset_s = 0.1 s`
的搜索窗内，会被当作 `timeshift_cam_imu` 估出来。**这正是要估的那个量。**

> ⚠️ 有一个坑这里已经替你避开了：**绝对不要拿飞控的 `time_usec`（开机微秒）当时间戳**。
> 它和宿主墙钟差着"开机时长 + 1970 年"这么大的量级，标定会直接报"时间重叠不足"。

但串口是批量读的：一次 `read` 可能拿到好几帧。若给同批帧打同一个时间戳，
200 Hz 下批内就有 5~15 ms 的人为抖动，而标定要估的时移精度是 1 ms 量级。
所以记录器用飞控自己的 `time_usec` 算出**批内相对间隔**，锚定到本次 read 的宿主时间上。
剩下那点固定 USB 延迟会被标定的 offset 搜索吸收。

---

## 采完之后必须看的三件事

`record_session.py` 会把前两件直接打出来：

| 看什么 | 合格线 | 不合格怎么办 |
|---|---|---|
| IMU 实测速率 | ≥ 40 Hz；50 Hz 能用但平移精度会从 ~3 mm 掉到 ~5 mm（实测） | 提高 `IMU_REQUEST_HZ`，或把飞控 `SCHED_LOOP_RATE` 调高 |
| IMU 最大空洞 | 有 >50 ms 的就必须用 `configs/probe.yaml`（`max_imu_gap_s=0.15`），默认 0.05 会整段拒绝 | 同上；换 USB 口/线 |
| 图像帧间隔 | 有 >350 ms 的就要挑一段连续可见的窗口 | 用 `configs/probe.yaml`（`camera_rate_hz=30`） |

第三件要等标定跑完，看 `results/.../extraction.json`：里面逐帧有 `detected` 标记。

- `true` 占比 **> 80%** → 链路通了
- **< 50%** → 板子/距离/光照有问题，先修再往下
- **= 0%** → 标定板的 `cols`/`rows` 或分辨率不对

---

## 常见故障

| 现象 | 原因 |
|---|---|
| **帧率上不去（十几~二十几张 / 10 秒）** | 保存循环里有两处「给内参标定用的重活」。**两处都已在 2026-10-15 修掉**，见下面「帧率是怎么修上来的」。你要是从别处拿了新版本覆盖了 `capture_config.py` / `vendor/capture_calibration.py`，先回来看这一节 |
| 标定报 `IMU: 至少2个严格递增时间戳` | 串口读得断续，批内插值把时间戳插得倒退了。**已在 `imu_recorder.py` 里强制递增**（2026-10-15）。如果还报，看 `record_session.py` 打印的「因乱序被强制递增」占比：占比高说明链路有问题，换 USB 口/线 |
| `没找到 USB 身份为 2bc5:0511 的相机` | 相机没插 / 线是充电线 / 被 Cheese 等程序占着 |
| `没找到在讲 MAVLink 的串口` | 飞控没通电，或串口是 ttyACM1 而不是 ttyACM0。在 `capture_config.py` 里把 `IMU_PORT` 写死试 |
| `内参写的分辨率 ... 与本次采集的 ... 不一致` | `capture_config.py` 的 `RESOLUTION` 和内参文件不匹配。工具**不会**自动缩放内参 |
| `会话目录里已经有东西了` | 改 `SESSION_NAME` 成 `session02`，或把 `REUSE_EXISTING_SESSION` 设成 `True` |
| 标定报 `有效标定板帧只有 0` | 标定板 `cols`/`rows` 和你实际摆的板子不一致（最容易踩：临时换屏幕板时忘了改配置） |
| 标定报 `IMU 数据空洞过大` | 用 `configs/probe.yaml` |
| 画面发白/过曝 | 把 `capture_config.py` 的 `EXPOSURE` 从 100 往下调（如 40） |

### 帧率是怎么修上来的（两次真机故障，都记在这）

第一次：9.15 秒 17 张（1.8 fps）
第二次：9.23 秒 25 张（2.6 fps）—— 相机侧 `v4l2_sequence` 已经 254，即相机在正常出 ~27.5 fps

| 元凶 | 每帧代价 | 修法 |
|---|---:|---|
| **Tk/PIL 预览窗**（这个 conda 的 OpenCV 没有 GUI，只能用 `ImageTk.PhotoImage`） | ~306 ms，**且与分辨率无关**（预览图一律缩到 960 宽） | `capture_config.py`：`SHOW_PREVIEW = False` |
| **保存时逐帧重新做棋盘检测**（内参标定时代的逻辑） | 1080p 上 180~330 ms | `vendor/capture_calibration.py` 第 616 行加 `and not args.no_preview` |

修完之后的预算（1080p，`--no-preview`）：

```
读帧+MJPG解码 ~10 + 灰度 0.2 + 检测 180/15≈12（0.5 秒才跑一次）+ PNG 27 + 写盘 4 + 打印 1  ≈ 55 ms  →  15~20 fps
```

**改过 vendor 的代码要记在 `vendor/PROVENANCE.md` 里** —— 我已经记了。

> ⚠️ `capture/` 靠"目录里已经有 PNG 就拒绝复用"来防止两轮数据混在一起
> —— 这是防止两轮数据混用的保护。**换轮次必须换 `SESSION_NAME`**，不必再改 `extrinsics_config.py` 的 `PATHS`。

### 关于帧率的经验值

保存循环的预算是这么算的（720p）：

```
读帧 2 ms + 灰度转换 2 ms + 棋盘检测 87ms/每0.5秒 + PNG 编码 11 ms + 写盘 3 ms ≈ 26 ms  →  ~30 fps
```

1080p 是 `2 + 5 + 169/8 + 18 + 3 ≈ 49 ms → ~12 fps`（检测每 0.5 秒跑一次，
但如果循环本身慢到超过 0.5 秒，检测就会退化成每帧都跑，自我放大 —— 第一次
1.8 fps 就是这么来的）。

---

## `vendored/` 里是什么

三个从你已有脚本**原样复制**的文件（相机采集、MAVLink 编解码），
复制是为了让这个 PyCharm 工程自包含。来源与哈希见
[vendor/PROVENANCE.md](vendor/PROVENANCE.md)。**不要在那里改任何东西。**

---

## 现在就能做的（相机还没插）

```bash
# 在 PyCharm 里直接点 selftest_capture.py 即可，等价于：
python capture/selftest_capture.py
```

它用**真实串口抓包**验证 MAVLink 解码、用**伪终端造一个假飞控**走完
记录器的完整路径（open/termios/select/read/解析/去重/插值/落盘），
再用标定程序自己的 loader 把生成的 CSV 读回来验证格式。**全程不需要硬件。**
