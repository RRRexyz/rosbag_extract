# rosbag_extract — 从 rosbag 提取时间戳最接近的「图像 + 点云」对

从 `campus1.bag`（8.0 GB，Livox 点云 + 压缩图像）中取出一帧点云和它在时间上最接近的一帧图像，
分别保存为 **`.pcd`** 与 **`.jpg`**，并把每一对的时间差写进 `pairs.csv`。

- **不扫描整个 bag**：只读 bag 尾部索引定位目标消息，取一帧约 2 秒（含 Python/ROS 启动）；
- **时间上尽可能接近**：先用索引记录时间初选，再用消息头 `header.stamp` 复核，必要时在 ±2 帧内比较；
- **不二次压缩**：bag 里的图像本来就是 JPEG，保存时直接写原始字节，画质与原数据完全一致；
- **开箱即用**：只依赖 ROS Noetic 自带的 `rosbag`/`genpy` 与 `numpy`/`opencv`，**不需要 catkin 编译**，
  也不需要安装 `livox_ros_driver` —— bag 内嵌的消息定义会被 rosbag 用 `genpy.dynamic` 动态生成解析类。

## 1. 环境

| 依赖 | 说明 |
| --- | --- |
| Ubuntu 20.04 + ROS Noetic | 提供 `rosbag`、`genpy`、`sensor_msgs`（`ros-noetic-ros-base` 即可） |
| python3.8 | ROS Noetic 自带 |
| numpy | `sudo apt install python3-numpy` |
| OpenCV | `sudo apt install python3-opencv`（`python3 -c "import cv2"` 能过即可） |

自检：

```bash
python3 -c "import rosbag, genpy, sensor_msgs.msg, cv2, numpy; print('ok')"
```

## 2. 目录结构

```
rosbag_extract/
├── extract_frames.py            # 命令行入口
├── rosbag_extract/
│   ├── bag_index.py             # bag 索引：按话题列时间戳 / 按索引精确读取单条消息
│   ├── lidar.py                 # livox_ros_driver/CustomMsg -> numpy 点云
│   ├── camera.py                # CompressedImage 解码 / 写 jpg（原始字节直存）
│   ├── pcd.py                   # PCD v0.7 写出（binary / ascii）
│   └── pairing.py               # 图像与点云的时间戳配对
├── tests/test_offline.py        # 不依赖 bag 的单元测试
├── pyrightconfig.json           # Pylance / pyright 配置（见第 6 节）
├── requirements.txt
└── output/                      # ← 提取结果（见第 5 节）
```

## 3. 快速开始

```bash
cd ~/project/rosbag_extract

# 默认 bag，取第 0 帧（输出到 ./output）
python3 extract_frames.py

# 看 bag 里有哪些话题
python3 extract_frames.py --list-topics

# 指定帧号 / 绝对时间戳
python3 extract_frames.py --frame-index 1165
python3 extract_frames.py --timestamp 1760225300.5

# 全程均匀取 5 帧 / 全部 3496 帧
python3 extract_frames.py --num-frames 5
python3 extract_frames.py --all

# 只看配对结果不写文件
python3 extract_frames.py --dry-run --num-frames 5
```

运行输出示例：

```
点云/图像  : /livox/lidar (3496 帧)  <->  /camera/image/compressed (3496 帧)
有效点云帧 : 3496（已跳过记录时间无效帧 4 帧）
图像周期   : 0.0999 s   本次提取 3 帧

frame   lidar_stamp         image_stamp             dt[ms]   points  files
--------------------------------------------------------------------------
frame_000000 1760225178.832201   1760225178.855851      +23.650    22037  frame_000000.jpg / frame_000000.pcd
frame_001748 1760225353.874444   1760225353.894740      +20.295    22170  frame_001748.jpg / frame_001748.pcd
frame_003495 1760225528.811862   1760225528.831014      +19.153    22437  frame_003495.jpg / frame_003495.pcd

完成: 3 帧, 平均 |dt| = 21.03 ms, 最大 |dt| = 23.65 ms
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--bag` | `/home/rex/data/shitong/campus/campus1.bag` | bag 路径 |
| `--output-dir` | `<项目>/output` | 输出根目录 |
| `--lidar-topic` | `/livox/lidar` | 点云话题（`livox_ros_driver/CustomMsg`） |
| `--image-topic` | `/camera/image/compressed` | 图像话题（`sensor_msgs/CompressedImage`） |
| `--frame-index` | — | 点云帧序号，可逗号分隔：`0,120,240` |
| `--timestamp` | — | 绝对时间戳（秒），取最近帧，可逗号分隔 |
| `--num-frames` | `1` | 在全程均匀取 N 帧 |
| `--all` | 关 | 提取全部有效帧 |
| `--pcd-format` | `binary` | `binary` 或 `ascii` |
| `--jpeg-quality` | — | 重编码 JPEG 的质量 1–100；不给则直存原始 JPEG 字节 |
| `--keep-zero-points` | 关 | 保留 Livox 的 `(0,0,0)` 无回波点 |
| `--skip-existing` | 关 | jpg 与 pcd 都已存在则跳过（断点续提） |
| `--search-radius` | `2` | 复核时向两侧扩展的候选图像数 |
| `--dry-run` | 关 | 只计算并打印配对结果，不写文件 |
| `--list-topics` | 关 | 列出 bag 话题后退出 |
| `--quiet` | 关 | 只输出汇总 |

`--frame-index` 的序号是「有效点云帧」的序号：bag 开头 4 条 `/livox/lidar` 是空帧
（`point_num = 0`、时间戳 0），已被自动跳过，所以 `--frame-index 0` 对应 bag 里第 5 条点云消息。

## 4. 时间戳是怎么对齐的

本 bag 里图像与点云都是约 10 Hz（图像周期 0.0999 s、点云周期 0.1005 s），相位并不固定，
所以每帧都要单独找最近的图像。`pairing.match_image()` 分两步：

1. **初选（读 1 条消息）**：用 bag 索引里的记录时间做二分，找到记录时间最接近点云 `header.stamp`
   的图像帧，只读这一条。记录时间与传感器时间戳通常只差零点几毫秒，绝大多数帧这一步就是最优解。
2. **复核（必要时）**：读出图像的 `header.stamp`，若与点云时间戳之差已经小于半个图像周期，
   就认为已达最优、立即结束；否则在 ±`--search-radius` 帧范围内继续读候选，取绝对差最小者。
   这样即使录制时钟不准（例如 lidar 前几帧记录时间为 0），也能找回正确的帧。

时间差 = `image.header.stamp - lidar.header.stamp`，正值表示图像比点云晚。本 bag 实测：

| 抽样 | 平均 \|dt\| | 最大 \|dt\| |
| --- | --- | --- |
| 3 帧（首/中/尾） | 21.0 ms | 23.7 ms |
| 30 帧均匀分布 | 17.9 ms | 37.0 ms |
| 300 帧均匀分布 | 18.2 ms | 42.2 ms |

由于两个传感器都是 10 Hz 且相位错开，最近帧的理论上界是半个周期（约 50 ms），
上面所有帧都落在界内，即**每一帧都已取到时间上最近的图像**（可用 `--dry-run` 自行核对）。

## 5. 输出

```
output/
├── images/frame_000000.jpg     # 图像（原始 JPEG 字节直存，2592x1944）
├── pcd/frame_000000.pcd        # 点云（FIELDS x y z intensity，binary）
├── meta/frame_000000.json      # 该帧全部时间戳/点数等元信息
└── pairs.csv                   # 所有已提取帧的配对表
```

文件名 `frame_XXXXXX` 中的数字是点云帧序号，jpg / pcd / json 三者的主干名相同，便于按对使用。
仓库里这份 `output/` 已是用 `--num-frames 3`（首/中/尾）跑出的 3 对示例结果；
`output/preview/` 下另有一张点云俯视图，只是人工核对用的渲染，不属于提取产物。

`pairs.csv` 列：

| 列 | 含义 |
| --- | --- |
| `frame_index` | 点云帧序号 |
| `lidar_record_time` | 点云的 bag 记录时间 |
| `lidar_stamp` | 点云 `header.stamp` |
| `image_record_time` | 图像的 bag 记录时间 |
| `image_stamp` | 图像 `header.stamp` |
| `delta_ms` | `image_stamp - lidar_stamp`（毫秒） |
| `num_points` | 该帧点云点数（已剔除无效点） |
| `image_file` / `pcd_file` | 相对输出目录的路径 |

`meta/*.json` 里还记录了 `timebase_ns`（本帧第一个点的纳秒时间）、`offset_time_range_ms`
（帧内点的时间跨度，约 100 ms，说明这是扫描式点云）、`dropped_invalid_points`、图像分辨率等。

PCD 只含 `x y z intensity` 四个 float32 字段，CloudCompare、`pcl_viewer`、Open3D 均可直接打开：

```bash
pcl_viewer output/pcd/frame_000000.pcd
python3 -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('output/pcd/frame_000000.pcd')])"
```

## 6. 单元测试与类型检查（VSCode / Pylance）

```bash
python3 -m unittest discover -s tests -v
```

覆盖 PCD 读写往返（binary/ascii/空点云）、时间戳配对（精确命中只读 1 条、记录时间有偏时复核、
边界截断）、CustomMsg 解析（剔除 `(0,0,0)`、时间戳异常退化）等，不依赖 bag 文件。

类型检查用项目自带的 `pyrightconfig.json`（Pylance 打开工程时会自动读取）：

```bash
npm i -g pyright        # 或 python3 -m pip install pyright
pyright                 # 0 errors, 0 warnings
```

配置里有两个关键点：

- `extraPaths` 指向 `/opt/ros/noetic/lib/python3/dist-packages`：Pylance 不会执行 `setup.bash`，
  拿不到 `PYTHONPATH`，不加这一项会报 `Import "rosbag" could not be resolved`；
- `useLibraryCodeForTypes: false`：本机 numpy 是 1.17.4（Python 2 时代代码、没有 `py.typed`），
  Pylance 解析它的源码时会把 `np.asarray` / `np.float32` / `np.isfinite` 等误判成
  「模块 numpy 没有这个属性」。这类报错（30 个左右）全部来自库本身，关掉源码推断即可消除；
  它只影响类型检查，不影响运行。若以后换成带类型标注的 numpy（≥1.20），可以删掉这一项以获得更严格的检查。

改完 `pyrightconfig.json` 后，在 VSCode 里执行 `Ctrl+Shift+P → Developer: Reload Window`
（或 `Python: Restart Language Server`）让 Pylance 重新加载配置。

## 7. 说明与限制

- 只做「取帧 + 配对」，不做去畸变、不下采样、不做点云到图像的投影；
  `../data/shitong/campus/campus.yaml` 中有 `li_extrinsic` / `ci_extrinsic` 与相机内参，可用于后续投影。
- 点云是 Livox 帧坐标系（`frame_id: livox_frame`），未转换到 IMU 或相机坐标系。
- 每帧 24096 个原始点中约 2000 个是 `(0,0,0)` 无回波点，默认剔除；`--keep-zero-points` 可保留。
- `--all` 会读完整 8 GB 点云并解码 3496 张 2592x1944 图像，实测约 0.13 s/帧
  （300 帧用时 40 s），全量约 8 分钟，输出约 7.8 GB。
