"""从 ROS bag 中提取时间戳最接近的「图像 + 点云」对。

子模块:
    bag_index   : rosbag 索引封装（按话题列出时间戳 / 按索引精确读取单条消息）
    lidar       : livox_ros_driver/CustomMsg -> numpy 点云
    camera      : sensor_msgs/CompressedImage -> 解码后的图像 / 写 jpg
    pcd         : 写 PCD v0.7（binary / ascii）
    pairing     : 图像与点云的时间戳配对

只依赖 ROS 自带模块（rosbag、genpy、sensor_msgs）与 numpy、opencv：
不需要 catkin 编译，也不需要本机安装 livox_ros_driver
（bag 中内嵌的消息定义会被 rosbag 动态生成反序列化类）。
"""

__version__ = "0.1.0"
