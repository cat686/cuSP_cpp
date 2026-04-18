# cuSP_cpp

`cuSP_cpp` 是从 `cusignal` 迁移出来的 CPU 版 C++ 实现，当前主要包含两个算法示例：

- `task1.cpp`：雷达 LFM 回波处理与检测
- `task2.cpp`：复合信号分析与多域特征提取

## Task1 逻辑概述

`task1` 对应一条典型的脉冲雷达处理链，核心流程如下：

1. 生成发射 LFM 脉冲。
2. 根据目标距离、速度和幅度模拟多目标回波，并叠加噪声。
3. 对每个脉冲做匹配滤波，完成脉冲压缩。
4. 在慢时间维做 Doppler 处理，形成距离-多普勒图。
5. 对距离-多普勒功率图执行 2D CA-CFAR 检测。
6. 计算模糊函数，用于观察波形分辨与耦合特性。
7. 根据检测峰值估计目标距离和速度。
8. 导出 CSV / 二进制结果，供后续可视化。

简而言之，`task1` 解决的是“发射 LFM 后，如何从回波中恢复目标的距离和速度”。

## Task2 逻辑概述

`task2` 面向复合信号建模和多域分析，核心流程如下：

1. 构造包含 LFM、Gaussian pulse、回波、方波干扰和噪声的接收信号。
2. 对原始信号做预处理，包括去趋势、带通滤波和平滑。
3. 从平滑结果中估计主活动区间。
4. 分别提取四类特征：
   - 调制域：瞬时频率、调制斜率
   - 时域：自相关、周期结构
   - 频域：功率谱主峰、带宽信息
   - 时频域：CWT ridge、中心频率与带宽
5. 在各域中做峰值检测。
6. 使用 Kalman 平滑稳定估计结果。
7. 输出主活动区、调制类型、频率范围、带宽和斜率等信息。
8. 导出可视化所需数据。

简而言之，`task2` 解决的是“在含干扰和噪声的复合信号中，提取稳定的时间、频率和调制特征”。

## 目录说明

- `common/`：基础数据结构与通用数学工具
- `radartools/`：`task1` 相关核心处理
- `waveforms/`、`filtering/`、`demod/`、`spectral_analysis/`、`wavelets/` 等：`task2` 使用的功能模块
- `cuSP_task1_ops.hpp`、`cuSP_task2_ops.hpp`：任务入口和公共模块之间的适配层

## 构建

```bash
cmake -S cuSP_cpp -B cuSP_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuSP_cpp/build --parallel
```

生成的可执行文件位于：

```text
cuSP_cpp/build/bin/task1_cpp
cuSP_cpp/build/bin/task2_cpp
```
