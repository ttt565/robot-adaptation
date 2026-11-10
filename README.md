# 移动机器人在线动力学适应

这个项目从一维速度模型开始，逐步验证差速机器人如何利用**预测误差和近期历史**适应运行中的动力学变化。后续实验使用 MuJoCo 的轮轴扭矩、接触和摩擦，比较在线窗口估计、车载相机线索和学习得到的潜在状态。每个阶段都可单独运行。

## 目录

| 阶段 | 主要内容 |
|---|---|
| [基础教学脚本](simulate_robot.py) | 从直线速度仿真开始，逐步加入 [参数更新](simulate_adaptive.py)、[噪声对照](compare_noisy_identification.py)和[自适应窗口](compare_adaptive_window.py) |
| [二维闭环对照](simulation_ready/README.md) | 数值差速模型，比较固定反馈、PI、固定窗口和自适应窗口 |
| [MuJoCo 物理仿真](mujoco_ready/README.md) | 自由底盘、轮轴电机、接触和摩擦，带实时三维查看器 |
| [车载相机诊断](mujoco_vision_ready/README.md) | 模拟纹理识别与运动异常线索，区分外观变化、隐蔽低摩擦和电机故障 |
| [视觉参与限速](mujoco_vision_control_ready/README.md) | 对照不主动限速、仅运动线索、视觉加运动线索，保留减速的任务代价 |
| [同日志预测](mujoco_prediction_ready/README.md) | 四种预测器复用同一份 MuJoCo 数据，比较 0.02 秒与 0.5 秒预测 |
| [CNN-GRU 学习预测](mujoco_learned_ready/README.md) | 采集 RGB 和状态、离线训练、独立测试、分布外检查及在线潜在状态演示 |

## Windows 启动

在克隆或解压得到的仓库根目录打开 **CMD**，使用 Python 3.12 创建虚拟环境：

```cmd
python --version
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r mujoco_learned_ready\requirements.txt
```

这个依赖清单包含 CPU 版 PyTorch。若暂时只运行学习版之前的实验，可安装 `mujoco_prediction_ready\requirements.txt`，省去 PyTorch。各子目录的 `.cmd` 启动器优先选择仓库根目录的 `.venv`。

已有权重可直接演示，无需下载或采集完整训练数据：

```cmd
python mujoco_learned_ready\demo.py --output-dir runs\online_demo
```

此命令打开 MuJoCo 三维场景和 Matplotlib 预测窗口。也可双击学习版的 `run_demo.cmd`。无窗口快速验证：

```cmd
python mujoco_learned_ready\demo.py --headless --max-steps 5 --output-dir runs\smoke
```

无窗口仍需离屏 OpenGL 渲染车载图像。演示使用已保存的场景清单重建物理运行，从新观测更新隐状态；网络权重保持不变。默认完整演示为 120 步，短检查不用于对比正式误差指标。

## 重新采集与训练

从仓库根目录执行：

```cmd
python mujoco_learned_ready\pipeline.py all --data-dir runs\learned_data --output-dir runs\trained
```

这会生成完整的 84 段 MuJoCo 数据、训练三个神经网络并运行测试。也可按同样的目录参数分别运行 `collect`、`train`、`evaluate`。如要演示新训练的权重：

```cmd
python mujoco_learned_ready\demo.py --data-dir runs\learned_data --model-dir runs\trained --output-dir runs\new_demo
```

仓库保留学习版权重、训练和评分记录及 `demo_assets/manifest.json`。视觉阶段保留各自完整的 `prepared_model` 校验缓存，样例目录保留结果图、摘要和紧凑日志。完整训练数据和较早实验的逐帧相机图像作为独立附件提供，从仓库根目录解压即可还原原目录布局。依赖版本、硬件和渲染器不同，重新采集与训练不保证字节级一致。[下载原始数据与相机样例](https://github.com/ttt565/robot-adaptation/releases/tag/v0.1.0)

各阶段的 `ready_check.json` 等报告记录原实验检查。报告里的来源哈希用于追踪历史版本；发布副本的说明、演示入口和路径可能已经调整，不能把旧哈希当作当前工作树的校验清单。

## 已观察到的结果

- **预测误差可帮助适应。** 电机故障的同日志对照中，两个种子平均的 0.5 秒角速度 RMSE 从固定模型的 0.20823 降至历史自适应的 0.05082 弧度/秒。误差达标确认延迟为 2.87 秒，包含评分窗口、保持条件及未来目标可用的等待时间。视觉缩窗的额外收益不稳定。
- **预测异常不等于应该减速。** 在可见低摩擦下，限速降低了轮周速度与车体速度差，却增加了原时间参考下的位置误差并损失进度。电机故障中，基线位置 RMSE 为 0.01872 米，持续限速的两组为 1.75833 米。轮体速度差只是滑移线索，并非标准滑移率。
- **学习版主要依赖运动历史。** 独立测试的 0.5 秒速度 RMSE，固定模型为 0.03065 米/秒，历史 GRU 为 0.01181，融合模型为 0.01139。替换融合模型的图像后结果几乎不变，重置历史状态后明显变差；分布外测试中融合的速度误差又高于历史模型，当前不足以把增益归因于视觉。

这些结果来自有限的模拟场景和训练种子。学习版相机是理想的接触区域下视相机，不能提前看到未来接触位置；外观变化不能唯一确定摩擦，潜在状态也不是已标定的物理参数。当前评估没有证明真实道路材料识别、实物适应或闭环控制普遍改善。
