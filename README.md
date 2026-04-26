# 论文2复现：多模态语义SLAM + 不确定性感知导航

本目录复现论文 **A Multi-Modal Semantic SLAM and Uncertainty-Aware Navigation Framework for Robust Autonomous Operation in Photovoltaic Cleaning Robots** 的关键技术链路：

- VIO/LIO 概率融合（高斯信息融合）
- 不确定度建模（协方差迹 + 视觉/激光/语义质量项）
- 语义栅格地图（log-odds 递推更新）
- 导航代价函数融合：路径长度 + 边界风险 + 定位不确定度 + 重复清扫惩罚

## 文件说明

- `main.py`：可运行仿真，构造光伏行列场景并执行覆盖式清扫

## 运行方式

```bash
python main.py
```

## 输出指标（与论文指标对应）

- 近似路径长度（Path）
- 清扫覆盖率（Coverage）
- 平均不确定度 `U`
- 轨迹误差代理指标 `sqrt(trace(Sigma))`
- 最小边界安全裕度

## 方法映射

- 论文融合公式 `Sigma_f^-1 = Sigma_1^-1 + Sigma_2^-1` -> `gaussian_fusion`
- 语义概率更新与 log-odds -> `SemanticGridMap.update_cell`
- 不确定性评分 `U_t` -> `MultiModalSLAMDemo.step`
- 不确定性感知路径规划目标 -> `UncertaintyAwarePlanner.plan`
