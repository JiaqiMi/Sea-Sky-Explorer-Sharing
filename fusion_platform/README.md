# Geo Fusion IDE

一个带有**菜单栏、工具栏、参数面板、可视化画布与按钮**的桌面式 IDE，
用于加载三类物理场（地形/重力/地磁）、并执行：

- **地图级融合**：统一网格并线性加权生成融合图
- **特征级融合**：提取 Raw/Gradient/Roughness 构成特征张量，PCA 融合展示
- **决策级融合**：基于点击的“测量位置”生成每通道似然图并乘积为后验热力图

> 基于 Tkinter + Matplotlib，轻依赖（仅 `numpy`, `matplotlib`；可选 `rasterio`）。

## 运行

```bash
python ide_app.py
```

## 使用指南
- **File → Open ...** 分别加载 Terrain / Gravity / Magnetic（支持 `.tif/.tiff`（需 rasterio）, `.npy`, `.txt`）
- 工具栏按钮或 **Tools** 菜单：运行三种融合模式
- 左侧 **Notebook** 面板设置各模式参数
- **Decision** 模式可勾选工具栏 *Pick measurement*，然后在画布上**点击**选择测量位置，再点击 *Decision Fusion* 刷新后验
- **File → Export PNG** 导出当前画布图像

## 说明
- 所有融合算法来自 `fusion_platform/`，可按需替换或扩展：
  - `map_fusion.py`：对齐/标准化/线性权重合成
  - `feature_fusion.py`：Sobel 梯度、局部粗糙度、PCA
  - `decision_fusion.py`：高斯似然与乘积后验
- 如需地理坐标显示/投影，请安装 `rasterio` 并在 `io_utils.py` 中扩展显示坐标轴。

## 已知限制
- 目前将栅格视作影像处理（像素坐标），不含地理参考显示；融合时已进行单通道标准化以减少量纲差异。
- GUI 为单文档视图，后续可扩展为多窗口/多图层叠加、图例、测距等。

