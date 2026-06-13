# 🏔️ 滑坡易发性分析工具
**Landslide Susceptibility Analysis Tool based on Information Value Model (IVM)**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Web App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## 🌐 在线使用

**无需安装任何软件，打开即用：**

👉 [https://landslide-ivm-tool-brutchr7glrzjy6pr5gqlq.streamlit.app](https://landslide-ivm-tool-brutchr7glrzjy6pr5gqlq.streamlit.app)

---

## 是什么？

这是一个面向地质灾害研究人员的**在线滑坡易发性自动评价工具**。

传统的 IVM 信息量模型评价流程繁琐，需要在 GIS 软件和 Python 脚本之间来回切换，数据处理步骤多、容易出错。这个工具把完整流程自动化，上传数据、点击运行、下载结果，三步完成一次评价。

---

## 能做什么？

### 📊 IVM 信息量评价
- 批量读取环境因子栅格（TIF），自动清洗无效值和 NoData
- 根据因子类型智能选择分级方式（坡度按10度分级、坡向9方向重分类、其他因子 Jenks 自然断点法）
- 计算各因子等级的信息量值 IV，生成综合易发性图
- 使用自然断点法将综合结果分为5级（极低→极高易发区）
- 自动输出统计表、各因子 IV 栅格、综合易发性图、5级分类图

### 📍 负样本生成
- 基于 IVM 分级结果，在低易发区自动生成与正样本1:1的负样本点
- 支持三种采样策略：**极低+低易发区**（推荐）、**仅极低易发区**、**全区随机**
- 正样本缓冲区避让，有效降低正负样本空间混淆
- 支持独立使用：只需上传一个 TIF 和正样本 SHP，即可生成随机负样本

### 🔗 两模块无缝衔接
IVM 计算完成后，分级图自动传递给负样本模块，无需重复上传，一键生成负样本。

---

## 适合谁用？

- 做**滑坡易发性评价**的研究生和科研人员
- 需要快速生成**机器学习训练样本**的 GIS 用户
- 想复现 IVM 模型但不熟悉 Python 的地质工程从业者

---

## 输入数据要求

| 数据 | 格式 | 说明 |
|---|---|---|
| 环境因子栅格 | `.tif` | 坐标系一致，分辨率建议一致 |
| 滑坡正样本 | `.zip`（含 .shp/.dbf/.shx/.prj） | 点数据，不支持面数据直接输入 |
| IVM 分级图 | `.tif` | 负样本模块单独使用时可手动上传 |

---

## 输出结果

| 文件 | 说明 |
|---|---|
| `IV_<因子名>.tif` | 每个因子的 IV 栅格 |
| `Final_Susceptibility_Map.tif` | 综合易发性图（连续值） |
| `Classified_Susceptibility_Map_5Class.tif` | 5级分类图（1=极低，5=极高） |
| `IV_Statistics.csv` | 各因子等级统计表 |
| `negative_samples.zip` | 负样本 Shapefile |

---

## 技术栈

| 组件 | 说明 |
|---|---|
| Streamlit | Web 界面框架 |
| Rasterio | 栅格数据读写 |
| GeoPandas | 矢量数据处理 |
| MapClassify | Jenks 自然断点分级 |
| Shapely | 空间几何运算 |
| SciPy | 二值腐蚀等空间分析 |
| NumPy / Pandas | 数组与表格处理 |

---

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/jgua760/landslide-ivm-tool.git
cd landslide-ivm-tool

# 安装依赖
pip install -r requirements.txt

# 启动
streamlit run app_web.py
```

---

## 引用 / 致谢

如果这个工具对你的研究有帮助，欢迎 Star ⭐ 或在论文中注明工具来源。

---

## 联系

如有问题或建议，欢迎提交 [Issue](https://github.com/jgua760/landslide-ivm-tool/issues) 或直接联系作者。

---

<p align="center">
  Made with ❤️ for geohazard researchers
</p>
