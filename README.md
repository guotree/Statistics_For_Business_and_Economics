# 商务与经济统计学习笔记

果tree 2022.04.05

本人数学系本科，有数理统计基础，但接触统计实际案例比较少，导致理论和生产脱节，趁着入职前的一段闲暇时间，来过一遍《商务与经济统计》

每章节课后题的python实现，计划两个月左右完成，会涉及到pandas，matplotlib，scikit-learn，seaborn，scipy等包，供学习使用
# 目录
- [第一章  数据与统计资料](#第一章--数据与统计资料)
- [第二章  描述统计学1：表格法和图形法](#第二章--描述统计学1表格法和图形法)
- [第三章  描述统计学2：数值方法](#第三章--描述统计学2数值方法)
- [第四章  概率](#第四章--概率)
- [第五章  离散型概率分布](#第五章--离散型概率分布)
- [第六章  连续型概率分布](#第六章--连续型概率分布)
- [第七章  抽样和抽样分布](#第七章--抽样和抽样分布)
- [第八章  区间估计](#第八章--区间估计)
- [第九章  假设检验](#第九章--假设检验)
- [第十章  两总体均值和比例的推断](#第十章--两总体均值和比例的推断)
- [第十一章  总体方差的统计推断](#第十一章--总体方差的统计推断)
- [第十二章  多个比例的比较、独立性及拟合优度检验](#第十二章--多个比例的比较独立性及拟合优度检验)
- [第十三章  实验设计与方差分析](#第十三章--实验设计与方差分析)
- [第十四章  简单线性回归](#第十四章--简单线性回归)
- [第十五章  多元回归](#第十五章--多元回归)
- [第十六章  回归分析：建立模型](#第十六章--回归分析建立模型)
- [第十七章  时间序列分析及预测](#第十七章--时间序列分析及预测)
- [第十八章  非参数方法](#第十八章--非参数方法)
- [第十九章  质量管理的统计方法](#第十九章--质量管理的统计方法)
- [第二十章  指数](#第二十章--指数)



## 第一章  数据与统计资料



## 第二章  描述统计学1：表格法和图形法

![数据可视化](./image/数据.svg)

[notebook](https://github.com/guotree/Statistics_For_Business_and_Economics/blob/main/notebook/第二章.ipynb)

数据可视化建议

1. 给与图形显示一个清晰简明的标题
2. 使图形显示保持简洁，当能用二维表示时不要用三维表示
3. 每个坐标轴有清楚的标记，并给出测量的单位
4. 如果使用颜色来区分类别，要确保颜色是不同的
5. 如果使用多种颜色或线型，要用图例来表明时，要将图例靠近所表示的数据

数据仪表盘：一个用易于阅读、了解和解释的方式组织和表述用于监控公司或机构业绩的直观显示集合。

### 需要熟悉的Python包

Matplotlib

```python
import matplotlib.pyplot as plt 
plt.pie() #饼图
plt.scatter() #散点图
plt.bar() #条形图
plt.hist() #直方图
plt.plot() #折线图
plt.vline() #线段
```

Seaborn

```python
import seaborn as sns
#没有饼图
sns.scatterplot() #散点图
sns.barplot() #条形图
sns.distplot() #直方图
sns.lineplot() #折线图
```



Plotly

```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

px.scatter() #散点图
px.line() #折线图
go.Scatter() #散点图，折线图

px.bar() #条形图
go.Bar() #条形图

px.pie() #饼图
go.Pie() #饼图

px.histogram() #直方图
go.Histogram() #直方图
px.distplot() #直方图混合
ff.create_distplot() #直方图混合
```



Pandas

```python
import pandas as pd
pd.cut() #数据分箱
pd.qcut() #数据按百分位分箱
pd.crosstab() #交叉表
pd.pivot_table() #数据透视表
```



## 第三章  描述统计学2：数值方法

![数值方法](./image/数值描述.svg)

[notebook](https://github.com/guotree/Statistics_For_Business_and_Economics/blob/main/notebook/%E7%AC%AC%E4%B8%89%E7%AB%A0.ipynb)

需要用到的Python包：numpy，scipy，matplotlib，plotly，seaborn

```python
import numpy as np
from scipy import stats
from matplotlib.pyplot as plt
import plotly.express as px
import plotly.gragh_objects as go
import seaborn as sns

np.mean() #算术平均数
np.average() #加权平均数
stats.gmean() #几何平均数
stats.hmean() #调和平均数
np.median() #中位数
stats.mode() #众数
np.quantile() #分位数

n #极差
stats.iqr() #四分位数间距
np.var() #方差
np.std() #标准差
np.std()/np.mean() #标准差系数

stats.skew() #偏度
stats.kurtosis() #峰度
stats.zscore() #zscore

np.cov() #协方差
np.correlate() #相关系数
stats.pearsonr() #皮尔逊相关系数
stats.spearmanr() #斯皮尔曼相关系数
stats.kendalltau() #肯德尔相关系数

#matplotlib
plt.boxplot()
plt.violinplot()
#seaborn
sns.boxplot()
sns.violinplot()
#plotly
px.boxplot()
go.Box()
px.violin()
go.Violin()
```



## 第四章  概率

![](./image/概率.svg)

## 第五章 离散型概率分布

```python
stats.rv_discrete()
```
![](./image/离散型概率分布.svg)
[notebook](https://github.com/guotree/Statistics_For_Business_and_Economics/blob/main/notebook/%E7%AC%AC%E4%BA%94%E7%AB%A0.ipynb)


## 第六章  连续型概率分布

```python
stats.rv_continuous()
```
![](./image/连续型概率分布.svg)

## 第七章  抽样和抽样分布

![](./image/抽样与抽样分布.svg)

## 第八章  区间估计

![](./image/区间估计.svg)

## 第九章  假设检验
## 第十章  两总体均值和比例的推断

![](./image/两总体均值和比例的推断.svg)

## 第十一章  总体方差的统计推断

![](./image/总体方差的统计推断.svg)

## 第十二章  多个比例的比较、独立性及拟合优度检验
## 第十三章  实验设计与方差分析
## 第十四章  简单线性回归
## 第十五章  多元回归
## 第十六章  回归分析：建立模型
## 第十七章  时间序列分析及预测
## 第十八章  非参数方法
## 第十九章  质量管理的统计方法
## 第二十章  指数
