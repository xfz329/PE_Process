# Python 下的统计分析——coding 实现SPSS的自动化分析任务
* collected and edited by 江锋 
* 2022.03.22
* SPSS分析时，需要人工设置设置数据，当有大量数据需要操作时，效率不够高，其次，若数据更新后，更新相关统计分析的结果时的操作更是繁杂。
* 偷懒的惰性促使我们需要找到更有效率的工具。
* 搜索了下Python下的统计分析工具，在不少package里面都发现了相应的工具，这里简单汇总并简单测试下。

## 一、Python工具包对比
### 1. pandas及numpy 
* 以下库的最底层依赖包，具有部分统计功能，这里不作介绍。
* 参考[NumPy, SciPy, and Pandas: Correlation With Python](https://realpython.com/numpy-scipy-pandas-correlation-python/#correlation)
### 2. Scipy.stats
* [Scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)模块包含大量概率分布、汇总和频率统计、相关函数和统计检验、掩蔽统计、核密度估计、准蒙特卡洛函数等，目前最新版本1.8.0.输出较为简洁。
* 强烈推荐阅读[官方文档](https://docs.scipy.org/doc/scipy/reference/stats.html)及[官方教程](https://docs.scipy.org/doc/scipy/tutorial/stats.html)，可参考国内[博客](https://blog.csdn.net/pipisorry/article/details/49515215)
* 下面以常用的统计分析方法在scipy下的api使用给出范例，参考[17 Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)
### 3. Statsmodels
* [statsmodels](http://www.statsmodels.org)是一个Python库，依赖Scipy.stats等包实现，用于拟合多种统计模型，执行统计测试以及数据探索和可视化，目前最新版本0.13.2.有统计图形输出功能。
### 4. Pingouin
* [Pingouin](https://pingouin-stats.org/#)是一个最新的完全基于Python3实现的依赖Scipy及Statsmodels等包的统计库。最新版本0.5.1.有统计图形输出功能。
### 5. 功能对比与小结
* Scipy.stats的统计结果最为简洁，输出最为简单，而剩下的几个工具包在其基础上进行了进一步的封装，不论是统计的输出或者输出结果的展示等方面都有着进一步的提升。
* 参考[Statistics with SciPy, Statsmodels, and Pingouin](https://python.cogsci.nl/numerical/statistics/)
### 6. 其他重要参考文献
* [Statistics in Python](https://www.reneshbedre.com/tags/#statistics)

## 二、统计分析检验工具选择
* 下图均来自pingouin的[guideline](https://pingouin-stats.org/guidelines.html)页面。
### 1. 方差分析(ANOVA)
![anova](https://pingouin-stats.org/_images/flowchart_one_way_ANOVA.svg)
### 2. 相关性分析(Correlation)
![corelation](https://pingouin-stats.org/_images/flowchart_correlations.svg)
### 3. 非参数检验(Non-parametric)
![Non-parametric](https://pingouin-stats.org/_images/flowchart_nonparametric.svg)

## 三、课题组此前项目使用过的统计分析工具
* 陈婉琳的博士课题《基于光电容积脉搏波的全身麻醉镇痛水平检测的研究》中使用过Kruskal-Wallis H Test、Wilcoxon Signed-Rank Test、Mann-Whitney U Test及Spearman’s Rank Correlation等工具

## 四、具体分析工具及Code Demo

### 1. 正态性检验
* 本部分列出了可用于检查数据是否具有高斯分布的统计检验

#### A. 夏皮罗-威尔克检验(Shapiro-Wilk Test)
* 目的
    * 测试数据样本是否具有高斯分布。
* 假设
    * 每个样本中的观测值都是独立且分布相同的(independent and identically distributed, IID)。
* 猜想
    * H0：样本具有高斯分布。
    * H1：样本没有高斯分布。
* 参考资料
    * [A Gentle Introduction to Normality Tests in Python](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
    * [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
    * [Shapiro-Wilk test on Wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
    * [国内博客](https://blog.csdn.net/lvsehaiyang1993/article/details/80473265)


```python
# Example of the Shapiro-Wilk Normality Test
from scipy.stats import shapiro
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')
```

    stat=0.895, p=0.193
    Probably Gaussian
    

#### B. D’Agostino’s K^2 Test(没找到中文翻译)
* 目的
    * 测试数据样本是否具有高斯分布。
* 假设
    * 每个样本中的观测值都是独立且分布相同的(independent and identically distributed, IID)。
* 猜想
    * H0：样本具有高斯分布。
    * H1：样本没有高斯分布。
* 参考资料
    * [A Gentle Introduction to Normality Tests in Python](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
    * [scipy.stats.normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
    * [D’Agostino’s K-squared test on Wikipedia](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)


```python
# Example of the D'Agostino's K^2 Normality Test
from scipy.stats import normaltest
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')
```

    stat=3.392, p=0.183
    Probably Gaussian
    

    C:\Environment\Python\Python39\lib\site-packages\scipy\stats\_stats_py.py:1477: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=10
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "
    

#### C. 安德森-达令检验(Anderson-Darling Test)
* 目的
    * 测试数据样本是否具有高斯分布。
* 假设
    * 每个样本中的观测值都是独立且分布相同的(independent and identically distributed, IID)。
* 猜想
    * H0：样本具有高斯分布。
    * H1：样本没有高斯分布。
* 参考资料
    *. [A Gentle Introduction to Normality Tests in Python](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
    *. [scipy.stats.anderson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)
    *. [Anderson-Darling test on Wikipedia](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)


```python
# Example of the Anderson-Darling Normality Test
from scipy.stats import anderson
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
result = anderson(data)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < cv:
		print('Probably Gaussian at the %.1f%% level' % (sl))
	else:
		print('Probably not Gaussian at the %.1f%% level' % (sl))
```

    stat=0.424
    Probably Gaussian at the 15.0% level
    Probably Gaussian at the 10.0% level
    Probably Gaussian at the 5.0% level
    Probably Gaussian at the 2.5% level
    Probably Gaussian at the 1.0% level
    

### 2. 相关性检验

#### A. 皮尔逊相关系数(Pearson’s Correlation Coefficient)
* 目的
    * 测试两个样本是否具有线性关系。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 每个样本中的观测值呈正态分布。
    * 每个样本中的观测值具有相同的方差。
* 猜想
    * H0：两个样本是独立的。
    * H1：样本之间存在依赖关系。
* 参考资料
    * [How to Calculate Correlation Between Variables in Python](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)
    * [scipy.stats.pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
    * [Pearson’s correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)


```python
# Example of the Pearson's Correlation test
from scipy.stats import pearsonr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
```

    stat=0.688, p=0.028
    Probably dependent
    

#### B. 斯皮尔曼等级相关系数(Spearman’s Rank Correlation)
* 目的
    * 测试两个样本是否具有单调关系。
* 假设
    * 每个样本中的观测值都是独立且分布相同的(independent and identically distributed, IID)。
    * 可以对每个样本中的观测值进行排名。
* 猜想
    * H0：两个样本是独立的。
    * H1：样本之间存在依赖关系。
* 参考资料
    * [How to Calculate Nonparametric Rank Correlation in Python](https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/)
    * [scipy.stats.spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
    * [Spearman’s rank correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)


```python
# Example of the Spearman's Rank Correlation Test
from scipy.stats import spearmanr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
```

    stat=0.855, p=0.002
    Probably dependent
    

#### C. 肯德尔等级系数(Kendall’s Rank Correlation)
* 目的
    * 测试两个样本是否具有单调关系。
* 假设
    * 每个样本中的观测值都是独立且分布相同的(independent and identically distributed, IID)。
    * 可以对每个样本中的观测值进行排名。
* 猜想
    * H0：两个样本是独立的。
    * H1：样本之间存在依赖关系。
* 参考资料
    * [How to Calculate Nonparametric Rank Correlation in Python](https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/)
    * [scipy.stats.kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
    * [Kendall rank correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)


```python
# Example of the Kendall's Rank Correlation Test
from scipy.stats import kendalltau
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
```

    stat=0.733, p=0.002
    Probably dependent
    

#### D. 卡方检验(Chi-Squared Test)
* 目的
    * 测试两个类别变量是相关还是独立。
* 假设
    * 计算列联表时使用的观测值是独立的。
    * 列联表的每个单元格中有 25 个或更多示例。
* 猜想
    * H0：两个样本是独立的。
    * H1：样本之间存在依赖关系。
* 参考资料
    * [A Gentle Introduction to the Chi-Squared Test for Machine Learning](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)
    * [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
    * [Chi-Squared test on Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test)


```python
# Example of the Chi-Squared Test
from scipy.stats import chi2_contingency
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
```

    stat=0.272, p=0.873
    Probably independent
    

### 3. 平稳性检验(Stationary Tests)(略)
* 本节列出了可用于检查时间序列是否平稳的统计检验。
#### A. 增强型 Dickey-Fuller 单元根测试(Augmented Dickey-Fuller Unit Root Test)
#### B. Kwiatkowski-Phillips-Schmidt-Shin

### 4. 参数统计假设检验(Parametric Statistical Hypothesis Tests)
* 本节列出了可用于比较数据样本的统计检验。

#### A. Student’s t-test
* 目的
    * 测试两个独立样本的均值是否显著不同。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 每个样本中的观测值呈正态分布。
    * 每个样本中的观测值具有相同的方差。
* 猜想
    * H0：样本的均值相等。
    * H1：样本的均值不相等。
* 参考资料
    * [How to Calculate Parametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)
    * [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
    * [Student’s t-test on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test)


```python
# Example of the Student's t-test
from scipy.stats import ttest_ind
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=-0.326, p=0.748
    Probably the same distribution
    

#### B. Paired Student’s t-test
* 目的
    * 测试两个配对样本的均值是否显著不同。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 每个样本中的观测值呈正态分布。
    * 每个样本中的观测值具有相同的方差。
    * 每个样本的观测值是成对的。
* 猜想
    * H0：样本的均值相等。
    * H1：样本的均值不相等。
* 参考资料
    * [How to Calculate Parametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)
    * [scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
    * [Student’s t-test on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test)


```python
# Example of the Paired Student's t-test
from scipy.stats import ttest_rel
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=-0.334, p=0.746
    Probably the same distribution
    

#### C. 方差分析(Analysis of Variance Test,ANOVA)
* 目的
    * 测试两个或多个独立样本的均值是否显著不同。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 每个样本中的观测值呈正态分布。
    * 每个样本中的观测值具有相同的方差。
* 猜想
    * H0：样本的均值相等。
    * H1：样本的一个或多个均值不相等。
* 参考资料
    * [How to Calculate Parametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)
    * [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
    * [Analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_variance)


```python
# Example of the Analysis of Variance Test
from scipy.stats import f_oneway
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=0.096, p=0.908
    Probably the same distribution
    

#### D. 重复测量方差分析检验(Repeated Measures ANOVA Test)
* 目的
    * 测试两个或多个配对样本的均值是否显著不同。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 每个样本中的观测值呈正态分布。
    * 每个样本中的观测值具有相同的方差。
    * 每个样本的观测值是成对的。
* 猜想
    * H0：样本的均值相等。
    * H1：样本的一个或多个均值不相等。
* 参考资料
    * [How to Calculate Parametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)
    * [Analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_variance)


```python
# scipy 没有相关模块，但 statsmodels.stats.anova 模块有相应功能
# 本例参考 https://www.statology.org/repeated-measures-anova-python/
# 参考 https://www.marsja.se/repeated-measures-anova-in-python-using-statsmodels/
import numpy as np
import pandas as pd

#create data
df = pd.DataFrame({'patient': np.repeat([1, 2, 3, 4, 5], 4),
                   'drug': np.tile([1, 2, 3, 4], 5),
                   'response': [30, 28, 16, 34,
                                14, 18, 10, 22,
                                24, 20, 18, 30,
                                38, 34, 20, 44, 
                                26, 28, 14, 30]})

#view first ten rows of data 
print(df.head(10))

from statsmodels.stats.anova import AnovaRM

#perform the repeated measures ANOVA
print(AnovaRM(data=df, depvar='response', subject='patient', within=['drug']).fit())
```

       patient  drug  response
    0        1     1        30
    1        1     2        28
    2        1     3        16
    3        1     4        34
    4        2     1        14
    5        2     2        18
    6        2     3        10
    7        2     4        22
    8        3     1        24
    9        3     2        20
                  Anova
    ==================================
         F Value Num DF  Den DF Pr > F
    ----------------------------------
    drug 24.7589 3.0000 12.0000 0.0000
    ==================================
    
    

### 5. 非参数统计假设检验(Nonparametric Statistical Hypothesis Tests)

#### A. 曼惠特尼U检验(Mann-Whitney U Test)
* 目的
    * 测试两个独立样本的分布是否相等。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 可以对每个样本中的观测值进行排名。
* 猜想
    * H0：两个样本的分布相等。
    * H1：两个样本的分布不相等。
* 参考资料
    * [How to Calculate Nonparametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
    * [scipy.stats.mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
    * [Mann-Whitney U test on Wikipedia](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)


```python
# Example of the Mann-Whitney U Test
from scipy.stats import mannwhitneyu
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=40.000, p=0.473
    Probably the same distribution
    

#### B. 威尔科克森符号秩检验(Wilcoxon Signed-Rank Test)
* 目的
    * 测试两个配对样本的分布是否相等。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 可以对每个样本中的观测值进行排名。
    * 每个样本的观测值是成对的。
* 猜想
    * H0：两个样本的分布相等。
    * H1：两个样本的分布不相等。
* 参考资料
    * [How to Calculate Nonparametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
    * [scipy.stats.wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
    * [Wilcoxon signed-rank test on Wikipedia](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)


```python
# Example of the Wilcoxon Signed-Rank Test
from scipy.stats import wilcoxon
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = wilcoxon(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=21.000, p=0.557
    Probably the same distribution
    

#### C. 克鲁斯卡尔-沃利斯检验(Kruskal-Wallis H Test)
* 目的
    * 测试两个或多个独立样本的分布是否相等。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 可以对每个样本中的观测值进行排名。
* 猜想
    * H0：所有样本的分布都相等。
    * H1：一个或多个样本的分布不相等。
* 参考资料
    * [How to Calculate Nonparametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
    * [scipy.stats.kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
    * [Kruskal-Wallis one-way analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)


```python
from scipy.stats import kruskal
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=0.571, p=0.450
    Probably the same distribution
    

#### D. 弗里德曼检验(Friedman Test)
* 目的
    * 测试两个或多个配对样本的分布是否相等。
* 假设
    * 每个样本中的观测值都是独立且分布相同的（iid）。
    * 可以对每个样本中的观测值进行排名。
    * 每个样本的观测值是成对的。
* 解释
    * H0：所有样本的分布都相等。
    * H1：一个或多个样本的分布不相等。
* 参考资料
    * [How to Calculate Nonparametric Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
    * [scipy.stats.friedmanchisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
    * [Friedman test on Wikipedia](https://en.wikipedia.org/wiki/Friedman_test)


```python
# Example of the Friedman Test
from scipy.stats import friedmanchisquare
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
```

    stat=0.800, p=0.670
    Probably the same distribution
    
