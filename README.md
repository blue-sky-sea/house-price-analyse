========================================
# house-price-analyse
========================================  

kaggle-house-price-data-analyse 

# 实验13: 综合应用  

# 一．《数据品质报告》  

数据总条目：1460
数据特征值数目：80（81）
有缺失数据条目：Alley，MasVnrType，MasVnrArea，GarageArea，GarageQual，FireplaceQu，PoolQC，Fence，MiscFeature
数据整体评价：数据量偏小，整体数据品质良好。存在少量缺失和异常数据，数据比较旧可能不太符合现在的预测规律

载入csv数据
查看colunms信息
Train.info()

总共1460条数据，并不多，总共80个特征，一个ID标识。
通过观察可发现有一些特征存在空值。
存在空值的有Alley，MasVnrType，MasVnrArea，GarageArea，GarageQual，FireplaceQu，PoolQC，Fence，MiscFeature



通过相关性热力图，我找出了相关性前10的特征

通过观察和经验考虑，我认为
GarageCars和GarageArea说的几乎就是同一件事情（0.88），相关性比较重复。我选择GarageCars（可以停几辆车），不管GarageArea。
TotalBsmtSF和 1stFlrSF相关度也很高（0.82），我选择地下室面积TotalBsmtSF，不管1stFlrSF
GrLivArea 和 TotRmsAbvGrd 类似（0.83），我选择 GrLivArea


通过房价概率分布，我了解到大部分数据的房价在120000～180000，有几组数据比较特殊到达了600000


结合热力图，我进行了散点分析，其中GrLivArea，TotalBsmtSF和SalPrice的散点明显指出有几个点不符合一般规律，综合考虑应当去除


# 二．《数据预处理》  

step1.去除掉离群点

step2.合并训练集和测试集，丢弃不要的特征

step3.对缺失值进行填充

step4.使用get_dummies进行one-hot编码 

step5.训练集测试集分开


# 三．《属性选择》  

## 构造新的特征
    create_feature(df)
## 丢弃特征
    df.drop(to_drop, axis=1, inplace=True)
这是构造的新特征

这是丢弃的特征

# 四．《建模分析》

通过模型评估找出比较好的模型
models是预置的各个模型的集合

通我选择Ridge模型，而实际上有不少人使用了多模型混合的技术，我没有成功实现

# 五．《参数优化》
接下来进行Ridge模型的调参，我首先构造了grid类，用于方便参数的优化，该类实际上就是对于模型在不同参数下的模拟
首先，我以5为间隔测试Ridge模型

观察发现，最小值应该处在5～15之间
于是第二次我将参数调笑，从5～15，以1为间隔进行测试

通过结果，我选择了alpha＝10作为最终的参数
回到正式代码
通过以下的函数进行Ridge回归

最终输出结果submission.Csv
