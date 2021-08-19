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
Train.info()。

![image](https://user-images.githubusercontent.com/26008298/130007523-c5214954-c6e5-4ac7-96b7-168a752dd1b6.gif)  

![image](https://user-images.githubusercontent.com/26008298/130007545-b8790420-c9a5-4039-87a1-996afdc3c90f.gif)
 
总共1460条数据，并不多，总共80个特征，一个ID标识。
通过观察可发现有一些特征存在空值。
存在空值的有Alley，MasVnrType，MasVnrArea，GarageArea，GarageQual，FireplaceQu，PoolQC，Fence，MiscFeature



通过相关性热力图，我找出了相关性前10的特征    

![image](https://user-images.githubusercontent.com/26008298/130007568-774c6f86-e1c4-42df-aafc-0778ba806385.gif)

通过观察和经验考虑，我认为
GarageCars和GarageArea说的几乎就是同一件事情（0.88），相关性比较重复。我选择GarageCars（可以停几辆车），不管GarageArea。
TotalBsmtSF和 1stFlrSF相关度也很高（0.82），我选择地下室面积TotalBsmtSF，不管1stFlrSF
GrLivArea 和 TotRmsAbvGrd 类似（0.83），我选择 GrLivArea


通过房价概率分布，我了解到大部分数据的房价在120000～180000，有几组数据比较特殊到达了600000

![image](https://user-images.githubusercontent.com/26008298/130007592-fe6398ba-35a7-4ec5-aa7b-d1d5ff180d18.gif)


结合热力图，我进行了散点分析，其中GrLivArea，TotalBsmtSF和SalPrice的散点明显指出有几个点不符合一般规律，综合考虑应当去除

![image](https://user-images.githubusercontent.com/26008298/130007605-eca179da-c1a5-4910-9461-ecd77bb93914.gif)

![image](https://user-images.githubusercontent.com/26008298/130007614-fac69218-6d1b-489b-9c5b-75026ccfa9a0.gif)

![image](https://user-images.githubusercontent.com/26008298/130007628-d2ae76dd-ae44-4369-8118-583e14cb3805.gif)


# 二．《数据预处理》  

step1.去除掉离群点

![image](https://user-images.githubusercontent.com/26008298/130007646-bb75c0e8-404a-4178-afcb-8dbf69c035bd.gif)


step2.合并训练集和测试集，丢弃不要的特征  

![image](https://user-images.githubusercontent.com/26008298/130007658-88f0d1db-cbcd-49e5-8699-7f70c9a2cfe7.gif)  

![image](https://user-images.githubusercontent.com/26008298/130007670-86dd7141-eafb-4cd6-8ab1-a598c0dcd165.gif)


step3.对缺失值进行填充  

![image](https://user-images.githubusercontent.com/26008298/130007685-0c266bc3-59ac-4563-8b05-83587666ed0a.gif)  

![image](https://user-images.githubusercontent.com/26008298/130007692-28c8c816-1c61-4c64-9061-9a0fa45f7d2a.gif)


step4.使用get_dummies进行one-hot编码  

![image](https://user-images.githubusercontent.com/26008298/130007705-1f63a858-8f5b-4f58-a5c5-300873445f8e.gif)


step5.训练集测试集分开  

![image](https://user-images.githubusercontent.com/26008298/130007714-ddc2bfeb-c89d-4806-ac69-b87dd3836757.gif)



# 三．《属性选择》  

## 构造新的特征
    create_feature(df)
## 丢弃特征
    df.drop(to_drop, axis=1, inplace=True)
## 这是构造的新特征

![image](https://user-images.githubusercontent.com/26008298/130007739-65080845-762c-4b46-a4fa-2115ef7c3638.gif)

## 这是丢弃的特征

![image](https://user-images.githubusercontent.com/26008298/130007754-3e61255f-6c54-4f68-8603-c7b231c71e92.gif)


# 四．《建模分析》  

![image](https://user-images.githubusercontent.com/26008298/130007773-ae930bd6-69e8-4d81-b9bc-195e343f2f53.gif)  

通过模型评估找出比较好的模型
models是预置的各个模型的集合

![image](https://user-images.githubusercontent.com/26008298/130007786-d4dd063f-f9a4-4cf2-a017-dfbb8ff8cd56.gif)

通我选择Ridge模型，而实际上有不少人使用了多模型混合的技术，我没有成功实现

# 五．《参数优化》
接下来进行Ridge模型的调参，我首先构造了grid类，用于方便参数的优化，该类实际上就是对于模型在不同参数下的模拟
首先，我以5为间隔测试Ridge模型  

![image](https://user-images.githubusercontent.com/26008298/130007816-d9d52263-7aaf-48b5-9a25-14d9ba3a1234.gif)

![image](https://user-images.githubusercontent.com/26008298/130007826-75faec1b-c25d-45d1-a710-f76513efc76c.gif)
  
观察发现，最小值应该处在5～15之间
于是第二次我将参数调小，从5～15，以1为间隔进行测试  

![image](https://user-images.githubusercontent.com/26008298/130007838-2b2fe3f5-c041-4f79-b399-d22887ede7fd.gif)

通过结果，我选择了alpha＝10作为最终的参数
回到正式代码
通过以下的函数进行Ridge回归

![image](https://user-images.githubusercontent.com/26008298/130007850-c7880480-b626-4733-a0d8-74a0f43bde03.gif)  


最终输出结果submission.csv
