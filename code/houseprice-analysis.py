import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from scipy import stats
from scipy.stats import norm

root_path = '/Users/tt/Desktop/kaggle_houseprice'

train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))
#print(train.columns)
train_corr=train.drop("Id",axis=1).corr()
#print(train_corr)

# 寻找K个最相关的特征信息，画出热力图

'''k = 10 # number of variables for heatmap
cols = train_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.5)
hm = plt.subplots(figsize=(20, 12))#调整画布大小
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()'''
#print(train.info())
#SalePrice 和相关变量之间的散点图
'''
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();'''

'''corr=train_corr["SalePrice"]
print(np.argsort(corr, axis=0))
corr[np.argsort(corr, axis=0)[::-1]]  #np.argsort()表示返回其排序的索引'''


#直方图
'''fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(train.SalePrice)
ax2.hist(np.log1p(train.SalePrice))
'''
'''
从直方图中可以看出：

* 偏离正态分布
* 数据正偏
* 有峰值
'''
'''
# 数据偏度和峰度度量：

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
'''
#正态分布概率
sns.distplot(train['SalePrice'] , fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
 
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
 
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


#缺失值分析
'''
    1. 对于缺失率过高的特征，例如 超过15% 我们应该删掉相关变量且假设该变量并不存在
    2. GarageX 变量群的缺失数据量和概率都相同，可以选择一个就行，例如：GarageCars
    3. 对于缺失数据在5%左右（缺失率低），可以直接删除/回归预测
'''
'''   total= train_test.isnull().sum().sort_values(ascending=False)
    percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Lost Percent'])

    print(missing_data[missing_data.isnull().values==False].sort_values('Total', axis=0, ascending=False).head(20))
'''

#双变量分析
'''var = 'YearBuilt'
#var = 'FullBath'
#var = 'GarageCars'
#var = 'TotalBsmtSF'
#var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()'''


# 删除点(查到不符合规律的点的对应id后，将其从数据集中删去)
'''test['SalePrice'] = None
train_test = pd.concat((train, test)).reset_index(drop=True)

print(train.sort_values(by='GrLivArea', ascending = False)[:2])
tmp = train_test[train_test['SalePrice'].isnull().values==False]

train_test = train_test.drop(tmp[tmp['Id'] == 1299].index)
train_test = train_test.drop(tmp[tmp['Id'] == 524].index)'''

