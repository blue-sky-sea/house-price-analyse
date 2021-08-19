#!/Users/tt/Desktop/kaggle_houseprice
# coding: utf-8
import time
import imp
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from scipy.stats import boxcox
from sklearn.linear_model import Ridge
import os.path
import warnings
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from operator import itemgetter
import itertools
warnings.filterwarnings('ignore')



warnings.filterwarnings('ignore')
#文件所在路径
data_dir = '/Users/tt/Desktop/kaggle_houseprice'
# 对结果影响很小,或者与其他特征相关性较高的特征将被丢弃
to_drop = [
    'Street', 'Utilities', 'Condition2', 'PoolArea', 'PoolQC', 'Fence',
    'YrSold', 'MoSold', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageQual', 'MiscVal',
    'EnclosedPorch', '3SsnPorch', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt',
    'BsmtFinType2', 'BsmtUnfSF', 'GarageCond', 'GarageFinish', 'FireplaceQu',
    'BsmtCond', 'BsmtQual', 'Alley'
]
# 加载数据
def opencsv():
    # 使用 pandas 打开csv文件
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return df_train, df_test


def saveResult(result):
    #保存为csv文件
    result.to_csv(
        os.path.join(data_dir, "submission.csv"), sep=',', encoding='utf-8')


def ridgeRegression(trainData, trainLabel, df_test):
    #设置α项，其值越大正则化项越大。
    ridge = Ridge(
        alpha=10.0
    )  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    ridge.fit(trainData, trainLabel)
    #使用ridge的predict方法进行预测
    predict = ridge.predict(df_test)
    #取预测结果的SalePrice列，并添加Id列去标识
    pred_df = pd.DataFrame(predict, index=df_test["Id"], columns=["SalePrice"])
    return pred_df


def create_feature(data):
    # 是否拥有地下室
    hBsmt_index = data.index[data['TotalBsmtSF'] > 0]
    data['HaveBsmt'] = 0
    data.loc[hBsmt_index, 'HaveBsmt'] = 1
    data['house_remod'] = data['YearRemodAdd'] - data['YearBuilt']
    data['livingRate'] = (data['GrLivArea'] /
                          data['LotArea']) * data['OverallCond']
    data['lot_area'] = data['LotFrontage'] / data['GrLivArea']
    data['room_area'] = data['TotRmsAbvGrd'] / data['GrLivArea']
    data['fu_room'] = data['FullBath'] / data['TotRmsAbvGrd']
    data['gr_room'] = data['BedroomAbvGr'] / data['TotRmsAbvGrd']


def dataProcess(df_train, df_test):
    #将训练集的离群点去除
    df_train= df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

    #trainLabel是训练集的标签，即最终的售价，是要预测和比对的值
    trainLabel = df_train['SalePrice']

    # 因为删除了几行数据,所以index的序列不再连续,需要重新reindex
    df_train.reset_index(drop=True, inplace=True)
    #prices = np.log1p(df_train.loc[:, 'SalePrice'])
    df_train.drop(['SalePrice'], axis=1, inplace=True)

    #df是训练集和测试集的总和数据集
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    #构造新的特征
    create_feature(df)
    # 丢弃特征
    df.drop(to_drop, axis=1, inplace=True)
    
    # 填充None值,因为在特征说明中,None也是某些特征的一个值,所以对于这部分特征的缺失值以None填充
    fill_none = ['MasVnrType', 'BsmtExposure', 'GarageType', 'MiscFeature']
    for col in fill_none:
        df[col].fillna('None', inplace=True)

    # 对其他缺失值进行填充,离散型特征填充众数,数值型特征填充中位数
    na_col = df.dtypes[df.isnull().any()]
    for col in na_col.index:
        if na_col[col] != 'object':
            med = df[col].median()
            df[col].fillna(med, inplace=True)
        else:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            
    #dropna滤除缺失值，axis＝1表示有空值删除整列，
    '''
            inplace＝True表示
                    修改一个对象时：
                    inplace=True：不创建新的对象，直接对原始对象进行修改；
                    inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
    '''
    #df.dropna(axis=1, inplace=True)

    #使用get_dummies进行one-hot编码
    #因为很多时候，特征并不总是连续值，而有可能是分类。将特征值用数字表示效率将会快很多
    df = pd.get_dummies(df)

    #将训练集从总和数据集中分出来
    trainData = df[:df_train.shape[0]]
    #将测试集从总和数据中分出来
    test = df[df_train.shape[0]:]

    return trainData, trainLabel, test


def Regression_ridge():
    #当前开始的时间
    start_time = time.time()

    # 加载数据
    df_train, df_test = opencsv()
    print("load data finish")
    
    #加载数据结束的时间
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))

    # 数据预处理
    train_data, trainLabel, df_test = dataProcess(df_train, df_test)

    # 模型训练预测
    result = ridgeRegression(train_data, trainLabel, df_test)

    # 结果的输出
    saveResult(result)
    print("run finish!")
    #整个预测结束运行的时间
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))
    print("sucess!")

# 定义验证函数
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]
            
           
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]

    
            return X
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#模型评估分析
def model_evaluate():
    # 加载数据
    df_train, df_test = opencsv()
    y_log = df_train['SalePrice']
    print("load data finish")
    
    # 数据预处理
    X_scaled, y_log, df_test = dataProcess(df_train, df_test)
    
    '''pca = PCA(n_components=410)
    X_scaled=pca.fit_transform(X_scaled)
    test_X_scaled = pca.transform(test_X_scaled)'''
    np.set_printoptions(threshold='nan')  #全部输出
    models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]
    
    #模型评估
    '''names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
    for name, model in zip(names, models):
        score = rmse_cv(model, X_scaled,y_log)
        print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))'''

    #grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})
    grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[5,6,7,8,9,10,11,12,13,14,15]})
if __name__ == '__main__':
    model_evaluate()

    
