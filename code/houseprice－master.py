#!/Users/tt/Desktop/kaggle_houseprice
# coding: utf-8
import time
import pandas as pd
from sklearn.linear_model import Ridge
import os.path

#文件所在路径
data_dir = '/Users/tt/Desktop/kaggle_houseprice'


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


def dataProcess(df_train, df_test):
    #trainLabel是训练集的标签，即最终的售价，是要预测和比对的值
    trainLabel = df_train['SalePrice']
    #df是训练集和测试集的总和数据集
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    #dropna滤除缺失值，axis＝1表示有空值删除整列，
    '''
            inplace＝True表示
                    修改一个对象时：
                    inplace=True：不创建新的对象，直接对原始对象进行修改；
                    inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
    '''
    df.dropna(axis=1, inplace=True)

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
    print("finish!")
    #整个预测结束运行的时间
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))


if __name__ == '__main__':
    Regression_ridge()
