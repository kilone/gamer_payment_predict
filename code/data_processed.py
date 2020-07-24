import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime


def get_data():
    '''读取数据'''

    path =r'D:\Dataset\Tap4fun'
    train = pd.read_csv(os.path.join(path, 'tap_fun_train.csv'))
    test = pd.read_csv(os.path.join(path, 'tap_fun_test.csv'))
    return train, test


def transforn_dtypes(data):
    """转换数据类型，节省内存占用，提高运行效率"""

    data['register_time'] = pd.to_datetime(data['register_time'])
    int64 = data.select_dtypes(include='int64').columns.values
    float64 = data.select_dtypes(include='float64').columns.values
    data[int64] = data[int64].astype(np.int32)
    data[float64] = data[float64].astype(np.float32)
    return data


def get_time_features(data):
    """注册日期相关的特征"""

    #注册日期在星期几
    data.loc[ : , 'weekday'] = data['register_time'].dt.weekday
    #注册日期是否为周末
    data.loc[ : , 'is_weekend'] = data['weekday'].map(lambda x : 1 if x >4 else 0)
    #注册时间为周末且付费的玩家数
    data.loc[ : , 'weekend_payed_num'] = data[(data['is_weekend'] == 1) & (data['pay_price'] > 0)]['user_id'].count()
    # 注册时间为周末的付费玩家占总付费玩家的比例
    data.loc[ : , 'pct_of_weekend_payed'] = round(data['weekend_payed_num']/data[data['pay_price'] > 0]['user_id'].count(), 2)
    return data

def resource_used_rate(data, resource_add, resource_reduce):
    """资源获取与资源消耗的比率"""

    # 有些列名首个单词重复，所以要区分开命名
    if resource_add.count('_') == 2:
        feature_name = resource_add.split('_')[0] + '_used_rate'
    else:
        feature_name = resource_add.split('_')[0] + '_' + resource_add.split('_')[1] +  '_used_rate'
    
    data.loc[ : ,feature_name] = round(data[resource_reduce] / data[resource_add], 2)
    #分母为零时，计算的结果为缺失值，需要用0补充
    data.loc[ : ,feature_name].fillna(0, inplace=True)
    data.loc[ : ,feature_name] = data.loc[ : ,feature_name].replace([np.inf, -np.inf], 0)
    return data

def get_online_time_type(row):
    """根据付费玩家的平均在线时长的四分位数划分在线时长类别"""

    if row < 50:
        x = 1
    elif row >= 50 and row < 125:
        x = 2
    elif row >= 125 and row < 250:
        x =3
    else:
        x = 4
    return x

def get_pv_features(data):
    """获取玩家PVP,PVE特征"""

    # 主动发起PVP次数占总PVP次数的比值
    data.loc[ : , 'pvp_lanch_rate'] = round(data.loc[ : , 'pvp_lanch_count'] / data.loc[ : , 'pvp_battle_count'], 2)
    # PVP胜率
    data.loc[ : , 'pvp_win_rate'] = round(data.loc[ : , 'pvp_win_count'] / data.loc[ : , 'pvp_battle_count'], 2)
    # pve胜率
    data.loc[ : , 'pve_win_rate'] = round(data.loc[ : , 'pve_win_count'] / data.loc[ : , 'pve_battle_count'], 2)
    # 平均充值金额
    data.loc[ : , 'avg_pay'] = data.loc[ : , 'pay_price'] / data.loc[ : , 'pay_count']
    return data

def get_features(data):
    """构建新的玩家特征"""

    get_time_features(data)
    for i in range(2, 33, 2):
        resource_add = data.iloc[ : , i ].name
        resource_reduce = data.iloc[ : , i + 1].name
        #print(resource_add, resource_reduce)
        resource_used_rate(data, resource_add, resource_reduce)
    # 根据付费玩家的平均在线时长的四分位数划分在线时长类别
    data.loc[ : , 'avg_online_minutes_type'] = data.loc[ : , 'avg_online_minutes'].apply(get_online_time_type)
    get_pv_features(data)
    return data


if __name__ == '__main__':
    strat_time = datetime.datetime.now()
    processed_data_path = 'processed_data'
    print(strat_time.strftime('%Y-%m-%d %H:%M:%S'))
    #获取数据
    train, test = get_data()
    #数据类型转换
    train = transforn_dtypes(train)
    test = transforn_dtypes(test)
    #构建特征
    processed_train = get_features(train)
    processed_test = get_features(test)
    # 保存数据
    processed_train.to_csv(os.path.join(processed_data_path, 'processed_train.csv'), index=None, header=None)
    processed_test.to_csv(os.path.join(processed_data_path, 'processed_test.csv'), index=None, header=None)

    # 花费时间打印
    print((datetime.datetime.now() - strat_time).seconds)
