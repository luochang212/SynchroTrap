import os
import collections
import random
import datetime
import numpy as np
import pandas as pd
import sklearn.metrics

from gen_data.source import Source, Info
from gen_data.distribution import DiscreteDistribution


# 一天的秒数
SEC = 24 * 60 * 60


def gen_abspath(directory: str, rel_path: str) -> str:
    """
    Generate the absolute path by combining the given directory with a relative path.

    :param directory: The specified directory, which can be either an absolute or a relative path.
    :param rel_path: The relative path with respect to the 'dir'.
    :return: The resulting absolute path formed by concatenating the absolute directory
             and the relative path.
    """
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


def read_csv(
    file_path: str,
    sep: str = ',',
    header: int = 0,
    on_bad_lines: str = 'warn',
    encoding: str = 'utf-8',
    dtype: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file from the specified path.
    """
    return pd.read_csv(file_path,
                       header=header,
                       sep=sep,
                       on_bad_lines=on_bad_lines,
                       encoding=encoding,
                       dtype=dtype,
                       **kwargs)


def gen_normal_samples(scale, num_samples, bins, func):
    """生成符合正态分布的样本

    :param scale: 正态分布的标准差
    :param num_samples: 样本总数
    :param bins: 分箱数，即无重复的样本数
    :param func: 样本值生成函数
    """

    # 生成符合正态分布的样本量
    sample_sizes = DiscreteDistribution.normal(scale=scale,
                                               num_samples=num_samples,
                                               bins=bins)

    # 生成无重复的 样本列表
    # 生成的样本量与分箱数 bins 相同
    s = Source()
    unique_sample_list = s.gen_feature(func=func,
                                       size=bins,
                                       max_tries=1000)

    # 打散 样本列表
    random.shuffle(unique_sample_list)
    samples_dict = { k:v for k, v in zip(unique_sample_list, sample_sizes) }

    return samples_dict


def gen_user_table(uid_num, poisson_lambda, ip_scale, ip_bins):
    """将 uid 和 ipv4 随机组合在一起，得到用户日志表"""

    query_num = np.random.poisson(lam=poisson_lambda, size=uid_num)

    # 生成 uid 字典，key 是 uid，value 是该 uid 产生的 query 数
    uid_dict = { k:v for k, v in zip(range(uid_num), query_num) if v > 0 }

    query_sum = sum(query_num)

    # 然后生成 query_sum 个 ipv4 样本
    ipv4_dict = gen_normal_samples(scale=ip_scale,
                                   num_samples=query_sum,
                                   bins=ip_bins,
                                   func=Info.IP_V4)

    # 将 uid 和 ipv4 展平后，随机混合在一起
    uid_list = [ k for k, v in uid_dict.items() for _ in range(v) ]
    ipv4_list = [ k for k, v in ipv4_dict.items() for _ in range(v) ]
    random.shuffle(ipv4_list)

    user_table = collections.defaultdict(list)
    for uid, ipv4 in zip(uid_list, ipv4_list):
        user_table['uid'].append(uid)
        user_table['ipv4'].append(ipv4)

    return pd.DataFrame(user_table)


def lambda_func(t: int, epsilon: list, k: float = 1):
    """泊松分布的 lambda 关于时间 t 的函数

    :param t: 一天中的时间，单位是秒
    :param epsilon: 扰动项
    :param k: 放缩系数
    """
    t = t % SEC
    e = epsilon[t]
    y = 0.1-(t-(SEC/2))**2/(9*10**9)
    y = 0 if y < 0 else round(y, 5)
    return k * (y + e)


def seconds_to_datetime(seconds):
    """将今天的第 x 秒转换为今天的时间"""
    t = (datetime.datetime.now() + datetime.timedelta(seconds=seconds))
    return t.strftime("%Y-%m-%d %H:%M:%S"), int(t.timestamp())


def gen_time(sample_num, init_t):
    """假设用户访问是一个泊松过程，生成用户访问的 时间 和 时间戳

    :param sample_num: 样本量
    :param init_t: 初始秒数
    """

    # 一个期望值很小的均匀分布
    epsilon = np.random.uniform(low=0.0, high=0.01, size=SEC)

    # 访问次数计数
    i = 0

    # 时间计数（秒）
    t = init_t

    time_list = []
    while i < sample_num:
        lam = lambda_func(t, epsilon, k=0.8)
        if random.random() < lam:
            # 用户到达
            i += 1
            time_list.append(t)
        t += 1

    date_dict = collections.defaultdict(list)
    for e in time_list:
        time_str, time_stamp = seconds_to_datetime(e)
        date_dict['time'].append(time_str)
        date_dict['timestamp'].append(time_stamp)

    return pd.DataFrame(date_dict)


def gen_user_df():
    # 生成 uid 和 ip
    user_table = gen_user_table(uid_num=1000,
                                poisson_lambda=2,
                                ip_scale=0.5,
                                ip_bins=800)
    
    # 将 uid 和 ip 的 index 随机化
    user_table = user_table.sample(frac=1, random_state=37).reset_index(drop=True)
    
    # 生成 时间 和 时间戳
    time_table = gen_time(sample_num=len(user_table), init_t=8*60*60)
    
    # 将两张表 concat 起来
    user_df = pd.concat([user_table, time_table], axis=1)

    # 按时间戳排序
    sorted_user_df = user_df.sort_values(by='timestamp', ascending=True)

    return sorted_user_df



def calc_attack_start_time(attack_duration, start_time, end_time):
    """随机选择攻击开始时间"""

    total_seconds = int((end_time - start_time).total_seconds())
    assert total_seconds > attack_duration
    total_seconds -= attack_duration
    random_seconds = random.randint(0, total_seconds)
    attack_start_time = start_time +  datetime.timedelta(seconds=random_seconds)

    return attack_start_time


def gen_ip_set(ip_num, normal_ip_list, normal_ip_rate):
    """获取 ip 池

    :param ip_num: ip 总数
    :param normal_ip_list: 大盘 ip 列表 
    :param normal_ip_rate: 坏人的 ip 池与大盘重合的比例
    """
    repeat_num = int(ip_num * normal_ip_rate)
    assert len(normal_ip_list) >= repeat_num
    ip_set = set(random.sample(normal_ip_list, repeat_num))

    # 不足的用 gen_feature 方法补足
    rest = ip_num - repeat_num
    s = Source() 
    ip_set |= set(s.gen_feature(func=Info.IP_V4,
                                size=rest,
                                max_tries=1000))

    assert len(ip_set) == ip_num

    return ip_set


def attack(attack_duration, ip_set, min_t, max_t):
    """记录攻击 ip 和 攻击秒数"""
    t = 0
    assert max_t > min_t
    t_diff = max_t - min_t
    ip_list = list(ip_set)
    log = []
    while t < attack_duration:
        stop_time = min_t + int(t_diff * random.random())
        t += stop_time
        ip = random.sample(ip_list, 1)[0]
        log.append((ip, t))

    return log


def gen_attack_df(
    attack_duration: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    ip_num: int,
    normal_ip_list: list,
    normal_ip_rate: float,
    uid_repeat_rate: float,
    min_t: int,
    max_t: int,
    epoch: int
) -> pd.DataFrame:
    """
    生成攻击日志

    :param attack_duration: 攻击持续时间
    :param start_time: 大盘数据的最早时间
    :param end_time: 大盘数据的最晚时间
    :param ip_num: ip 资源池大小
    :param normal_ip_list: 正常 ip 列表
    :param normal_ip_rate: 正常 ip 的比例
    :param uid_repeat_rate: 当前时刻复用之前用过的 uid 的概率
    :param min_t: 最小攻击间隔，单位秒
    :param max_t: 最大攻击间隔，单位秒
    :param epoch: 攻击线程数
    """
    attack_dict = collections.defaultdict(list)

    attack_start_time = calc_attack_start_time(attack_duration, start_time, end_time)

    ip_set = gen_ip_set(ip_num=ip_num,
                        normal_ip_list=normal_ip_list,
                        normal_ip_rate=normal_ip_rate)

    uid = -1
    for _ in range(epoch):
        log = attack(attack_duration, ip_set, min_t, max_t)
        for ip, attack_seconds in log:

            # uid 有 uid_repeat_rate 的概率复用之前用过的 uid
            if random.random() < uid_repeat_rate:
                uid = random.sample(range(uid, 0), 1)[0]
            else:
                uid -= 1

            attack_dict['uid'].append(uid)
            attack_dict['ipv4'].append(ip)
    
            attack_time = attack_start_time + datetime.timedelta(seconds=attack_seconds)
            attack_dict['time'].append(attack_time.strftime("%Y-%m-%d %H:%M:%S"))
            attack_dict['timestamp'].append(int(attack_time.timestamp()))

    return pd.DataFrame(attack_dict)


def eval_binary(
    y_true,
    y_label
):
    """
    Evaluate a binary classification task.
    """

    # Metrics that require the predicted labels (y_label)
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_label)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_label)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_label)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    tn, fp, fn, tp = cm.ravel()

    print(f'accuracy: {acc:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall: {recall:.5f}')
    print(f'f1_score: {f1:.5f}')
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(f'confusion matrix:\n{cm}')
