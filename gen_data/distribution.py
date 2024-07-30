"""生成样本量分布
    DESC: 生成各种分布下的分箱样本量。先指定分箱数，然后计算每个分箱的样本量
"""

import numpy as np


class DiscreteDistribution:
    """生成离散型样本量分布"""

    @staticmethod
    def uniform(num_samples, bins):
        """生成均匀分布的分箱样本数列表"""
        samples = np.random.uniform(low=0.0, high=1.0, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts
    
    @staticmethod
    def normal(scale, num_samples, bins):
        """生成正态分布的分箱样本数列表"""
        samples = np.random.normal(loc=0, scale=scale, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def poisson(lam, num_samples, bins):
        """生成泊松分布的分箱样本数列表"""
        samples = np.random.poisson(lam=lam, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def exponential(scale, num_samples, bins):
        """生成指数分布的分箱样本数列表"""
        samples = np.random.exponential(scale=scale, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def binomial(n, p, num_samples, bins):
        """生成二项分布的分箱样本数列表"""
        samples = np.random.binomial(n=n, p=p, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def chisquare(df, num_samples, bins):
        """生成卡方分布的分箱样本数列表"""
        samples = np.random.chisquare(df=df, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def gamma(shape, scale, num_samples, bins):
        """生成伽马分布的分箱样本数列表"""
        samples = np.random.gamma(shape=shape, scale=scale, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def beta(a, b, num_samples, bins):
        """生成Beta分布的分箱样本数列表"""
        samples = np.random.beta(a=a, b=b, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def lognormal(sigma, num_samples, bins):
        """生成对数正态分布的分箱样本数列表"""
        samples = np.random.lognormal(mean=0.0, sigma=sigma, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def pareto(a, num_samples, bins):
        """生成对Pareto分布的分箱样本数列表"""
        samples = np.random.pareto(a=a, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts

    @staticmethod
    def weibull(a, num_samples, bins):
        """生成对Weibull分布的分箱样本数列表"""
        samples = np.random.weibull(a=a, size=num_samples)
        bin_counts, _ = np.histogram(samples, bins=bins)
        return bin_counts
