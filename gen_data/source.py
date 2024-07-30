"""生成源数据
    DESC: 生成指定数量的无重复的源数据
"""

import collections
import pandas as pd
from enum import Enum
from typing import List

from mimesis import Person, Internet
from mimesis.locales import Locale

from .distribution import DiscreteDistribution


class Info(Enum):
    FULL_NAME = Person(Locale.EN).full_name
    EMAIL = Person(Locale.EN).email
    IP_V4 = Internet().ip_v4
    USER_AGENT = Internet().user_agent


class Source:
    """生成源数据"""

    def __init__(self):
        self.ddist = DiscreteDistribution()

    @staticmethod
    def gen_feature(func, size, max_tries):
        """生成单一维度的特征"""
        assert size <= max_tries

        i = 0
        s = set()
        while len(s) < size:
            s.add(func())
            if i > max_tries:
                break
            i += 1

        assert len(s) == size, f'len(s)={len(s)} != size={size}'
        return list(s)

    def gen_features(self, size, max_tries, funcs: List[Info]):
        """生成多维度特征"""
        d = collections.defaultdict(list)
        for func in funcs:
            d[func.__name__] = self.gen_feature(func=func,
                                                size=size,
                                                max_tries=max_tries)
        return pd.DataFrame(d)
