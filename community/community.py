# -*- coding: utf-8 -*-

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame


T_sim = 300
THRESHOLD = 0.05
MIN_CNT = 2

FILE_PATH = '../data/'
USER_FILE = 'user.csv'
RESULT_PATH = 'component'


def gen_abspath(directory, rel_path):
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Community") \
    .getOrCreate()

# 只能在单节点 Spark 应用这么写
csv_file_path = gen_abspath(directory=FILE_PATH, rel_path=USER_FILE)
spark_df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# 建临时表
spark_df.createOrReplaceTempView("user_table")

sim = spark.sql(f"""
SELECT
    uid_a,
    uid_b,
    round(same_cnt / all_cnt, 4) as jaccard_sim
FROM
(
    SELECT
        a.uid as uid_a,
        b.uid as uid_b,
        count(distinct if(a.ipv4=b.ipv4, a.ipv4, null)) as same_cnt,
        size(array_distinct(split(concat_ws(',',
            concat_ws(',', collect_set(a.ipv4)),
            concat_ws(',', collect_set(b.ipv4))
        ), ','))) as all_cnt
    FROM
        user_table a
    JOIN
        user_table b ON a.uid < b.uid
        AND b.timestamp - a.timestamp <= {T_sim}
        AND b.timestamp - a.timestamp >= -{T_sim}
    GROUP BY uid_a, uid_b
) t
WHERE (same_cnt / all_cnt) > {THRESHOLD}
    AND all_cnt > {MIN_CNT}
""")

# 获取点
vertices = sim.select(col("uid_a")) \
    .withColumnRenamed("uid_a", "id") \
    .union(sim.select(col("uid_b")) \
           .withColumnRenamed("uid_b", "id")) \
    .distinct()

# 获取边
edge1 = sim.select([col("uid_a"), col("uid_b")]) \
    .withColumnRenamed("uid_a", "src") \
    .withColumnRenamed("uid_b", "dst")

edge2 = sim.select([col("uid_b"), col("uid_a")]) \
    .withColumnRenamed("uid_b", "src") \
    .withColumnRenamed("uid_a", "dst")

edges = edge1.union(edge2)

# 建图
g = GraphFrame(vertices, edges)

# 计算全连接
results = g.stronglyConnectedComponents(20)

# 计算社区大小
results.createOrReplaceTempView("results")

result_with_size = spark.sql(f"""
SELECT
    a.id,
    a.component,
    b.size
FROM
    results a
LEFT JOIN
(
    SELECT
        component,
        count(distinct id) AS size
    FROM
        results
    GROUP BY
        component
) b ON a.component = b.component
ORDER BY
    b.size DESC
""")

# 在日志中输出结果
result_with_size.show()

# 把结果存成 CSV
res_path = gen_abspath(directory=FILE_PATH, rel_path=RESULT_PATH)
result_with_size.coalesce(1) \
    .write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save(res_path)
