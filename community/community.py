import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame


T_sim = 120
THRESHOLD = 0.03
MIN_CNT = 1

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
    sum(if(ip_a=ip_b, 1, 0)) as ip_same_cnt,
    count(*) as cnt,
    round(sum(if(ip_a=ip_b, 1, 0)) / count(*), 3) as jaccard_sim
FROM
(
    SELECT
        a.uid as uid_a,
        b.uid as uid_b,
        a.ipv4 as ip_a,
        b.ipv4 as ip_b
    FROM
        user_table a
    JOIN
        user_table b ON a.uid < b.uid
        AND b.timestamp - a.timestamp <= {T_sim}
        AND b.timestamp - a.timestamp >= -{T_sim}
) t
GROUP BY uid_a, uid_b
HAVING jaccard_sim > {THRESHOLD}
    AND cnt > {MIN_CNT}
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
