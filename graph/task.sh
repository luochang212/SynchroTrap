spark-submit \
--conf spark.yarn.maxAppAttempts=1 \
--num-executors 10 \
--executor-cores 4 \
--jars ../jars/graphframes-0.8.4-spark3.5-s_2.12.jar \
  graph.py
