spark-submit \
--master local[1] \
--conf spark.yarn.maxAppAttempts=1 \
--executor-memory 512M \
--driver-memory 512M \
--jars ../jars/graphframes-0.8.4-spark3.5-s_2.12.jar \
  community.py
