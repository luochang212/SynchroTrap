# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from graphframes import GraphFrame


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Graph") \
    .getOrCreate()

# Create DataFrames for vertices and edges
vertices = spark.createDataFrame([
    ("a", "Alice", 34),
    ("b", "Bob", 36),
    ("c", "Charlie", 30),
    ("d", "David", 29),
    ("e", "Esther", 32),
    ("f", "Fanny", 36),
    ("g", "Gabby", 60)
], ["id", "name", "age"])

vertices.show()

edges = spark.createDataFrame([
    ("a", "b", "friend"),
    ("b", "c", "follow"),
    ("c", "b", "follow"),
    ("f", "c", "follow"),
    ("e", "f", "follow"),
    ("e", "d", "friend"),
    ("d", "a", "friend"),
    ("a", "e", "friend")
], ["src", "dst", "relationship"])

# Create a GraphFrame
g = GraphFrame(vertices, edges)

# Run a basic graph algorithm - strongly connected components
results = g.stronglyConnectedComponents(5)

# Show the result
results.show()
