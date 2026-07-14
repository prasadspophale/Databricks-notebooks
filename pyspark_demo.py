from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Demo") \
    .getOrCreate()

data = [
    (101, "Prasad", 50000),
    (102, "Rahul", 60000),
    (103, "Amit", 70000)
]

df = spark.createDataFrame(data, ["EmpID", "Name", "Salary"])

df.show()

spark.stop()