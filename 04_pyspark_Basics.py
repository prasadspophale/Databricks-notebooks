from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

employee_data = [
    (101, "Rahul", "Sharma", 1),
    (102, "Priya", "Patil", 2),
    (103, "Amit", "Kumar", 1),
    (104, "Neha", "Joshi", 3)
]

employee_df = spark.createDataFrame(
    employee_data,
    ["EmpID", "FirstName", "LastName", "DeptID"]
)

employee_df.show()