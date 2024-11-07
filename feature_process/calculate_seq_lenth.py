from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, length,explode, max,split, when,collect_set
from pyspark.sql.functions import split, explode, collect_set
from pyspark.sql.functions import udf
from pyspark.sql.types import MapType, StringType, ArrayType, IntegerType
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import create_map
from pyspark.sql.functions import col, split, collect_set, array, lit, to_json, struct
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FeatureColumnStats").getOrCreate()

# 读取数据
df = spark.read.format("orc") \
    .option("compression", "snappy") \
    .option("recursiveFileLookup", "true") \
    .load("s3://sagemaker-us-west-2-*****/cust-poc/raw/")

feature_columns = [col for col in df.columns if col.startswith('_')]


def process_column(dataframe, column_name):
    return dataframe.agg(
        lit(column_name).alias("column_name"),
        max(
            when(
                (col(column_name).isNull()) | (col(column_name) == ""),
                0
            ).otherwise(
                size(split(col(column_name), '\x01'))
            )
        ).alias("max_sequence_length")
    )


feature_dfs = [process_column(df, col_name) for col_name in feature_columns]


result_df = reduce(DataFrame.unionByName, feature_dfs)



json_df = result_df.select(
    to_json(create_map("column_name", "max_sequence_length")).alias("json")
)


s3_output_path = "s3://sagemaker-us-west-2-*****/cust-poc/output/"
json_df.repartition(1).write.mode("overwrite").json(s3_output_path)
