import argparse
import os
import subprocess
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct
from pyspark.sql.types import StringType, BinaryType, StructType, StructField
import boto3
import io
import sys
import subprocess
import datetime
import tensorflow as tf


 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-uri', type=str, default='s3://sagemaker-us-west-2-*******/poc/raw3/2024/06/17/*')
    parser.add_argument('--output-data-uri', type=str, default='s3://sagemaker-us-west-2-*******/tf_train/features')
    parser.add_argument('--num-partitions', type=int, default=10)
    return parser.parse_args()




def process_feature(value,column):
    result = []
    #print("column",column)
    features = value.split('\x01')
    for feature in features:
        #print(f"features==={feature}")
        if feature:
            hash_value, weight = feature.split('\x03')
            hash_int = int(hash_value)
            weight_float = float(weight)
            if weight_float == 0:
                weight = "0.00000001"
        else:
            hash_value="000000000"
            weight="0.00000000001"
        result.append(f"{hash_value}\x03{weight}")
    return '\x01'.join(result)




def create_example(row, feature_columns):
    feature = {}
    for col in feature_columns:
        if '\x01' in row[col]:
            values = row[col].split('\x01')
        else:
            values = [row[col]]
        
        int64_list = []
        float_list = []
        
        for value in values:
            parts = value.split('\x03')
            if len(parts) >= 1:
                try:
                    int64_list.append(str(parts[0]))
                except ValueError:
                    print("too big to convert into int ",str(parts[0]))
                    int64_list.append(0)
            
            if len(parts) >= 2:
                try:
                    float_list.append(float(parts[1]))
                except ValueError:
                    float_list.append(0.0)

        feature[f"id_{col}"] = int64_list
        feature[f"weighted_id_{col}"] = float_list

    feature['target'] = int(row['label'])
    return feature

def process_partition(iterator, feature_columns):
    
    for row in iterator:
        processed_row = {}
        for column in feature_columns:
            processed_row[column] = process_feature(row[column],column)
        processed_row['label'] = row['label']
        yield create_example(processed_row, feature_columns)


        
def main():
    args = parse_args()    
    spark = SparkSession.builder\
            .appName("FeatureProcessing") \
            .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console -Dlog4j.logger.org.apache.spark=ERROR -Dlog4j.logger.org.apache.hadoop=ERROR") \
            .config("spark.executor.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console -Dlog4j.logger.org.apache.spark=ERROR -Dlog4j.logger.org.apache.hadoop=ERROR") \
            .getOrCreate()
    sc = spark.sparkContext
    

    df = spark.read.format("orc").option("compression", "snappy").load(args.input_data_uri)
    
    ## 拆分到多个分区
    df = df.repartition(args.num_partitions)
    
    feature_columns = [col for col in df.columns if col.startswith('_')]
    
    df_processed = df.rdd.mapPartitions(
        lambda iterator: process_partition(iterator, feature_columns)
    ).toDF()

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_data_uri}_{timestamp}"

    df_processed.write.format("tfrecord").option("recordType", "Example").save(output_path)
    
    spark.stop()

if __name__ == "__main__":
    main()
