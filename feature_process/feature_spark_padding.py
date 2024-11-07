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



def load_feature_hash_dict(file_path):
    bucket_name, key = file_path.replace("s3://", "").split("/", 1)
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    raw_data = obj['Body'].read().decode('utf-8')
 
    feature_hash_dict = {}
    for line in raw_data.splitlines():
        line_data = json.loads(line)
        inner_json = json.loads(line_data["json"])
        for k, v in inner_json.items():
            feature_hash_dict[k] = set(int(x) for x in v if x.strip())
    
    return feature_hash_dict 

def load_feature_lenth_dict(file_path):
    bucket_name, key = file_path.replace("s3://", "").split("/", 1)
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    raw_data = obj['Body'].read().decode('utf-8')
 
    feature_lenth_dict = {}
    for line in raw_data.splitlines():
        line_data = json.loads(line)
        inner_json = json.loads(line_data["json"])
        for k, v in inner_json.items():
            feature_lenth_dict[k] = int(v)   
    return feature_lenth_dict    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-hash-dict', type=str, default='s3://sagemaker-us-west-2-687912291502/tf_train/features_20240710_041505/part-00000-757be5db-4d45-493f-a7b9-836104157742-c000.json')
    parser.add_argument('--feature-lenth-dict', type=str, default='s3://sagemaker-us-west-2-687912291502/tf_train/features_20240710_041505/part-00000-f02ce9a7-3057-4b81-b6aa-520954364de1-c000.json')
    parser.add_argument('--input-data-uri', type=str, default='s3://sagemaker-us-west-2-687912291502/poc/raw3/2024/06/18/')
    parser.add_argument('--output-data-uri', type=str, default='s3://sagemaker-us-west-2-687912291502/tf_train/features')
    parser.add_argument('--num-partitions', type=int, default=10)
    parser.add_argument('--padding-lenth', type=int, default=10)
    return parser.parse_args()



def process_feature(value, hash_set, column_lenth,padding_lenth):
    result = []

    if value:
        features = value.split('\x01')
        for feature in features:
            hash_value, weight = feature.split('\x03')
            hash_int = int(hash_value)
            if float(weight) == 0:
                weight = "0.00000001"
            result.append(f"{hash_value}\x03{weight}")
            if len(result) >= padding_lenth:
                break  # if padding values has more than padding lenth, stop the process

    # if padding value less than the padding lenth，random choose the key from the unique feature hash_set
    if len(result) < column_lenth and len(result) < padding_lenth:
        fill_count = 1
        while len(result) < column_lenth:
            fill_hash = f"00000000000000{fill_count}"
            result.append(f"{fill_hash}\x030.000000001")
            fill_count = fill_count+1
            if len(result) >= padding_lenth:
                break
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

def process_partition(iterator, feature_columns,FEATURE_HASH_DICT_BROADCAST,FEATURE_LENTH_DICT_BROADCAST,padding_lenth):
    feature_hash_dict = FEATURE_HASH_DICT_BROADCAST.value
    feature_lenth_dict = FEATURE_LENTH_DICT_BROADCAST.value
    
    for row in iterator:
        processed_row = {}
        for column in feature_columns:
            if column in feature_hash_dict:
                processed_row[column] = process_feature(row[column], feature_hash_dict[column], feature_lenth_dict[column],padding_lenth)
            else:
                processed_row[column] = row[column]
        processed_row['label'] = row['label']
        yield create_example(processed_row, feature_columns)

def main():
    args = parse_args()
    FEATURE_HASH_DICT = load_feature_hash_dict(args.feature_hash_dict)
    FEATURE_LENTH_DICT = load_feature_lenth_dict(args.feature_lenth_dict)
    
    
    spark = SparkSession.builder\
            .appName("FeatureProcessing") \
            .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console -Dlog4j.logger.org.apache.spark=ERROR -Dlog4j.logger.org.apache.hadoop=ERROR") \
            .config("spark.executor.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console -Dlog4j.logger.org.apache.spark=ERROR -Dlog4j.logger.org.apache.hadoop=ERROR") \
            .getOrCreate()
    sc = spark.sparkContext
    
    FEATURE_HASH_DICT_BROADCAST = sc.broadcast(FEATURE_HASH_DICT)
    FEATURE_LENTH_DICT_BROADCAST = sc.broadcast(FEATURE_LENTH_DICT)

    df = spark.read.format("orc").option("compression", "snappy").load(args.input_data_uri+"/*")
    
    ## 拆分到多个分区
    df = df.repartition(args.num_partitions)
    
    feature_columns = [col for col in df.columns if col.startswith('_')]
    
    df_processed = df.rdd.mapPartitions(
        lambda iterator: process_partition(iterator, feature_columns, FEATURE_HASH_DICT_BROADCAST,FEATURE_LENTH_DICT_BROADCAST,args.padding_lenth)
    ).toDF()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_data_uri}_{timestamp}"

    #you can repartition to merge small files into bigger ones , to get more optimal performance for consequnced training job
    df_processed.repartition(15000).write.format("tfrecord").option("recordType", "Example").save(output_path)
    #df_processed.write.format("tfrecord").option("recordType", "Example").save(output_path)
    
    spark.stop()

if __name__ == "__main__":
    main()
