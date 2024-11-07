# tensorflow_varlen_feature_training

Here's the English translation:

<p align="left">
    &nbsp;English&nbsp; | <a href="README.md">中文</a>&nbsp;
</p>
<br>

## Basic Introduction

This example is a TensorFlow distributed training sample code developed for variable-length feature data of rerank models based on customer recommendation business scenarios. The code includes the following parts:
* Feature processing of original dummy data, including feature processing that truncates variable-length features and pads them to a fixed feature length, and feature processing that maintains the original variable-length feature format without padding
* TensorFlow multi-machine distributed training script (CPU instance) for variable-length features truncated and padded to fixed length, as well as TensorFlow multi-machine distributed training (CPU instance) for variable-length features without padding

<br>

## Training Model Network Structure Description
- See SimpleChannel and SimpleEmbedding in train/DNN-fixlen-feature.py for details, as follows:

1. SimpleChannel layer:
- Uses a simple fully connected network (two Dense layers)
- The first Dense layer uses ReLU activation function
- Returns four values, but s1, s2, s3 are just simple placeholders with no actual computational significance

2. SimpleEmbedding layer:
- A simple embedding dense example, containing only one Dense layer for embedding processing

## Code Structure

- As shown in the following diagram, the variable-length feature dummy sample data is in raw/, the feature processing code is in processing job/, and the TensorFlow SageMaker model training code is in the train/ directory,
  ```shell
  .
  ├── raw                                                # Original dummy data directory
  │   ├── test_dummpy_001.snappy.orc                       
  ├── feature_process                                    # Feature processing directory
  │   ├── feature_processing_sagemaker.ipynb               # SageMaker processing job feature processing script (including padding and non-padding processing logic)
  │   └── feature_spark_padding.py                         # Feature processing logic for padding to fixed length
  │   └── feature_spark_without_padding.py                 # Feature processing logic without padding to fixed length
  ├── train                                              # TensorFlow model training directory
  │   └── DNN-fixlen-feature-padding.py                    # Model training for fixed-length features
  │   └── DNN-varlen-feature-raggedtensor.py               # Model training for variable-length features
  │   └── tensorflow_script_mode_training.ipynb            # SageMaker multi-machine CPU instance TF PS strategy distributed training
  ```

<br>

Here's the English translation of the FAQ section:

## Frequently Asked Questions

**Q1:** Why truncate variable-length features and pad them to fixed-length features?

**A1:** When using TF's feature column API, features need to be converted to fixed length. For structured feature modeling, very long sequence features don't necessarily lead to better model performance, so here we try truncating sequence features to a certain length (e.g., 10).
Specific APIs involved are:
- tf.io.FixedLenFeature for fixed-length feature columns
- tf.feature_column.categorical_column_with_hash_bucket for fixed-length feature column bucket hash encoding
- tf.feature_column.weighted_categorical_column for fixed-length weighted feature column encoding

**Q2:** How to pad features to a fixed length, and how to determine the length value?

**A2:** The fixed-length padding logic is as follows:
- For each feature column in the original data, calculate the unique value length and maximum sequence length across all data
- Note that the unique value length and maximum sequence length refer to all values and the longest sequence, respectively
- For each feature in each row of data, if it hasn't reached the maximum sequence length, sequentially search for unused values from all unique values for padding (to find a meaningless feature value for padding), and append them to the original value list array. The corresponding weight sequence also needs to be padded to the same fixed length, using a very small float (e.g., 1e-7) as the padding value for the weight sequence.

**Q3:** Why use TF's embedding column API instead of shared embedding column API in the fixed-length feature approach?

**A3:** When using tf.feature_column API + tf.keras API with TF2.14 on SageMaker, attempts to use the shared embedding column API didn't work. Therefore, we use a separate embedding table for each feature.

**Q4:** How does TensorFlow support training data without padding to fixed-length features?

**A4:** Currently, there's no official universal method for variable-length feature columns in TensorFlow. This sample demonstrates a method using RaggedTensor to represent tensors with irregular shapes and the corresponding TensorFlow training code, specifically:
- Encapsulate DenseToRaggedLayer(tf.keras.layers.Layer) to convert dense Tensor to RaggedTensor
- Use tf.keras.layers.Hashing to support RaggedTensor
- Use tf.keras.layers.CategoryEncoding to support RaggedTensor

**Q5:** How to implement weighted shared embedding for training data without padding to fixed-length features?

**A5:** The tricks are as follows:
- In tf.keras, tf.keras.layers.Embedding doesn't have a combiner option, but you can use tf.keras.layers.Dense to achieve the same effect (refer to https://tensorflow.google.cn/guide/migrate/migrating_feature_columns?hl=zh-cn).
- In tf.keras, there's no direct implementation of a shared embedding layer, but you can indirectly achieve sharing by calling the same layer for different inputs.
- When using the preprocessing API in tf.keras, use the count_weights parameter of the CategoryEncoding API to incorporate weights. Note that count_weights supports variable-length weights.

<br>

Here's the English translation of the "Key Implementations and Optimizations" and "Original Data Field Description and Variable-length Feature Processing Logic" sections:

## Key Implementations and Optimizations
* Processing Job for feature padding and conversion to TF Record
  - Distributed processing using PySpark DataFrame
  - Simultaneously pass in unique value length dictionary (group-id unique hash dict) and maximum sequence length dictionary (group-id max length hash dict) for each feature column
  - Truncate group_id according to the corresponding value in the maximum sequence length dictionary (in this sample, group_id is padded to a maximum of 10 based on unique values)
  - Weight value processing (When using weight feature column in TF, the set weight cannot be 0 or 0.0 (will cause an error). Therefore, when converting padded features to lists, to mask the padded ids during embedding, consider setting the weight of corresponding padded positions to a very small float, e.g., 0.000001)
  - uint64 is not supported as list type in TF feature, convert to bytelist (string)

* Processing Job to calculate group-id feature column length
  - Distributed processing using PySpark DataFrame
  - Deduplicate group-id and construct dictionary in Processing job
  - Optimization using withColumn -> MapPartition
  - Small file merging (Merging small files into large files for TF ParameterServer Strategy distributed training, experimentally verified to significantly improve speed)

* TensorFlow ParameterServer strategy distributed training
  - Lustre FSx Sagemaker training Job mounting (a shared file system is needed for checkpoint saving when using TensorFlow ParameterServer strategy training)
  - Use variable-length sequences (no truncation, no padding) + all features share the same embedding table
  - Use Sagemaker's Fastfile mode for streaming training data
  - tf.data.TFRecordDataset data sharding (use tf.distribute.coordinator.experimental_get_current_worker_index API with tf.dataset.shard when training with TensorFlow ParameterServer strategy)
  - SageMaker tensorflow PS strategy multi-machine CPU instance cluster (chief, ps, worker node construction)

## Original Data Field Description and Variable-length Feature Processing Logic

* Description of raw original feature fields
  A total of 324 feature columns and one label field column
  If a column is a single-value feature, the string can be split into two parts by \003, the first part is a 64-bit unsigned integer hash value calculated based on the plaintext feature; the second part is the weight corresponding to the hash value
  If a column is a multi-value feature, the string is split into multiple single-value features by \001, and each single-value feature is constructed as described above
  Label field column has two values: 0 represents negative sample, 1 represents positive sample

* Feature example (multiple secondary classifications)
  Feature column name 1 (single-value feature): _11001
  Value: 12249734119335549443\0030.6931471
  Feature column name 2 (multi-value feature): _12001
  Value: 13237249145391275771\0030.693147\00118246459961346429377\0030.693147
  ...omitted
  Label column name: label
  Value: 0

* Feature processing rules
  Keep feature column names unchanged
  Pass in a dict where keys are feature column names and values are lists of all integer hash values corresponding to that feature column name
  Pad multi-value feature column values based on all integer hash values for that feature column, fill missing integer hash value weights with a very small value (e.g., 1e-7), keep \003 as the separator
  For large amounts of raw feature data, use sharding method for processing job inputs on files in the directory

<br>