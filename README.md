# tensorflow_varlen_feature_training

<p align="left">
    &nbsp中文&nbsp ｜ <a href="README_EN.md">English</a>&nbsp 
</p>
<br>

## 基本介绍

本示例是基于客户推荐业务场景，rerank模型的变长特征数据开发的tensorflow 分布式训练示例代码
该代码包括如下部分：
* 原始dummy数据的特征处理，其中包括将变长特征截断且padding为固定特征长度的特征处理，和不用padding，保持原有变长特征格式的特征处理
* 将变长特征截断且padding为固定长度的tensorflow多机分布式训练脚本（CPU实例），以及不用padding，保持变长特征格式的tensoflow多机分布式训练（CPU实例）


<br>

## 训练模型网络结构说明
- 详见train/DNN-fixlen-feature-padding.py中的 SimpleChannel 和 SimpleEmbedding，具体如下：

1. SimpleChannel层:
  - 使用了一个简单的全连接网络（两个Dense层）
  - 第一个Dense层使用ReLU激活函数。 
  - 返回四个值，但s1, s2, s3只是简单的占位符，没有实际的计算意义。

2. SimpleEmbedding 层:
  - 简单的embedding dense 示例，只包含一个 Dense 层用于嵌入处理

## 代码结构

  - 如下图所示 ，变长特征dummy示例数据在raw/下，特征处理代码在processing job/下, tensorflow sagemaker 模型训练代码在train/目录下,
    ```shell
    .
    ├── raw                                                # 原始dummy数据目录
    │   ├── test_dummpy_001.snappy.orc                       
    ├── feature_process                                    # 特征处理目录
    │   ├── feature_processing_sagemaker.ipynb               # sagemaker processing job 特征处理脚本（包括padding和不padding的处理逻辑）
    │   └── feature_spark_padding.py                      # padding为定长的特征处理逻辑
    │   └── feature_spark_without_padding.py                 # 不padding为定长的特征处理逻辑
    ├── train                                              # tensoflow模型训练目录
    │   └── DNN-fixlen-feature-padding.py                    # 定长特征的模型训练
    │   └── DNN-varlen-feature-raggedtensor.py               # 变长特征的模型训练
    │   └── tensorflow_script_mode_training.ipynb            # sagemaker 多机CPU实例 TF PS strategy 分布式训练
    ```

<br>

## 常见问题

**Q1:** 为什么要将可变长特征截断并padding到固定长度特征

**A1:**  使用TF的feature column API的时候，需要把特征变成定长。对于结构化特征的建模，序列特征很长不一定模型效果就好，因此这里把序列特征截断到一定长度（比如10）来做尝试。
具体涉及到的API如下：
   - tf.io.FixedLenFeature padding为定长特征列
   - tf.feature_column.categorical_column_with_hash_bucket 定长特征列bucket hash编码
   - tf.feature_column.weighted_categorical_column 定长带权重特征列编码



**Q2:** 如何Padding特征到固定长度，长度值如何确定？

**A2:** 定长padding逻辑如下：
   - 每一列特征在原始数据中，需要统计所有数据的唯一值长度，以及最大序列长度
   - 注意这里的唯一值长度和最大序列长度一个是所有的值，一个是最大的长度序列）
   - 对数据中的每一行的每一个特征，如果没有达到最大序列长度，从所有唯一值中依次查找没有出现过的值进行padding（目的是找一个没有意义的特征值作为padding的值），拼接到原来的值列表数组中。特征对应的weight序列也需要变成同样固定长度的，weight序列的padding值就可以使用类似很小的浮点数（比如1e-7）.


**Q3:** 为什么定长特征的方案中用的是TF的embedding column API而不是shared embedding column API？

**A3:** 当使用tf.feature_column API + tf.keras API，且使用TF2.14 on Sagemaker的方案，尝试使用shared embedding column API不能work。因此这里就每个特征使用单独的embedding table的方案。


**Q4:** 不padding为固定长度特征训练数据，tensorflow的训练如何支持？

**A4:** 目前业界不定长特征列在tensorflow中并没有官方通用的方法，本sample示例中给出了一种使用RaggedTensor表示shape不规整的tensor的方法及对应的tensorflow训练代码，具体如下：
   - 封装DenseToRaggedLayer(tf.keras.layers.Layer) 把dense Tensor转为RaggedTensor
   - 使用tf.keras.layers.Hashing以便支持RaggedTensor 
   - 使用tf.keras.layers.CategoryEncoding以便支持RaggedTensor


**Q5:** 不padding为固定长度特征训练数据，如何实现带有权重的share embedding？

**A5:** trick如下：
   - 在tf.keras中，tf.keras.layers.Embedding 没有 combiner 选项，但可以使用 tf.keras.layers.Dense 实现相同的效果（参考 https://tensorflow.google.cn/guide/migrate/migrating_feature_columns?hl=zh-cn）。
   - 在tf.keras中，tf.keras没有直接实现share embedding的layer，直接用同一个layer针对不同的input调用就间接实现了share.
   - 在tf.keras中，使用preprocessing API的时候，使用CategoryEncoding API的参数count_weights把权重带入，注意count_weights是支持变长权重的。

<br>

## 关键实现及优化
* Processing Job 特征padding填充并转TF Record
  - pyspark DataFrame分布式处理
  - 同时传入每个特征列的唯一值长度字典（group-id unique hash dict）和在训练数据中最大序列长度字典（group-id max length hash dict） 
  - group_id 按最大序列长度字典中的对应值截断（在本sample示例中group_id 按唯一值padding到最长10计算）
  - weight值处理（TF使用weight feature column的时候，设置的weight不能是0或者0.0（会报错），因此对于填充过的变成list作为特征的时候，为了在后面做embedding的时候把这个填充的id mask掉，这里可以考虑设置对应填充位置的weight为很小的浮点数比如0.000001）
  - uint64在TF feature中不支持list类型， 转为bytelist（string）

* Processing Job统计group-id特征列长度
  - pyspark DataFrame分布式处理
  - Processing job剔重group-id并构造dict字典
  - withColumn-》MapPartition优化
  - 小文件合并 （小文件合并为大文件进行TF ParameterServer Strategy分布式训练，经过实验验证提升速度很明显）
 

* TensorFlow ParameterServer stratey 分布式训练
  - lustre fsx Sagemaker training Job挂载（使用TensorFlow ParameterServer stratey训练时，checkpoint保存需要一个share的文件系统）
  - 使用变长序列（不截断，不padding）+ 所有特征share同一个embedding table
  - 训练数据使用Sagemaker的Fastfile mode方式进行流式训练
  - tf.data.TFRecordDataset数据shard（在TensorFlow ParameterServer stratey训练时，使用tf.distribute.coordinator.experimental_get_current_worker_index API配合tf.dataset.shard来做）
  - SageMaker tensorflow PS strategy 多机CPU实例cluster（chief，ps，worker 节点构建）

  

## 原始数据字段说明及变长特征处理逻辑

* raw原始特征字段说明如下
一共324特征列，一个label字段列 
如果某一列是单值特征，字符串通过\003可分成两部分，第⼀部分是根据明⽂特征计算出的64位⽆符号整型哈希值；第⼆部分是哈希值对应的weight 
如果某一列是多值特征，字符串通过\001分成多个单值特征，每个单值特征的构成与上述描述⼀致 
label字段列，共有两个取值：0，代表负样本 1，代表正样本 

* 特征示例（多个二级分类）
特征列名1（单值特征）：_11001
值：12249734119335549443\0030.6931471
特征列名2（多值特征）：_12001
值：13237249145391275771\0030.693147\00118246459961346429377\0030.693147
...省略
label列名: label
值：0

* 特征处理规则
保持特征列名不变
传入一个dict，其中key为每个特征列名，value为该特征列名对应的所有整型哈希值的list列表
把多指特征列的值根据该特征列所有的整型哈希值进行填充，缺失的整型哈希值的权重weight填充为很小的一个值（比如1e-7），分隔符号仍然为\003
大量原始特征数据的情况下，processing job inputs对目录下文件做sharding方式 


<br>
