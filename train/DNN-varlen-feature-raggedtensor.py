import os
import json
import time
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from sagemaker_tensorflow import PipeModeDataset
import boto3
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

#re-configure the TF_CONFIG to meet with ParameterServerStrategy
tf_config = json.loads(os.environ['TF_CONFIG'])
if tf_config.get('cluster'):
    NUM_PS = len(tf_config.get('cluster').get('ps'))
    if tf_config['cluster'].get('master'):
        tf_config['cluster']['chief'] = tf_config['cluster'].get('master')
        del tf_config['cluster']['master']
    # change master task to chief task
    if tf_config.get('task'):
        if tf_config['task']['type'] == 'master':
            tf_config['task']['type'] = 'chief'
print('updated tf_config:', tf_config)
os.environ["TF_CONFIG"] = json.dumps(tf_config)


#Sagemaker will start a parameter server on each training instance
number_workers = NUM_PS - 1    #exclude chief
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print(cluster_resolver)
print("---task_type ---------", cluster_resolver.task_type)
print("---task id ---------", cluster_resolver.task_id)




## Set the parallelism for TF intra-op and inter-op, as well as MKLDNN-related environment variables.
## Tuning these parameters can have a significant impact on training speed.
## Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# Set the parallelism and MKLDNN thread count to half of the vCPU count (which is equivalent to the number of physical cores)
number_CPU = int(int(os.environ.get('SM_NUM_CPUS')) / 2)
tf.config.threading.set_intra_op_parallelism_threads(number_CPU)
tf.config.threading.set_inter_op_parallelism_threads(number_CPU)


## Whether to enable MKLDNN for training is an optional choice.
## For some customer‘s project, it was found that enabling MKLDNN actually slowed down the training speed.

os.environ["KMP_AFFINITY"]= "verbose,disabled"
os.environ['OMP_NUM_THREADS'] = str(number_CPU)
os.environ['KMP_SETTINGS'] = '1'

    
# Each feature name is represented as id_xxx, and each feature corresponds to a weighted_id_xxx to represent the weight of that feature or feature sequence
# All feature sequences are converted to fixed length, padded in advance using a separate task before the training task
# Now we proceed to construct the feature description
feature_description = {
  'target': tf.io.FixedLenFeature([], tf.int64),   #label
}

#These feature IDs were created from dummy data
feature_cl = [[11001,11004],           #groupid 11000~11099 (User behavior: 4)
              [11007,11008],           #groupid 11000~11099 (User behavior: 2)
              [11021,11024],           #groupid 11000~11099 (User behavior: 4)
              [11041,11046],           #groupid 11000~11099 (User behavior: 6)
              [11061,11066],           #groupid 11000~11099 (User behavior: 6)
              [11081,11086],           #groupid 11000~11099 (User behavior: 6)
              [11601,11603],           #groupid xxxx (User behavior: 3)
              [12001,12006],           #groupid 12000~12159 (User behavior: 6)
              [20001,20003],           #groupid 20000~20099 (Device information: 3)
              [20101,20102],           #groupid 20100~20199 (Device information: 2)
              [20201,20210],           #groupid 20200~20299 (Device information: 10)
              [30001,30006],           #groupid xxxx (User behavior: 6)
              [30201,30207],           #groupid 30200~30299 (Supply side information: 7)
              [40001,40005],           #groupid xxxx (User behavior: 5)
              [40201,40215],           #groupid 40200~40229 (Material/template related information: 15)
              [40231,40231],           #groupid 40230~40259 (Creative related information: 1)
              [40301,40307],           #groupid 40300~40399 (Demand side information: 7)
              [40321,40324],           #groupid 40300~40399 (Demand side information: 4)
              [50801,50802],           #groupid 50800~50819 (User characteristics: 2)
              [50805,50807],           #groupid 50800~50819 (User characteristics: 3)
              [50810,50810],           #groupid 50800~50819 (User characteristics: 1)
              [51001,51006],           #groupid 51000~52999 (Combined features of demand side and user behavior: 6)
              [51011,51014],           #groupid 51000~52999 (Combined features of demand side and user behavior: 4)
              [51021,51026],           #groupid 51000~52999 (Combined features of demand side and user behavior: 6)
              [51031,51036],           #groupid 51000~52999 (Combined features of demand side and user behavior: 6)
              [51041,51046],           #groupid 51000~52999 (Combined features of demand side and user behavior: 6)
              [52001,52005],           #groupid 51000~52999 (Combined features of demand side and user behavior: 5)
              [61101,61103],           #groupid 61100~61199 (User behavior: 3)
              [70001,70011],           #groupid 70000~79999 (Combined features of demand side and supply side: 11)
              [70031,70041],           #groupid 70000~79999 (Combined features of demand side and supply side: 11)
              [70301,70306],           #groupid 70000~79999 (Combined features of demand side and supply side: 6)
              [70601,70606],           #groupid 70000~79999 (Combined features of demand side and supply side: 6)
              [70901,70906],           #groupid 70000~79999 (Combined features of demand side and supply side: 6)
              [72401,72402],           #groupid 70000~79999 (Combined features of demand side and supply side: 2)
              [80001,80115],           #groupid 80001~80151 (UPS features: 115)
              [80125,80151],           #groupid 80001~80151 (UPS features: 27)
              ]


for start, end in feature_cl:
    feature_description.update({
      'id__{}'.format(colname): tf.io.VarLenFeature(tf.string)
          for colname in range(start, end+1) 
      })

    feature_description.update({
      'weighted_id__{}'.format(colname): tf.io.VarLenFeature(tf.float32)
          for colname in range(start, end+1) 
  })


print(feature_description.keys())
# no secondary hashing has been applied to the feature values currently represented by hash values. If tf.feature_column.categorical_column_with_identity is used for feature processing,
# the maximum hash value plus 1 should be used as num_buckets. However, since this hash value has a fixed length of 64 bits, using tf.feature_column.categorical_column_with_identity
# for feature mapping is not appropriate as it would waste too many buckets. Therefore, we will perform a secondary hash on these hash-represented feature values and use tf.feature_column.categorical_column_with_hash_bucket for feature mapping.
# For the hash bucket size, we'll consider the number of unique values/cardinality of the feature. If the cardinality is not large, say a few thousand, then the hash bucket size can be 3 to 5 times the cardinality of the discrete feature;
# If the cardinality is very large, say tens of millions, then a starting point for choosing the hash bucket size could be the fourth root of the cardinality of this ID feature in the training set.

# in currently sample code， all group IDs share the same embedding table.
# For different projects, the decision on whether different features should share an embedding table can be based on the physical meaning of the features.

share_embedding_sparse_config = [
  {'name': 'share_emb_1',
   'columns': {    
       colname : {}
          for colname in feature_description.keys() if colname.startswith('id_')},
   'bucket_size': 50000000, #5000w + 4K mini batch per work can work on r5.4xlarge.
   'embedding_size': 16},
]


# Next is the network structure
## A simple fully connected network is used (two Dense layers), with the first Dense layer using ReLU activation function.
## Four values are returned, but s1, s2, s3 are just simple placeholders without actual computational significance.
class SimpleChannel(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.fl = int(in_size / 4)
        self.out_size = out_size

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.out_size, activation='relu')
        super().build(input_shape)

    def call(self, inputs, i1, i2, i3):
        x = self.dense(inputs)
        s1 = tf.reduce_sum(x[:, :self.out_size//3], axis=1, keepdims=True) + i1
        s2 = tf.reduce_sum(x[:, self.out_size//3:2*self.out_size//3], axis=1, keepdims=True) + i2
        s3 = tf.reduce_sum(x[:, 2*self.out_size//3:], axis=1, keepdims=True) + i3

        return x, s1, s2, s3

# Simple embedding dense example, containing only one Dense layer for embedding processing
class SimpleEmbedding(tf.keras.layers.Layer):
    def __init__(self, out_size, emb_size):
        super().__init__()
        self.out_size = out_size
        self.emb_size = emb_size

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.out_size * self.emb_size, activation='relu')
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)
        
def _get_file_path_v2(root_path):
    file_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def get_file_path_list(root_path):
    file_list = []
    file_list=_get_file_path_v2(root_path)
    return file_list

#file mode and fast file mode
train_file_path = '/opt/ml/input/data/training'
train_file_list = get_file_path_list(train_file_path)
valid_file_list = train_file_list


#The global batch size is equal to the mini-batch size of each worker multiplied by the number of workers (excluding the chief).
global_batch_size = 4*1024*number_workers

# When using the parameter server strategy, if you use the Model.fit and Model.evaluate APIs, their underlying implementation performs inline (distributed) evaluation.
valid_batch_size = 4*1024*number_workers 


def parse_record_batch(message):
    parsed_feature_dict = tf.io.parse_example(message, feature_description)
    label = parsed_feature_dict.pop('target')
    for feature_name, cur_tensor in parsed_feature_dict.items():
        if feature_name.startswith('id'):
            # Padding the ID feature with a meaningless value '-1'
            parsed_feature_dict[feature_name] = tf.sparse.to_dense(cur_tensor, '-1')
        if feature_name.startswith('weighted'):
            # Set the weight for the padding of ID features to '0',
            # thereby eliminating the influence of the padded ID feature values
            parsed_feature_dict[feature_name] = tf.sparse.to_dense(cur_tensor, 0)

    return parsed_feature_dict, label


def train_dataset_fn(input_context):
    #print("---enter into train_dataset_fn ---------")
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    #tf.print("---batch size per replica ---------", batch_size)
    
    batch_size = int(global_batch_size / number_workers)
    
    #Pipe mode does not work with SageMaker TensorFlow 2.14/2.11/2.9 and ParameterServer strategy
    #dataset = PipeModeDataset('training', record_format='TFRecord')
    dataset = tf.data.TFRecordDataset(train_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    tf.print("---num_input_pipelines-----", input_context.num_input_pipelines)
    tf.print("---input_pipeline_id-----", input_context.input_pipeline_id)
    #For the parameter server strategy in TF 2.X, the sharding below essentially has no effect. As stated in the official documentation, each worker receives the same data here.
    #However, due to the presence of shuffling below, in reality, each worker receives different batch data for each step.
    #dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    
    #new api for shard in parameterserver strategy.
    worker_index = tf.distribute.coordinator.experimental_get_current_worker_index()
    tf.print("---worker_index-----", worker_index)
    dataset = dataset.shard(num_shards=number_workers, index=worker_index)

    #just for test , in prod , it should be the same amount of the train epochs
    dataset = dataset.repeat(100)
    dataset = dataset.shuffle(batch_size * 128, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func=parse_record_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def gen_valid_dataset(dataset, batch_size):
    dataset = dataset.repeat(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func=parse_record_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.prefetch(1)
    return dataset

#raggedlayer能表示shape不规整的tensor
class DenseToRaggedLayer(tf.keras.layers.Layer):
    
    def __init__(self, ignore_value, **kwargs):
        super(DenseToRaggedLayer, self).__init__(**kwargs)
        self.ignore_value = ignore_value

    def call(self, inputs):
        return tf.RaggedTensor.from_tensor(inputs, padding=self.ignore_value)
    
def build_model():
    deep_inputs = []
    deep_raw_inputs = []
 
    # The implementation of weighted shared embedding below has several tricks:
        # 1. In tf.keras, tf.keras.layers.Embedding doesn't have a 'combiner' option, but the same effect can be achieved using tf.keras.layers.Dense (refer to https://tensorflow.org/guide/migrate/migrating_feature_columns).
        # 2. In tf.keras, there's no layer that directly implements shared embedding. Sharing is indirectly achieved by using the same layer for different inputs.
        # 3. When using the preprocessing API in tf.keras, use the 'count_weights' parameter of the CategoryEncoding API to incorporate weights. Note that 'count_weights' supports variable-length weights.
    for conf in share_embedding_sparse_config:
        shared_embedding = tf.keras.layers.Dense(conf['embedding_size'], use_bias=False)
        for feature_name, inner_conf in conf['columns'].items():
            cur_input = keras.Input(shape=(None,), name=feature_name, dtype=tf.string) #id特征
            weight_name = 'weighted_' + feature_name
            weigh_cur_input = keras.Input(shape=(None,), name=weight_name, dtype=tf.float32) #id特征的权重
            # preprocessing.Hashing supports tf.sparse tensor
            # The output_mode of the hashing below should be set to int (index number), otherwise it will occupy a lot of memory due to the large bucket size.
            ragged_hashed_input = tf.keras.layers.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash', output_mode='int',)(DenseToRaggedLayer(name=feature_name + '_rag', ignore_value = '-1')(cur_input))
            
            # To use feature weights, the output_mode parameter needs to be set to 'count'
            encoded_data = tf.keras.layers.CategoryEncoding(num_tokens=conf['bucket_size'], output_mode='count', sparse=True)(ragged_hashed_input, count_weights= DenseToRaggedLayer(name=weight_name + '_rag', ignore_value = 0)(weigh_cur_input))

            shared_emb_data = shared_embedding(encoded_data)
            
                      
            deep_raw_inputs.append(cur_input)
            deep_raw_inputs.append(weigh_cur_input)
            deep_inputs.append(shared_emb_data)
    
    # BUILD MODEL

    deep = layers.Concatenate()(deep_inputs)

    feature_count = 323
    embedding_size = 16

    bn = tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-5,)(deep)
    d = SimpleEmbedding(feature_count, embedding_size)(bn)

    o = tf.keras.layers.Dense(1024)(d)
    zero = tf.zeros([1, 1])
    r, s1, s2, s3 = SimpleChannel(1024, 512)(o, zero, zero, zero)
    r, s1, s2, s3 = SimpleChannel(512, 256)(r, s1, s2, s3)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(r)

    model = tf.keras.Model(deep_raw_inputs, output)
    
    return model



if cluster_resolver.task_type in ('ps', 'worker'):
    logging.info("[{}] Start {}({})...".format(time.time(), cluster_resolver.task_type, cluster_resolver.task_id))
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    server.join()
elif cluster_resolver.task_type == 'chief':
    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 4),
            max_shards=NUM_PS))
    print('after variable_partitioner')
    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner
        )
    print('after strategy')


    with strategy.scope():
        #build网络
        model = build_model()
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'AUC'])


        train_dataset_creator = tf.keras.utils.experimental.DatasetCreator(train_dataset_fn)
        
        valid_dataset = tf.data.TFRecordDataset(valid_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        valid_dataset = gen_valid_dataset(valid_dataset, valid_batch_size)
        
        #Batch-level `Callback`s are not supported with `ParameterServerStrategy`. 
        class print_on_end(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs={}):
                print()

        checkpoint_path = '/opt/ml/input/data/fsx/testforSM/DNN/checkpoints/cpt'
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        history = model.fit(train_dataset_creator,
                      #validation_data=valid_dataset,
                      #validation_steps = 6,
                      epochs=2,
                      steps_per_epoch = 10,
                      #class_weight=class_weight,
                      verbose=2,
                      callbacks=[checkpoint_callback]
                      )

        '''
        print("Final loss = {}, accuracy = {}, AUC = {}, val_loss = {}, val_accuracy={}, val_AUC = {}".format(
    history.history['loss'][-1], history.history['accuracy'][-1], history.history['auc'][-1],
    history.history['val_loss'][-1], history.history['val_accuracy'][-1], history.history['val_auc'][-1]))
        '''
        
    #model_dir='/opt/ml/model'+'/DNN'
    model_dir = '/opt/ml/input/data/fsx/testforSM/DNN'
    print('Exporting to {}'.format(model_dir))
    model.save(model_dir, save_format="tf")
    print('------chief finish save model------------')

    '''
    model_loaded = tf.keras.models.load_model(model_dir)

    eval_valid_result = model_loaded.evaluate(valid_dataset,  steps = 6,)
    print('\nvalid evaluate result：{}'.format(eval_valid_result))
    print('------chief finish work------------')
    '''

    ## Due to the behavior of TensorFlow ParameterServer strategy where other workers do not exit after the chief worker exits at the end of training,
    ## we manually stop this training job here as a workaround.
    job_name = os.environ['job_name']
    client = boto3.client('sagemaker', region_name='us-east-1')
    client.stop_training_job(TrainingJobName=job_name)

