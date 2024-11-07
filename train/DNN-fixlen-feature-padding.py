import os
import json
import time
import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from sagemaker_tensorflow import PipeModeDataset
import boto3


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#os.environ['TF_GRPC_TIMEOUT_SEC'] = '1800'  # or a higher value

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


#First, read the maximum sequence length for each group ID
import pandas as pd

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

## Read the JSON result file of sequence lengths after feature processing and store it in a dictionary
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

seq_lenth_json = os.environ['feature_lenth_dict']
feature_hash_json = os.environ['feature_hash_dict']

sequence_length = load_feature_lenth_dict(seq_lenth_json)
feature_hash = load_feature_hash_dict(feature_hash_json)



#Force disable GPU devices (use CPU for training)
os.environ["CUDA_VISIBLE_DEVICES"] = ''


#Set the parallelism for TF intra-op and inter-op, as well as the relevant environment variables for MKLDNN.
# Adjusting these parameters can have a significant impact on training speed.
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

#Here, set the parallelism and the number of MKLDNN threads to half of the vCPU count (which is equivalent to the number of physical cores).
number_CPU = int(int(os.environ.get('SM_NUM_CPUS')) / 2)
tf.config.threading.set_intra_op_parallelism_threads(number_CPU)
tf.config.threading.set_inter_op_parallelism_threads(number_CPU)

os.environ["KMP_AFFINITY"]= "verbose,disabled"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,scatter,1,0"
os.environ['OMP_NUM_THREADS'] = str(number_CPU)
os.environ['KMP_SETTINGS'] = '1'


#Each feature name is represented as id_xxx, and each feature corresponds to weighted_id_xxx to represent the weight of that feature or feature sequence
#Feature sequences are all converted to fixed length, padded in advance using a separate task before the training task.
#Now we proceed to build the feature description
feature_description = {
  'target': tf.io.FixedLenFeature([], tf.int64, default_value=0),   #label
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
      'id__{}'.format(colname): tf.io.FixedLenFeature([min(sequence_length['_{}'.format(colname)],10)], tf.string)
          for colname in range(start, end+1)
      })

    feature_description.update({
      'weighted_id__{}'.format(colname): tf.io.FixedLenFeature([min(sequence_length['_{}'.format(colname)],10)], tf.float32)
          for colname in range(start, end+1)
  })


print(feature_description.keys())
# Do not perform a second hash on the feature value represented by the current hash value. If using tf.feature_column.categorical_column_with_identity for feature processing,
# you would need to use the maximum value of this hash plus 1 as num_buckets. However, since this hash value is 64-bit fixed length, using tf.feature_column.categorical_column_with_identity
# for feature mapping is not appropriate as it would waste too many buckets. Therefore, we perform a second hash on this feature value represented by the hash value, using tf.feature_column.categorical_column_with_hash_bucket for feature mapping.
# For the hash bucket size, consider the number of unique values/cardinality of the feature. If the cardinality is not large, for example, a few thousand, then the hash bucket size can be 3 to 5 times the cardinality of the discrete feature;
# If the cardinality is very large, for example, tens of millions, then the choice of hash bucket size can start with the fourth root of the cardinality of this ID feature in the training set as an initial attempt.

hash_bucket_size = {}
TEN_MILLION = 10000000

# Iterate through each key-value pair in feature_hash_dict
for key, value in feature_hash.items():
    new_key = key
    # Get the length of the value (array)
    length = len(value)
    # Determine the value of hash_bucket_size based on the length
    if length <= TEN_MILLION:
        hash_bucket_size[new_key] = length*3
    else:
        # If it exceeds 10 million, take the fourth root of the value
        hash_bucket_size[new_key] = int(math.pow(length, 0.25))

tf.print("hash_bucket_size:")
for key, value in hash_bucket_size.items():
    tf.print(f"{key}: {value}")

sparse_col = {
  colname:  tf.feature_column.categorical_column_with_hash_bucket(colname, hash_bucket_size=min(hash_bucket_size[colname.split("id_")[-1]],10), dtype=tf.string)
    for colname in feature_description.keys() if colname.startswith('id__')
}


weighted_sparse_col = {
    'weighted_{}'.format(colname) : tf.feature_column.weighted_categorical_column(col, 'weighted_{}'.format(colname))
      for colname, col in sparse_col.items()
  }


# Each id_xxl is weighted, then embedded, and the pooling method used is sum. Note that the column used below is weighted_sparse_col
embed = {
    'embed_{}'.format(colname) : tf.feature_column.embedding_column(col, 16, combiner='sum')
      for colname, col in weighted_sparse_col.items()
  }
'''
embed = {
    'embed_{}'.format(colname) : tf.feature_column.embedding_column(col, 16, combiner='sum')
      for colname, col in sparse_col.items()
  }
'''

# Here, each feature in feature_description is passed into the input layer
inputs ={
    colname : tf.keras.layers.Input(name=colname, shape=(), dtype='string')
      for colname in feature_description.keys() if colname.startswith('id__')
  }

inputs.update({
    colname : tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
      for colname in feature_description.keys() if colname.startswith('weighted_id__')
  })


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



base_path = '/opt/ml/input/data/training'
train_file_paths = [f'{base_path}{i}' for i in range(1, 8)]

train_file_list = []

for path in train_file_paths:
    if os.path.exists(path):
        train_file_list.extend(get_file_path_list(path))
valid_file_list = train_file_list

# Global batch size equals mini batch per worker * number of workers (excluding chief)
global_batch_size = 2*1024*number_workers

# When using ParameterServer strategy, if you use Model.fit or Model.evaluate APIs, their underlying implementation performs inline (distributed) evaluation.
valid_batch_size = 10*1024*number_workers


def parser(batch_examples):
    tf.print("---enter into parser ---------")
    features = tf.io.parse_example(batch_examples, features=feature_description)
    #tf.print(features)
    labels = features.pop('target')
    return features, labels

def train_dataset_fn(input_context):
    tf.print("---enter into train_dataset_fn ---------")
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    tf.print("---batch size per replica ---------", batch_size)

    batch_size = int(global_batch_size / number_workers)

    # Pipe mode does not work with SageMaker TensorFlow 2.14/2.11/2.9 and ParameterServer strategy
    #dataset = PipeModeDataset('training', record_format='TFRecord')
    dataset = tf.data.TFRecordDataset(train_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    tf.print("---num_input_pipelines-----", input_context.num_input_pipelines)
    tf.print("---input_pipeline_id-----", input_context.input_pipeline_id)
    ## For TensorFlow 2.X ParameterServer strategy, the sharding below essentially has no effect.
    # As stated in the official documentation, each worker receives the same data here.
    # However, due to the presence of shuffling below, in each step, each worker actually receives different batch data.
    #dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

    #new api for shard in parameterserver strategy.
    worker_index = tf.distribute.coordinator.experimental_get_current_worker_index()
    tf.print("---worker_index-----", worker_index)
    dataset = dataset.shard(num_shards=number_workers, index=worker_index)

    #just for test , in prod , it should be the same amount of the train epochs
    dataset = dataset.repeat(8)
    dataset = dataset.shuffle(batch_size * 128, reshuffle_each_iteration=True)
    tf.print("---shuffle finished-----")

    dataset = dataset.batch(batch_size)
    tf.print("---batch finished-----")
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def gen_valid_dataset(dataset, batch_size):
    dataset = dataset.repeat(10)   #just for test
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.prefetch(1)
    return dataset

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
            #min_shard_bytes=(256 << 10),
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
        feature_count = 323
        embedding_size = 16    # every groupid embedding size is 16
        #feature_dim = feature_count * embedding_size

        # Build a DNN model.
        deep = tf.keras.layers.DenseFeatures(embed.values(), name='deep_inputs')(inputs)
        bn = tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-5,)(deep)
        d = SimpleEmbedding(feature_count, embedding_size)(bn)

        o = tf.keras.layers.Dense(1024)(d)
        zero = tf.zeros([1, 1])
        r, s1, s2, s3 = SimpleChannel(1024, 512)(o, zero, zero, zero)
        r, s1, s2, s3 = SimpleChannel(512, 256)(r, s1, s2, s3)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(r)

        model = tf.keras.Model(inputs, output)
        learning_rate = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            #decay_steps=int((407658153*7)/global_batch_size),
            decay_rate=0.96,
            staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0) 
        model.compile(
                    #optimizer='adam',
                    optimizer= opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'AUC'])
        #model.compile(optimizer='adam',
        #            loss='binary_crossentropy',
        #            metrics=['accuracy', 'AUC'])
        
        


        train_dataset_creator = tf.keras.utils.experimental.DatasetCreator(train_dataset_fn)
        print("create train dataset finished")
        
        #valid_dataset = tf.data.TFRecordDataset(valid_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        #valid_dataset = gen_valid_dataset(valid_dataset, valid_batch_size)
        
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
                      #validation_steps = int(407658153/valid_batch_size),
                      epochs=1,
                      steps_per_epoch = int((407658153*7)/global_batch_size),
                      #class_weight=class_weight,
                      verbose=2,
                      callbacks=[checkpoint_callback]
                      )

        print("Final loss = {}, accuracy = {}, AUC = {}, val_loss = {}, val_accuracy={}, val_AUC = {}".format(
    history.history['loss'][-1], history.history['accuracy'][-1], history.history['auc'][-1],0,0,0))
    #history.history['val_loss'][-1], history.history['val_accuracy'][-1], history.history['val_auc'][-1]))

    #model_dir='/opt/ml/model'+'/DNN'
    model_dir = '/opt/ml/input/data/fsx/testforSM/DNN'
    print('Exporting to {}'.format(model_dir))
    model.save(model_dir, save_format="tf")
    print('------chief finish save model------------')

    model_loaded = tf.keras.models.load_model(model_dir)

    #eval_valid_result = model_loaded.evaluate(valid_dataset,  steps = 6,)
    #print('\nvalid evaluate result：{}'.format(eval_valid_result))
    print('------chief finish work------------')

    ## Due to the behavior of TensorFlow ParameterServer strategy where other workers do not exit after the chief worker exits at the end of training,
    ## we manually stop this training job here as a workaround.
    job_name = os.environ['job_name']
    client = boto3.client('sagemaker', region_name='us-west-2')
    client.stop_training_job(TrainingJobName=job_name)

