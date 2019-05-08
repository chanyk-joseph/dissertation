from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())


# Use specific GPU
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 0: 2080ti | 1: 2070
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
# TBA
session.close()