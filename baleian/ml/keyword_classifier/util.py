import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.util import event_pb2


def load_data(path):
    df = pd.read_parquet(path)
    features, labels = df['feature'], df['label']
    features = np.array(list(features.values)).astype('int32')
    labels = np.array(list(labels.values)).astype('int32')
    return features, labels


def list_filenames(path):
    return glob.glob(f'{path}/*')


def get_tensorboard_summaries(path):
    history = []
    for summary_path in list_filenames(path):
        for event in tf.data.TFRecordDataset(summary_path):
            e = event_pb2.Event.FromString(event.numpy())
            for v in e.summary.value:
                history.append({'time': e.wall_time, 'step': e.step, 'tag': v.tag, 'value': v.simple_value})
    return sorted(history, key=lambda x: x['time'])
