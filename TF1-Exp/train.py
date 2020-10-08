import tensorflow as tf
from tensorflow import logging as tflogging

from configs.path_configs import TRAIN_DATA_PATH, MODEL_CKPT_DIR_PATH

flags = tf.app.flags

flags.DEFINE_string('train_data_path', TRAIN_DATA_PATH, 'Path to training data')
flags.DEFINE_string('model_ckpt_dir_path', MODEL_CKPT_DIR_PATH, 'Path to directory that saves model checkpoints')

FLAGS = flags.FLAGS


def build_graph():
    pass


def get_meta_filename():
    pass


def build_model():
    pass


def train():
    tflogging.info(FLAGS.train_data_path)


def main(a):
    pass


if __name__ == '__main__':
    tf.app.run()
