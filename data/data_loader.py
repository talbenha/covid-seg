import tensorflow as tf
from data.data_handler_func import data_loader_train_test


def load_data(config, train_flag=True):
    data = data_loader_train_test(config, train_flag=train_flag)
    return data

def transform(data, config):
    def transform_example(image,label):
        image, label = tf.cast(image, tf.float32), tf.cast(label, tf.int64)
        image = tf.divide(image, 255.)
        return image, label

    data['train'] = data['train'].map(transform_example)
    data['test'] = data['test'].map(transform_example)

    return data

def make_iterators(data, config):
    def augment_example(image, label):
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        return image,label

    train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_eval_iter = data['train'].batch(config.batch_size_eval).take(-1)
    test_iter = data['test'].batch(config.batch_size_eval).take(-1)

    iterators = {'train': train_iter,
                 'train_eval': train_eval_iter,
                 'test': test_iter}
    return iterators
