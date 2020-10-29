import tensorflow as tf
import collections


def create_context(config):
    def ctx(): pass

    ctx.spm = config.spm_path
    return ctx


def create_example(par, config, context):
    par = lemmatize(par)

    features = collections.OrderedDict()
    features["text"] = par

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def lemmatize(par):
    return par