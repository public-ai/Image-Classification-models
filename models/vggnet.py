import tensorflow as tf
import numpy as np
"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


class VGGNet:
    """
    Building TF Graph & Session for Classification Model, VGGNet

    Order

    """

    def __init__(self, num_classes=200):
        """

        """
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self._num_classes = num_classes

    def initialize_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()

            is_not_initialized = self.session.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [
                v for (
                    v,
                    f) in zip(
                    global_vars,
                    is_not_initialized) if not f]

            if len(not_initialized_vars):
                self.session.run(
                    tf.variables_initializer(not_initialized_vars))

    def build_network(self):
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, (None, 64, 64, 3), name='X')
            labels = tf.placeholder(tf.float32,
                                          shape=(None,),
                                          name='labels')
            is_train = tf.placeholder_with_default(False, (), name='is_train')
            lr = tf.placeholder_with_default(0.001, (),
                                                   name='learning_rate')


