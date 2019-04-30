import tensorflow as tf
import numpy as np
"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


class Alexnet:
    """
    Building TF Graph & Session for Classification Model, Alexnet

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

            tf.add_to_collection('inputs', x)
            tf.add_to_collection('inputs', labels)
            tf.add_to_collection('inputs', is_train)
            tf.add_to_collection('inputs', lr)


            with tf.variable_scope('C1'):
                c1 = tf.layers.Conv2D(24, (11, 11), strides=4, padding='SAME',
                                      activation=tf.nn.relu)(x)
                c1 = tf.nn.local_response_normalization(c1,
                                                        depth_radius=2,
                                                        bias=1,
                                                        alpha=2e-5,
                                                        beta=0.75,
                                                        name='LRN')

            s2 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='S2')(c1)

            with tf.variable_scope('C3'):
                c3 = tf.layers.Conv2D(64, (5, 5), strides=1, padding='SAME',
                                      activation=tf.nn.relu)(s2)
                c3 = tf.nn.local_response_normalization(c3,
                                                        depth_radius=2,
                                                        bias=1,
                                                        alpha=2e-5,
                                                        beta=0.75,
                                                        name='LRN')

            s4 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='S4')(c3)

            c5 = tf.layers.Conv2D(96, (3, 3), strides=1, padding='SAME',
                                  activation=tf.nn.relu, name='C5')(s4)
            c6 = tf.layers.Conv2D(96, (3, 3), strides=1, padding='SAME',
                                  activation=tf.nn.relu, name='C6')(c5)
            c7 = tf.layers.Conv2D(64, (3, 3), strides=1, padding='SAME',
                                  activation=tf.nn.relu, name='C7')(c6)

            with tf.variable_scope('F8'):
                c7 = tf.layers.Flatten()(c7)
                f8 = tf.layers.Dense(64, activation=tf.nn.relu)(c7)
                f8 = tf.layers.Dropout(rate=0.5)(f8, training=is_train)

            with tf.variable_scope('F9'):
                f9 = tf.layers.Dense(64, activation=tf.nn.relu)(f8)
                f9 = tf.layers.Dropout(rate=0.5)(f9, training=is_train)

            with tf.variable_scope('OUTPUT'):
                logits = tf.layers.Dense(self._num_classes)(f9)

            logits = tf.identity(logits, name='logits')
            y_pred = tf.nn.softmax(logits, name='prediction')

            loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

            train_op = tf.train.AdamOptimizer(lr).minimize(loss)

            with tf.variable_scope('metrics'):
                top_5 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=5), tf.float32))
                top_1 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32))

            top_5 = tf.identity(top_5, name='top_5_accuracy')
            top_1 = tf.identity(top_1, name='top_1_accuracy')

        return self