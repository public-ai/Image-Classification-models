import tensorflow as tf
"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


class Alexnet:
    """
    Building TF Graph & Session for Classification Model, Alexnet

    Order
        1) net._initialize_placeholders()
        2) net._attach_inference_network()
        3) net._attach_loss_network()
        4) net._attach_metric_network()
        5) net._attach_summary_network()
        6) net._attach_optimizer_network()

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

    def _initialize_placeholders(self, input_shape=(None, 64, 64, 3)):
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, input_shape, name='X')
            labels = tf.placeholder(tf.int32, (None,), name='labels')

            is_train = tf.placeholder_with_default(False, (), name='is_train')
            lr = tf.placeholder_with_default(0.001, (), name='learning_rate')
        return self

    def _attach_inference_network(self, f_ratio=2.):
        x = self.graph.get_tensor_by_name('X:0')
        is_train = self.graph.get_tensor_by_name('is_train:0')

        with self.graph.as_default():
            kernel_init = tf.initializers.random_normal(mean=0.0, stddev=0.01)
            bias_one_init = tf.initializers.constant(1)

            with tf.variable_scope('CONV1'):
                c1 = tf.layers.Conv2D(96 // f_ratio, (11, 11), strides=4, padding='SAME',
                                      kernel_initializer=kernel_init,
                                      activation=tf.nn.relu)(x)
                c1 = tf.nn.local_response_normalization(c1, depth_radius=2,
                                                        bias=1, alpha=2e-5,
                                                        beta=0.75, name='LRN')

            s1 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MAXPOOL1')(c1)

            with tf.variable_scope('CONV2'):
                c2 = tf.layers.Conv2D(256 // f_ratio, (5, 5), strides=1, padding='SAME',
                                      kernel_initializer=kernel_init,
                                      bias_initializer=bias_one_init,
                                      activation=tf.nn.relu)(s1)
                c2 = tf.nn.local_response_normalization(c2, depth_radius=2,
                                                        bias=1, alpha=2e-5,
                                                        beta=0.75, name='LRN')

            s2 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MAXPOOL2')(c2)

            c3 = tf.layers.Conv2D(384 // f_ratio, (3, 3), strides=1, padding='SAME',
                                  kernel_initializer=kernel_init,
                                  activation=tf.nn.relu, name='CONV3')(s2)
            c3 = tf.layers.Conv2D(384 // f_ratio, (3, 3), strides=1, padding='SAME',
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_one_init,
                                  activation=tf.nn.relu, name='CONV4')(c3)
            c4 = tf.layers.Conv2D(256 // f_ratio, (3, 3), strides=1, padding='SAME',
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_one_init,
                                  activation=tf.nn.relu, name='CONV5')(c3)

            with tf.variable_scope('FC1'):
                c4 = tf.layers.Flatten()(c4)
                f1 = tf.layers.Dense(4096 // f_ratio, kernel_initializer=kernel_init,
                                     activation=tf.nn.relu)(c4)
                f1 = tf.layers.Dropout(rate=0.5)(f1, training=is_train)

            with tf.variable_scope('FC2'):
                f2 = tf.layers.Dense(4096 // f_ratio, kernel_initializer=kernel_init,
                                     activation=tf.nn.relu)(f1)
                f2 = tf.layers.Dropout(rate=0.5)(f2, training=is_train)

            with tf.variable_scope('OUTPUT'):
                logits = tf.layers.Dense(self._num_classes, kernel_initializer=kernel_init)(f2)

            logits = tf.identity(logits, name='logits')
            y_pred = tf.nn.softmax(logits, name='prediction')

        return self

    def _attach_loss_network(self):
        labels = self.graph.get_tensor_by_name('labels:0')
        logits = self.graph.get_tensor_by_name('logits:0')

        with self.graph.as_default():
            with tf.variable_scope('losses'):
                cee = tf.losses.sparse_softmax_cross_entropy(labels, logits)
            loss = tf.identity(cee, name='loss')

        return self

    def _attach_metric_network(self):
        labels = self.graph.get_tensor_by_name('labels:0')
        logits = self.graph.get_tensor_by_name('logits:0')

        with self.graph.as_default():
            with tf.variable_scope('metrics'):
                top_5 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=5), tf.float32))*100
                top_1 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32))*100

            top_5 = tf.identity(top_5, name='top_5_accuracy')
            top_1 = tf.identity(top_1, name='top_1_accuracy')
            tf.summary.scalar('top5_accuracy', top_5)
            tf.summary.scalar('top1_accuracy', top_1)

        return self

    def _attach_summary_network(self):
        with self.graph.as_default():
            with tf.variable_scope('summaries'):
                merged = tf.summary.merge_all()
                tf.add_to_collection(tf.GraphKeys.SUMMARY_OP, merged)
        return self

    def _attach_optimizer_network(self, momentum=0.9, weight_decay=5e-4):
        lr = self.graph.get_tensor_by_name('learning_rate:0')
        loss = self.graph.get_collection(tf.GraphKeys.LOSSES)[0]

        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                with tf.variable_scope('l2_loss'):
                    weights = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="[\w\/]*/kernel:0")
                    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in weights], name='l2_loss')
                    self.graph.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,l2_loss)

                    loss = loss + l2_loss

                global_step = tf.train.get_or_create_global_step()
                train_op = (tf.train
                            .MomentumOptimizer(lr, momentum=momentum, use_nesterov=True)
                            .minimize(loss, global_step=global_step))

        return self

    def build_network(self):
        return (self._initialize_placeholders()
                ._attach_inference_network()
                ._attach_loss_network()
                ._attach_metric_network()
                ._attach_summary_network()
                ._attach_optimizer_network())