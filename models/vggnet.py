import tensorflow as tf
"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


class VGGNet:
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
                v for (v,f) in zip(global_vars,is_not_initialized) if not f]

            if len(not_initialized_vars):
                self.session.run(
                    tf.variables_initializer(not_initialized_vars))

    def _initialize_placeholders(self, input_shape=(None, 64, 64, 3)):
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, input_shape, name='X')
            labels = tf.placeholder(tf.int32, (None,), name='labels')

            is_train = tf.placeholder_with_default(False, (), name='is_train')
        return self

    def _attach_inference_network(self, conv_ratio=2., fc_ratio=4.):
        x = self.graph.get_tensor_by_name('X:0')
        is_train = self.graph.get_tensor_by_name('is_train:0')

        with self.graph.as_default():
            he_init = tf.initializers.he_normal()
            # (None,64,64,3) -> (None,32,32,...)
            with tf.variable_scope('VGG_Block1'):
                conv = tf.layers.Conv2D(64//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(x)
                conv = tf.layers.Conv2D(64//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                pool = tf.layers.MaxPooling2D((2, 2), (2, 2))(conv)

            # (None,32,32,...) -> (None,16,16,...)
            with tf.variable_scope('VGG_Block2'):
                conv = tf.layers.Conv2D(128//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(pool)
                conv = tf.layers.Conv2D(128//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                pool = tf.layers.MaxPooling2D((2, 2), (2, 2))(conv)

            # (None,16,16,...) -> (None,8,8,...)
            with tf.variable_scope('VGG_Block3'):
                conv = tf.layers.Conv2D(256//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(pool)
                conv = tf.layers.Conv2D(256//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                conv = tf.layers.Conv2D(256//conv_ratio, (1, 1), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                pool = tf.layers.MaxPooling2D((2, 2), (2, 2))(conv)

            # (None,8,8,...) -> (None,4,4,...)
            with tf.variable_scope('VGG_Block4'):
                conv = tf.layers.Conv2D(512//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(pool)
                conv = tf.layers.Conv2D(512//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                conv = tf.layers.Conv2D(512//conv_ratio, (1, 1), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                pool = tf.layers.MaxPooling2D((2, 2), (2, 2))(conv)

            with tf.variable_scope('VGG_Block5'):
                conv = tf.layers.Conv2D(512//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(pool)
                conv = tf.layers.Conv2D(512//conv_ratio, (3, 3), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)
                conv = tf.layers.Conv2D(512//conv_ratio, (1, 1), padding='SAME',
                                        kernel_initializer=he_init,
                                        activation=tf.nn.relu)(conv)

            with tf.variable_scope('VGG_FC'):
                conv = tf.layers.Flatten()(conv)
                fc1 = tf.layers.Dense(4096//fc_ratio,activation=tf.nn.relu,
                                     kernel_initializer=he_init)(conv)
                drop1 = tf.layers.Dropout(rate=0.5)(fc1, training=is_train)
                fc2 = tf.layers.Dense(4096//fc_ratio, activation=tf.nn.relu,
                                     kernel_initializer=he_init)(drop1)
                drop2 = tf.layers.Dropout(rate=0.5)(fc2, training=is_train)

            with tf.variable_scope('OUTPUT'):
                logits = tf.layers.Dense(self._num_classes)(drop2)

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
        loss = self.graph.get_collection(tf.GraphKeys.LOSSES)[0]

        with self.graph.as_default():
            with tf.variable_scope('metrics'):
                top_5, top_5_op = tf.metrics.mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=5), tf.float32) * 100)
                top_1, top_1_op = tf.metrics.mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32) * 100)
                full_loss, loss_op = tf.metrics.mean(loss)

            metric_init_op = tf.group([var.initializer for var in
                                       self.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)],
                                      name='metric_init_op')
            metric_update_op = tf.group([top_5_op, top_1_op, loss_op], name='metric_update_op')

            self.graph.add_to_collection('metric_ops', metric_init_op)
            self.graph.add_to_collection('metric_ops', metric_update_op)
            top_5 = tf.identity(top_5, name='top_5_accuracy')
            top_1 = tf.identity(top_1, name='top_1_accuracy')
            full_loss = tf.identity(full_loss, name='full_loss')
            tf.summary.scalar('top5_accuracy', top_5)
            tf.summary.scalar('top1_accuracy', top_1)
            tf.summary.scalar('full_loss', full_loss)

        return self

    def _attach_summary_network(self):
        with self.graph.as_default():
            with tf.variable_scope('summaries'):
                weights = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   scope="[\w\/]*/kernel:0")
                # Weight Distribution Check
                for weight in weights:
                    tf.summary.histogram(weight.op.name, weight)

                merged = tf.summary.merge_all()
                tf.add_to_collection(tf.GraphKeys.SUMMARY_OP, merged)
        return self

    def _attach_optimizer_network(self, init_lr=0.001, decay_steps=1562, decay_rate=0.9,
                                  momentum=0.9, weight_decay=5e-4):
        loss = self.graph.get_collection(tf.GraphKeys.LOSSES)[0]

        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                global_step = tf.train.get_or_create_global_step()
                with tf.variable_scope('learning_rate_decay'):
                    lr = tf.train.exponential_decay(init_lr, global_step,
                                                    decay_steps, decay_rate)
                lr = tf.identity(lr, name='learning_rate')

                with tf.variable_scope('l2_loss'):
                    weights = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope="[\w\/]*/kernel:0")
                    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in weights],name='l2_loss')
                    loss = loss + l2_loss

                self.graph.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)
                train_op = (tf.train.AdamOptimizer(lr)
                            .minimize(loss, global_step=global_step))

        return self

    def build_network(self):
        return (self._initialize_placeholders()
                ._attach_inference_network()
                ._attach_loss_network()
                ._attach_metric_network()
                ._attach_summary_network()
                ._attach_optimizer_network())