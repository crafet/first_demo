# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:12:23 2017


"""
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
from vocab import vocab

class DNNModel(object):
    def __init__(self,FLAGS,wide_dim,config = None):
        self.FLAGS = FLAGS
        hidden_units = [int(i) for i in FLAGS.model_network.split(',')]
        self.hidden_units = hidden_units
        self.wide_dim = wide_dim
        self.num_ps = config['num_ps'] if config else 0
        self.params = []
        self.total_params = 0

    def identity(self,x):
        return x
        
    def linear(self,inputs,input_dim,output_dim,act=tf.nn.sigmoid,name="linear"):
        info = "Linear: name = {}, shape = ({},{}), act = {}, num_params = {}".format(
            name,input_dim,output_dim,act.func_name,input_dim*output_dim
        )
        self.total_params += input_dim * output_dim
        self.params.append(info)
        with tf.variable_scope(name):
            weights = tf.get_variable("W",shape=[input_dim,output_dim],initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("B",shape=[output_dim],initializer=tf.zeros_initializer())
            return act(tf.matmul(inputs,weights) + biases)

    def linear_sparse(self,inputs,input_dim,output_dim,act=tf.nn.sigmoid,name="linear_sparse"):
        info = "Linear_sparse: name = {}, shape = ({},{}), act = {}, num_params = {}".format(
            name,input_dim,output_dim,act.func_name,input_dim*output_dim
        )
        self.total_params += input_dim * output_dim
        self.params.append(info)
        with tf.variable_scope(name):
            weights = tf.get_variable("W",[input_dim,output_dim],initializer=tf.truncated_normal_initializer(stddev=0.1))
            # weights = tf.get_variable("W",[input_dim,output_dim],initializer=tf.zeros_initializer())
            biases = tf.get_variable("B",[output_dim],initializer=tf.zeros_initializer())
            activation = act(tf.sparse_tensor_dense_matmul(inputs,weights) + biases)
            return activation
    
    def hash_column(self,input_column,hash_bucket_size,name="hash_column"):
        info = "Hash: name = {}, bucket_size = {}, num_params = 0".format(
            name,hash_bucket_size
        )
        self.params.append(info)
        with tf.variable_scope(name):
            col = tf.string_to_hash_bucket_fast(tf.as_string(input_column),hash_bucket_size)
            return col

    def float_reshape(self,input_column,name="float_reshape"):
        with tf.variable_scope(name):
            col = tf.to_float(tf.reshape(input_column,[-1,1]))
            return col

    def embedding_layer(self,inputs,input_dim,output_dim,name="embedding_layer"):
        info = "Embedding: name = {}, shape = ({},{}), num_params = {}".format(
            name,input_dim,output_dim,input_dim * output_dim
        )
        self.total_params += input_dim * output_dim
        self.params.append(info)
        with tf.variable_scope(name):
            embedding_variables = tf.get_variable("embedding_w",
                    [input_dim,output_dim],
                    initializer=tf.random_uniform_initializer())
            embedding = tf.nn.embedding_lookup(embedding_variables,inputs)
            return embedding

    def get_optimizer(self,opt_name, learning_rate):
        if opt_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif opt_name == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5, use_nesterov=True)
        elif opt_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif opt_name == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif opt_name == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        else:
            optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                               initial_accumulator_value=1.0,
                                               l1_regularization_strength=1.5,
                                               l2_regularization_strength=1.0)
        return optimizer

    def _check_no_sync_replicas_optimizer(self,optimizer):
        if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
            raise ValueError(
                'SyncReplicasOptimizer does not support multi optimizers case. '
                'Therefore, it is not supported in DNNLinearCombined model. '
                'If you want to use this optimizer, please use either DNN or Linear '
                'model.')

    def build(self):
        input_dim = self.FLAGS.input_dim
        model_type = self.FLAGS.model_type

        ## step 1. build inputs and labels
        dense_inputs = tf.placeholder(tf.float32,[None,input_dim],name=vocab.dense_inputs)
        sparse_inputs = tf.sparse_placeholder(tf.float32,name="sparse_inputs")
        labels = tf.placeholder(tf.float32,[None,1],name=vocab.labels)

        self.dense_inputs = dense_inputs
        self.sparse_inputs = sparse_inputs
        self.labels = labels

        ## STEP 2. build model
        deep_columns_dict={}
        deep_dim_dict={}
        order_bucket = 500000
        channel_bucket = 150000
        plat_bucket = 100
        area_bucket = 5000
        advertiser_bucket = 50000
        industry_bucket = 2000
        loc_bucket = 5000
        tuwenid_bucket = 1000

        deep_dim_dict["order_embedding_dim"] = 16
        deep_dim_dict["channel_embedding_dim"] = 14
        deep_dim_dict["plat_embedding_dim"] = 4
        deep_dim_dict["area_embedding_dim"] = 10
        deep_dim_dict["advertiser_embedding_dim"] = 13
        deep_dim_dict["industry_embedding_dim"] = 8
        deep_dim_dict["loc_embedding_dim"] = 9
        deep_dim_dict["tuwenid_embedding_dim"] = 6

        ## embedding dict of deep dim dict
        num_ps_replicas = self.num_ps

        dnn_parent_scope = "dnn"
        if model_type == "wide":
            dnn_logits = None
        else:
            dnn_optimizer = self.get_optimizer(self.FLAGS.dnn_optimizer,self.FLAGS.dnn_lr)
            self._check_no_sync_replicas_optimizer(dnn_optimizer)
            dnn_partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)
            with tf.variable_scope(dnn_parent_scope,
                                   partitioner=dnn_partitioner):
                orders = self.hash_column(dense_inputs[:,1],order_bucket,name="orders")
                channels = self.hash_column(dense_inputs[:, 3], channel_bucket, name="channels")
                plats = self.hash_column(dense_inputs[:, 0], plat_bucket, name="plats")
                areas = self.hash_column(dense_inputs[:, 6], area_bucket, name="areas")
                advertiser = self.hash_column(dense_inputs[:, 8], advertiser_bucket, name="advertiser")
                industry = self.hash_column(dense_inputs[:, 10], industry_bucket, name="industry")
                location = self.hash_column(dense_inputs[:, 2], loc_bucket, name="location")
                tuwenid = self.hash_column(dense_inputs[:, 9], tuwenid_bucket, name="tuwenid")
                deep_columns_dict["order_embedding"] = self.embedding_layer(
                    orders, order_bucket, deep_dim_dict["order_embedding_dim"],
                    name="order_embedding")
                deep_columns_dict["channel_embedding"] = self.embedding_layer(
                    channels, channel_bucket, deep_dim_dict["channel_embedding_dim"],
                    name="channel_embedding")
                deep_columns_dict["plat_embedding"] = self.embedding_layer(
                    plats, plat_bucket, deep_dim_dict["plat_embedding_dim"],
                    name="plat_embedding")
                deep_columns_dict["area_embedding"] = self.embedding_layer(
                    areas, area_bucket, deep_dim_dict["area_embedding_dim"],
                    name="area_embedding")
                deep_columns_dict["advertiser_embedding"] = self.embedding_layer(
                    advertiser, advertiser_bucket, deep_dim_dict["advertiser_embedding_dim"],
                    name="advertiser_embedding")
                deep_columns_dict["industry_embedding"] = self.embedding_layer(
                    industry, industry_bucket, deep_dim_dict["industry_embedding_dim"],
                    name="industry_embedding")
                deep_columns_dict["loc_embedding"] = self.embedding_layer(
                    location, loc_bucket, deep_dim_dict["loc_embedding_dim"],
                    name="loc_embedding")
                deep_columns_dict["tuwenid_embedding"] = self.embedding_layer(
                    tuwenid, tuwenid_bucket, deep_dim_dict["tuwenid_embedding_dim"],
                    name="tuwenid_embedding")
                deep_columns_dict["gender"] = self.float_reshape(dense_inputs[:, 5], name="gender")
                deep_columns_dict["network"] = self.float_reshape(dense_inputs[:, 4], name="network")
                deep_columns_dict["clk7d"] = self.float_reshape(dense_inputs[:, 11], name="clk7d")
                deep_columns_dict["imp7d"] = self.float_reshape(dense_inputs[:, 12], name="imp7d")
                deep_columns_dict["age"] = self.float_reshape(dense_inputs[:, 7], name="age")
                embedding_dim = 0
                for k, v in deep_dim_dict.items():
                    embedding_dim += v
                embedding_dim += 5
                deep_columns = [column for name, column in deep_columns_dict.items()]

                deep_input_layer = tf.concat(deep_columns, 1,name="deep_input_layer") # input_layer
                layer_num = len(self.hidden_units)
                layer = self.linear(deep_input_layer, embedding_dim, self.hidden_units[0], act=tf.nn.tanh, name="hidden_0")
                for i in range(1, layer_num):
                    layer = self.linear(layer, self.hidden_units[i - 1], self.hidden_units[i],
                                        act=tf.nn.tanh, name="hidden_%d" % i)
                dnn_logits = self.linear(layer, self.hidden_units[-1], 1, act=self.identity,name="dnn_logits")

        linear_parent_scope = "linear"
        if model_type == "deep":
            linear_logits = None
        else:
            linear_optimizer = self.get_optimizer(self.FLAGS.linear_optimizer,self.FLAGS.linear_lr)
            self._check_no_sync_replicas_optimizer(linear_optimizer)
            linear_partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)
            with tf.variable_scope(linear_parent_scope,partitioner=linear_partitioner):
                ## bug_fix. Cause loss = NaN error
                ## data index starts from 1, but weights index start from zero
                linear_logits = self.linear_sparse(sparse_inputs,
                                                 self.wide_dim + 1,1,
                                                 act=self.identity,
                                                 name="linear_logits")

        if model_type == 'deep':
            logits = dnn_logits
        elif model_type == 'wide':
            logits = linear_logits
        else:
            logits = dnn_logits + linear_logits


        ## step 3. build loss, metrics, train_op or prediction_op and so on
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=labels,
                                logits=logits),name=vocab.loss)
        train_ops = []
        if dnn_logits is not None:
            grads_and_vars = dnn_optimizer.compute_gradients(loss,
                                                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                               scope = dnn_parent_scope))
            cliped_gvs =[(tf.clip_by_value(grad,-5.0,5.0),var) for grad,var in grads_and_vars]
            train_ops.append(
                dnn_optimizer.apply_gradients(cliped_gvs)
            )

        if linear_logits is not None:
            grads_and_vars = linear_optimizer.compute_gradients(loss,
                                                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope = linear_parent_scope))
            cliped_gvs = [(tf.clip_by_value(grad,-5.0,5.0),var) for grad,var in grads_and_vars]
            train_ops.append(
                linear_optimizer.apply_gradients(cliped_gvs)
            )

        train_op = control_flow_ops.group(*train_ops,name=vocab.train_op)
        global_step = tf.Variable(0, name=vocab.global_step, trainable=False)
        step_update_op = tf.assign_add(global_step, 1, name=vocab.step_update_op)
        prediction_op = tf.nn.sigmoid(logits,name=vocab.prediction_op)
        _, auc_op = tf.metrics.auc(labels=self.labels, predictions=prediction_op)

        tf.summary.scalar("loss",loss)
        tf.summary.scalar("auc",auc_op)
        summary_op = tf.summary.merge_all()

        self.params.append("Total params: {}".format(self.total_params))
        self.loss = loss
        self.global_step = global_step
        self.step_update_op = step_update_op
        self.train_op = train_op
        self.prediction_op = prediction_op
        self.auc_op = auc_op
        self.summary_op = summary_op
