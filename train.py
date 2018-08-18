# -*- coding: utf-8 -*-
"""
modified by online_train_distributed.py, converted into tensorflowonspark

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster, TFNode
from datetime import datetime
import sys, os


def parse_files(data_path):
    file_names = list()
    for idx in range(0, 256):
        file_names.append(data_path + "/part-" + "%05d" % idx)
    return file_names


def main_fun(argv, ctx):
    import pprint
    import numpy as np
    import tensorflow as tf
    import online_model
    import tfos_online_data_reader

    sys.argv = argv
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('batch_size', 100, 'data batch size')
    flags.DEFINE_integer('num_epoch', 1, 'train epoches for dataset ')
    flags.DEFINE_string('mapping_data',
                        'hdfs://appcluster-cdh/user/root/Adwin_Refactoring_Test/instance_build_txt/mix_dev_wx_interest2/20171022_map',
                        'id mapping path')
    flags.DEFINE_string('train_data',
                        'hdfs://appcluster-cdh/user/root/Adwin_Refactoring_Test/instance_build_txt/mix_dev_wx_interest2/20171022',
                        'train data path')
    #flags.DEFINE_string('mapping_data',
    #                    'hdfs://appcluster-cdh/user/root/tensorflow/app/online_train_distributed/mix_dev_wx_interest2/20171022_map',
    #                    'id mapping path')
    #flags.DEFINE_string('train_data',
    #                    'hdfs://appcluster-cdh/user/root/tensorflow/app/online_train_distributed/mix_dev_wx_interest2/20171022',
    #                    'train data path')
    flags.DEFINE_string('log_dir',
                        'hdfs://appcluster-cdh/user/root/tensorflow/app/online_train_distributed/model',
                        'log directory')

    flags.DEFINE_float('linear_lr', 0.1, 'wide part learning rate. default 0.1')
    flags.DEFINE_float('dnn_lr', 0.001, 'deep part learning rate. default 0.001')
    flags.DEFINE_string('linear_optimizer', 'ftrl',
                        'optimizer: adadelta | adagrad | sgd | adam | ftrl | momentum. default is ftrl')
    flags.DEFINE_string('dnn_optimizer', 'adagrad',
                        'optimizer: adadelta | adagrad | sgd | adam | ftrl | momentum. default is adagrad')

    flags.DEFINE_integer('input_dim', 13, 'input dimension')
    flags.DEFINE_string("model_network", "100,20", "The neural network of model, as 100,50,20")
    flags.DEFINE_string("model_type", "wide_deep", "model type: wide | deep | wide_deep")
    flags.DEFINE_integer('display_step', 200, 'display_step')

    flags.DEFINE_integer('ps_num', '64', 'Comma-separated list of hostname:port pairs')
    flags.DEFINE_integer('task_num', '128', 'Comma-separated list of hostname:port pairs')

    pprint.PrettyPrinter().pprint(FLAGS.__flags)
    cluster_spec, server = TFNode.start_cluster_server(ctx)
    if ctx.job_name == "ps":
        server.join()
    elif ctx.job_name == "worker":
        total_file_names = parse_files(FLAGS.train_data)
        print("total_file_names:")
        print(total_file_names)
        print("task_index: " + str(ctx.task_index))
        task_file_names = [name for idx, name in enumerate(total_file_names) if idx % FLAGS.task_num == ctx.task_index]
        print("task_file_names:")
        print(task_file_names)
        train_reader = tfos_online_data_reader.Reader(
            task_file_names,
            FLAGS.mapping_data,
            batch_size=FLAGS.batch_size,
            delimiter='\t')
        wide_dim = train_reader.wide_dim

        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%ctx.task_index,
                                                      cluster=cluster_spec)):
            config = {}
            config['num_ps'] = FLAGS.ps_num
            dnn_model = online_model.DNNModel(FLAGS,wide_dim,config)
            dnn_model.build()
            dense_inputs = dnn_model.dense_inputs
            sparse_inputs = dnn_model.sparse_inputs
            labels = dnn_model.labels

            global_step = dnn_model.global_step
            step_update_op = dnn_model.step_update_op
            train_op = dnn_model.train_op
            loss = dnn_model.loss
            auc_op = dnn_model.auc_op
            summary_op = dnn_model.summary_op

        saver = tf.train.Saver()
        init_op = [tf.global_variables_initializer(),
                    tf.local_variables_initializer()]

        summary_writer = tf.summary.FileWriter("tensorboard_%d" % ctx.worker_num, graph=tf.get_default_graph())
        sv = tf.train.Supervisor(is_chief = (ctx.task_index == 0),
                                 logdir = FLAGS.log_dir,
                                 init_op = init_op,
                                 summary_op = None,
                                 summary_writer=summary_writer,
                                 global_step = global_step,
                                 saver=saver,
                                 save_model_secs = 300)

        shape = np.array([FLAGS.batch_size, wide_dim + 1])
        begin_time = datetime.now()
        with sv.managed_session(server.target) as sess:
            if not sv.should_stop():
                for epoch in range(FLAGS.num_epoch):
                    train_batches = train_reader.yieldBatches()
                    print("Epoch: %d" % epoch)
                    step = 0
                    for dense_x,sparse_idx,sparse_values,y in train_batches:
                        start_time = datetime.now()
                        _ ,train_loss,train_auc,summ,_ = sess.run([train_op,loss,auc_op,summary_op,step_update_op],
                           feed_dict={dense_inputs:dense_x,sparse_inputs:(sparse_idx,sparse_values,shape),labels:y})
                        step += 1
                        assert not np.isnan(train_loss), 'Model diverged with loss = NaN'
                        time_used = datetime.now() - start_time
                        if step % FLAGS.display_step == 0:
                            g_step, = sess.run([global_step])
                            print("step: " + str(step) + ", global_step: " + str(g_step))
                            summary_writer.add_summary(summ,g_step)
                            print("Step = {}, Examples = {}, Time = {}, Minibatch Loss = {}, Auc = {}".format(
                                 g_step, g_step*FLAGS.batch_size, time_used, train_loss, train_auc))
                            sys.stdout.flush()
            total_time = datetime.now() - begin_time
            print("Training Done!!")
            print("Total time used: {}".format(total_time))


if __name__ == "__main__":
    sc = SparkContext(conf=SparkConf().setAppName("tfos_online_train_distributed"))
    num_executors = int(sc._conf.get("spark.executor.instances"))
    num_ps = 64
    tensorboard = False
    cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_ps, tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown()
