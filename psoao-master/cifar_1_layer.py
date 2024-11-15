import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops #用于控制依赖项，确保在训练时更新 batch normalization 参数。
from datetime import datetime
import numpy as np
import os
import get_data #这是用户自定义的模块，用于获取训练和测试数据
import tensorflow as tf

'''
2.79E-03
C[K:2-F:24-L2:0.0016]
C[K:3-F:57-L2:0.0001]
C[K:5-F:63-L2:0.0096]
C[K:5-F:35-L2:0.0071]
C[K:3-F:76-L2:0.0015]
P[K:2-S:2]
'''

batch_size = 128
total_epochs = 100

def model():

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name='Input') 
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='True_Y')
    y = tf.cast(y, tf.int64)
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='dropout')     #dropout 概率，用于控制 dropout 的保留概率。
    is_training = tf.placeholder(tf.bool, shape=())    #布尔值占位符，用于区分训练和测试阶段（主要用于 batch normalization）

    #卷积层和池化层
    #使用 CReLU 作为激活函数，比 ReLU 复杂度稍高，但能捕捉更多特征。
    # slim.arg_scope：提供参数共享，减少代码冗余。
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.crelu, normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training, 'decay': 0.95}):
        h = slim.conv2d(inputs=x, num_outputs=24, kernel_size=2, weights_regularizer=slim.l2_regularizer(0.0016))
        h = slim.conv2d(inputs=h, num_outputs=57, kernel_size=3, weights_regularizer=slim.l2_regularizer(0.0001))
        h = slim.conv2d(inputs=h, num_outputs=63, kernel_size=5, weights_regularizer=slim.l2_regularizer(0.0096))
        h = slim.conv2d(inputs=h, num_outputs=35, kernel_size=5, weights_regularizer=slim.l2_regularizer(0.0071))
        h = slim.conv2d(inputs=h, num_outputs=76, kernel_size=3, weights_regularizer=slim.l2_regularizer(0.0015))
        h = slim.max_pool2d(h, kernel_size=2, stride=2)
        flatten = slim.flatten(h)
        full = slim.fully_connected(flatten, 512)
        drop_full = slim.dropout(full, keep_prob)     #使用 dropout 防止过拟合。
        
        with tf.name_scope('accuracy'):
            logits = slim.fully_connected(drop_full, 10, activation_fn=None)    #模型的最终输出，10 个类别的得分。
            correct_prediction = tf.equal(tf.argmax(logits, 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # accuracy：计算模型预测的准确率。
        with tf.name_scope('loss'):    #使用交叉熵损失函数，并加入 L2 正则化项。
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))+ tf.add_n(tf.losses.get_regularization_losses())
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()    #采用 Adam 优化器。
            step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)#简单来说，这个变量的作用就是记录当前训练到了第几步
            train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)#这个函数的作用就是根据损失函数和优化器来创建一个训练操作。 每执行一次这个操作，模型的参数就会更新一次，同时 "step" 变量也会加1，。
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    #确保 batch normalization 的参数在训练过程中更新。
            if update_ops:
                updates = tf.group(*update_ops)
                loss = control_flow_ops.with_dependencies([updates], loss)

        #训练与测试
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())    #执行训练操作。
                train_data, train_label = get_data.get_train_data(True)
                validate_data, validate_label = get_data.get_test_data(True)
                epochs = total_epochs
                for current_epoch in range(epochs):
                    train_loss_list = []
                    train_accu_list = []
                    total_length = train_data.shape[0]
                    idx = np.arange(total_length)
                    np.random.shuffle(idx)    #每个 epoch 进行数据随机化，避免模型对训练数据顺序的依赖。
                    train_data = train_data[idx]
                    train_label = train_label[idx]
                    total_steps = total_length // batch_size
                    for step in range(total_steps):
                        batch_train_data = train_data[step*batch_size:(step+1)*batch_size]
                        batch_train_label = train_label[step*batch_size:(step+1)*batch_size]
                        _, loss_v, accuracy_str = sess.run([train_op, loss, accuracy], {x:batch_train_data, y:batch_train_label, keep_prob:0.5, is_training:True})
                        train_loss_list.append(loss_v)
                        train_accu_list.append(accuracy_str)

                    #test 每个 epoch 后测试一次模型，输出训练和测试的损失及准确率。
                    test_length = validate_data.shape[0]
                    test_steps = test_length // batch_size
                    test_loss_list = []
                    test_accu_list = []
                    for step in range(test_steps):
                        batch_test_data = validate_data[step*batch_size:(step+1)*batch_size]
                        batch_test_label = validate_label[step*batch_size:(step+1)*batch_size]
                        loss_v, accuracy_str = sess.run([loss, accuracy], {x:batch_test_data, y:batch_test_label, keep_prob:1.0, is_training:False})
                        test_loss_list.append(loss_v)
                        test_accu_list.append(accuracy_str)

                    print('{}, epoch:{}/{}, step:{}/{}, loss:{:.6f}, accu:{:.4f}, test loss:{:.6f}, accu:{:.4f}'.format(datetime.now(), current_epoch, total_epochs, total_steps*current_epoch+step, total_steps*epochs, np.mean(train_loss_list), np.mean(train_accu_list), np.mean(test_loss_list), np.mean(test_accu_list)))

if __name__ =='__main__':
    #cuda 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #指定使用第 0 块 GPU
    tf.reset_default_graph()     #清空默认图，避免重复定义时的冲突。
    model() # mean

