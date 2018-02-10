import sys
import os,shutil
import collections
import numpy as np
import os
import tensorflow as tf
import cv2
from lib.lstm.utils.timer import Timer
from ..lstm.config import cfg,get_encode_decode_dict


class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, output_dir, logdir, pretrained_model=None):
        self.net = network
        self.imgdb = imgdb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        print('done')

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)


    def compute_acc(self, org, res):
        num_rt = 0.0
        for i in range(0, len(org)):
            if i > len(res) - 1:
                break
            if org[i] == res[i]:
                num_rt += 1
        acc = num_rt / len(org)
        return acc, num_rt, len(org)

    def compute_acc_liu(self, a, b):
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n
        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)
        acc = 1 - (current[n]/(1.0*m))
        right_words = m - current[n]
        total_words = m
        return acc, right_words, m

    def test_model(self,sess,testDir=None,restore = True):
        logits = self.net.get_output('logits')
        time_step_batch = self.net.get_output('time_step_len')
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, time_step_batch, merge_repeated=True)
        dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)

        img_size = cfg.IMG_SHAPE
        global_step = tf.Variable(0, trainable=False)
        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        # resuming a trainer
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, tf.train.latest_checkpoint(self.output_dir))
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise Exception('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        timer = Timer()
        num_right_words = 0
        num_total_words = 0
        
        fp_org = open('org_labels_1', mode = 'a')
        fp_dec = open('dec_labels_1', mode = 'a')

        for file_name in os.listdir(testDir):
            timer.tic()
            print (file_name)
            print (testDir)
            if cfg.NCHANNELS == 1: img = cv2.imread(os.path.join(testDir,file_name),0)/255.
            else : img = cv2.imread(os.path.join(testDir,file_name),1)/255.
#            w = img.shape[1]
#            h = img.shape[0]
#            w_new = int((60 * w) / h)
#            img = cv2.resize(img, (w_new, 60))
            img = cv2.resize(img,tuple(img_size))
            img = img.swapaxes(0,1)
            img = np.reshape(img, [1,img_size[0],cfg.NUM_FEATURES])
            feed_dict = {
                self.net.data: img,
                self.net.time_step_len: [cfg.TIME_STEP],
                self.net.keep_prob: 1.0
            }
            res = sess.run(fetches=dense_decoded[0], feed_dict=feed_dict)
            def decodeRes(nums,ignore= 0):
                encode_maps,decode_maps = get_encode_decode_dict()
                res = [decode_maps[i] for i in nums if i!=ignore]
                return res
            res = ''.join(decodeRes(res))
            org = file_name.rsplit('.',1)[0].rsplit('_',1)[1]
            if (res != '~' and res !='' and res != ' '):
                fp_org.write(org+'\n')
                fp_dec.write(res+ '\n')

            acc,right_num,total_num = self.compute_acc_liu(org, res)
            num_right_words += right_num
            num_total_words += total_num
            _diff_time = timer.toc(average=False)
        #    print('cost time: {:.3f}\n'.format(_diff_time))
            print ('acc = ',acc)
            print('org: {}\n'.format(org))
            print('res: {}\n'.format(res))
        total_acc = num_right_words / (1.0 * num_total_words)
        print('total_acc = {:.3f}'.format(total_acc))
  
        fp_org.close()
        fp_dec.close()

def test_net(network, imgdb, testDir, output_dir, log_dir, pretrained_model=None,restore=True):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.test_model(sess,testDir, restore=restore)
        print('done solving')
    

