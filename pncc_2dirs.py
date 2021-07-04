import argparse
import os
from skimage.measure import compare_psnr, compare_ssim
import imageio
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import subprocess
from scipy import io
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./val/T')
parser.add_argument('-d1','--dir1', type=str, default='./pred')
parser.add_argument('-o','--out', type=str, default='pncc.txt')

opt = parser.parse_args()

vgg_path = io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")


# build VGG19 to load pre-trained parameters

def build_net(ntype, nin, nwb=None, name=None):
        if ntype == 'conv':
                return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
        elif ntype == 'pool':
                return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

def build_vgg19(input, reuse=False):
        with tf.variable_scope("vgg19"):
                if reuse:
                        tf.get_variable_scope().reuse_variables()
                net = {}
                vgg_layers = vgg_path['layers'][0]
                net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
                net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
                net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
                net['pool1'] = build_net('pool', net['conv1_2'])
                net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
                net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
                net['pool2'] = build_net('pool', net['conv2_2'])
                net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
                net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
                net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
                net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
                net['pool3'] = build_net('pool', net['conv3_4'])
                net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
                net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
                net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
                net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
                net['pool4'] = build_net('pool', net['conv4_4'])
                net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
                net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
                return net

def compute_percep_ncc_loss(input, output, reuse=False):
        weight_in = 1 / (tf.reduce_max(tf.abs(input)) + 1e-10)
        weight_out = 1 / (tf.reduce_max(tf.abs(output)) + 1e-10)

        input = input * weight_in
        output = output * weight_out

        zero_mat = tf.zeros(tf.shape(output), tf.float32)
        output = tf.where(tf.greater(output, 0), output, zero_mat)
        input = tf.where(tf.greater(input, 0), input, zero_mat)

        losses = []
        for l in range(3):
                losses.append(compute_pncc_loss(input, output))
                input = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                output = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        return sum(losses) / len(losses)

def compute_ncc_loss(a, b):
        vector_a = slim.flatten(a)[0]
        vector_b = slim.flatten(b)[0]
        mean_a, var_a = tf.nn.moments(vector_a, axes=0)
        mean_b, var_b = tf.nn.moments(vector_b, axes=0)
        new_a = tf.divide((vector_a - mean_a), tf.sqrt(var_a) + 1e-7)
        new_b = tf.divide((vector_b - mean_b), tf.sqrt(var_b) + 1e-7)
        return tf.abs(tf.reduce_mean(new_a * new_b))

def compute_pncc_loss(input, output, reuse=False):
        vgg_real = build_vgg19(output * 255.0, reuse=reuse)
        vgg_fake = build_vgg19(input * 255.0, reuse=True)
        p1 = compute_ncc_loss(vgg_real['conv1_2'], vgg_fake['conv1_2'])
        p2 = compute_ncc_loss(vgg_real['conv2_2'], vgg_fake['conv2_2'])
        p3 = compute_ncc_loss(vgg_real['conv3_2'], vgg_fake['conv3_2'])
        p4 = compute_ncc_loss(vgg_real['conv4_2'], vgg_fake['conv4_2'])
        p5 = compute_ncc_loss(vgg_real['conv5_2'], vgg_fake['conv5_2'])
        return (p2 + p3 + p4) / 3

# get available GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

with tf.variable_scope(tf.get_variable_scope()):
        I1=tf.placeholder(tf.float32,shape=[None,None,None,3])
        I2=tf.placeholder(tf.float32,shape=[None,None,None,3])
        tf_pncc=compute_percep_ncc_loss(I1,I2)

######### Session #########
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())
######### Session #########

def compare_pncc(sess, tf_pncc, img1, img2):
        pncc = sess.run(tf_pncc,feed_dict={I1:img1[np.newaxis,:,:,:],I2:img2[np.newaxis,:,:,:]})
        return pncc

# crawl directories
f = open(opt.out,'w')
files = sorted(os.listdir(opt.dir0)) # gt
files1 = sorted(os.listdir(opt.dir1)) # pred

dist_list = []
missing_files = []
for idx in tqdm(range(len(files))):
        file = files[idx]
        if(os.path.exists(os.path.join(opt.dir1,files1[idx]))):
                file1 = files1[idx].replace("_M_", "_T_")
                if file1 != file:
                        print("!!!WARNING: unmatched pair - %s vs %s"%(file, file1))
                # Load images
                img0 = imageio.imread(os.path.join(opt.dir0,file))/255. # RGB image from [-1,1]
                img1 = imageio.imread(os.path.join(opt.dir1,files1[idx]))/255.
                # Compute distance
                dist01 = compare_pncc(sess, tf_pncc, img0,img1)
                # print('%s %.3f'%(file,dist01))
                f.writelines('%s %.6f\n'%(file,dist01))
                dist_list.append(dist01)
        else:
                missing_files.append(file)
print("np.mean(dist_list)", np.mean(dist_list))
if len(missing_files) > 0:
        print("warnings - missing following files in your output dir: ")
        print(missing_files)
f.close()
