from imageio import imread, imsave
from glob import glob
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import cv2
import os
import json
import argparse
from scipy import io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import subprocess



parser = argparse.ArgumentParser()
parser.add_argument("--method", default="crop", help="crop or padding")
parser.add_argument("--pred_dir", default="../../results/CoRRN/testset/testset_32/", help="path to prediction results")
parser.add_argument("--baseline", default="CoRRN", help="baseline_name")

ARGS = parser.parse_args()


gt_path = '../../data/testset/final_png_32/*T*'
pred_path= ARGS.pred_dir + '/*T*'
test_method = ARGS.method
baseline_name = ARGS.baseline



vgg_path=io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")
# build VGG19 to load pre-trained parameters

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

def compute_percep_ncc_loss(input, output, reuse=False):
    weight_in = 1/(tf.reduce_max(tf.abs(input))+1e-10)
    weight_out = 1/(tf.reduce_max(tf.abs(output))+1e-10)
 
    input = input * weight_in
    output= output* weight_out

    zero_mat= tf.zeros(tf.shape(output),tf.float32)
    output  = tf.where(tf.greater(output,0),output,zero_mat)
    input  = tf.where(tf.greater(input,0),input,zero_mat)
 
    losses = [] 
    for l in range(3):
        losses.append(compute_pncc_loss(input,output))
        input=tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], padding='SAME')
        output=tf.nn.avg_pool(output, [1,2,2,1], [1,2,2,1], padding='SAME')
    return sum(losses)/len(losses) 

def compute_ncc_loss(a, b):
    vector_a = slim.flatten(a)[0]
    vector_b = slim.flatten(b)[0]
    mean_a, var_a = tf.nn.moments(vector_a,axes=0)
    mean_b, var_b = tf.nn.moments(vector_b,axes=0)
    new_a = tf.divide((vector_a-mean_a),tf.sqrt(var_a)+1e-7)
    new_b = tf.divide((vector_b-mean_b),tf.sqrt(var_b)+1e-7)
    return tf.abs(tf.reduce_mean(new_a*new_b))

def compute_pncc_loss(input, output, reuse=False):
    vgg_real=build_vgg19(output*255.0,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,reuse=True)
    p1=compute_ncc_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])
    p2=compute_ncc_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])
    p3=compute_ncc_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])
    p4=compute_ncc_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])
    p5=compute_ncc_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])
    return  (p2 +p3 +p4)/3





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


def compare_PNCC(sess, tf_pncc, img1, img2):
    pncc = sess.run(tf_pncc,feed_dict={I1:img1[np.newaxis,:,:,:],I2:img2[np.newaxis,:,:,:]})
    return pncc

def compare_SI(img1, img2):
    a = img1.flatten()
    b = img2.flatten()
    var_a, var_b = np.var(a), np.var(b)
    c = 1e-10
    SI = (2*var_a*var_b+c)/(var_a**2 + var_b**2+c)
    return SI


def compare_NCC(img1, img2):
    a = img1.flatten()
    b = img2.flatten()
    a = (a - np.mean(a)) / (1e-10+np.std(a))
    b = (b - np.mean(b)) / (1e-10+np.std(b))
    c = np.mean(a*b)
    # print("NCC: ", c)
    return c

def json_to_mask(path_json):
    base_name = path_json.split('/')[-1][0]
 #   print(base_name)
    if base_name == 'N':
        org_h, org_w = 2020//2,3032//2 
    elif base_name =="C":
        org_h, org_w =1835//2, 2748//2

    mask = np.zeros((org_h,3*org_w,3))
    with open(path_json) as f:
        data = json.load(f)
    channel_count = 3  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    crop_hw = []

    for j in range(len(data['shapes'])):
        label = data['shapes'][j]
        if label['shape_type'] == 'rectangle':
            points = np.array([label['points']], dtype=np.int32)//2
            (w1,h1), (w2,h2) = (points[0,0,0],points[0,0,1]),(points[0,1,0],points[0,1,1])
            cv2.rectangle(mask, (w1,h1), (w2,h2),  (255,255,255),-1)
            crop_hw.append([(min(h1,h2), min(w1-2*org_w,w2-2*org_w)), (max(h1,h2),max(w1-2*org_w,w2-2*org_w))])

    return mask[:,2*org_w:,:]/255.,  crop_hw

def eval_single(path_gt, path_pred, method='padding'):
    gt_t= imread(path_gt)/255.#[::10,::10]
    pred_t  = imread(path_pred)/255.#[::10,::10]
    h, w = gt_t.shape[:2]
    # print(path_gt)
    path_json = path_gt.replace('_T','').replace('.png','.json').replace('final_png','json')
    mask, crop_hw = json_to_mask(path_json)

    psnrs,ssims,areas, nccs,SIs=[],[],[],[], []

    for i in range(len(crop_hw)):
        (h1,w1),(h2,w2) = crop_hw[i]
      #  print(path_json, h, w, h1, h2 ,w1, w2)
        imsave(path_json.replace('.json','T%d.jpg')%i,gt_t[h1:min(h,h2),w1:min(w,w2)])
        psnr=compare_psnr(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2])
        ssim=compare_ssim(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2],multichannel=True)
        # ncc =compare_NCC(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2])
        ncc = compare_PNCC(sess, tf_pncc, gt_t[h1:h2:2,w1:w2:2], pred_t[h1:h2:2,w1:w2:2])

        SI  =compare_SI(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2])

        pixels_num = (h2-h1)*(w2-w1)
        psnrs.append(psnr*pixels_num)
        ssims.append(ssim*pixels_num)
        nccs.append(ncc*pixels_num)
        SIs.append(SI*pixels_num)
        areas.append(pixels_num) 

    psnr = sum(psnrs)/sum(areas)
    ssim = sum(ssims)/sum(areas)
    ncc = sum(nccs)/float(sum(areas))
    SI  = sum(SIs)/float(sum(areas))
    return psnr, ssim, ncc, SI

gt_imgs = sorted(glob(gt_path))
pred_imgs = sorted(glob(pred_path))


all_ssim, all_psnr,all_ncc = 0,0,0
all_SI = 0 
cnt = 0
records = []
for idx in range(len(pred_imgs)):
    path_pred= pred_imgs[idx] 
    path_gt  = gt_imgs[idx]
    if path_pred.split("/")[-1] !=  path_gt.split("/")[-1]:
        print(idx,path_pred, path_gt)    
    tmp_psnr, tmp_ssim,tmp_ncc,tmp_SI = eval_single(path_gt, path_pred, test_method)
    all_ssim+= tmp_ssim
    all_psnr+= tmp_psnr
    all_ncc += tmp_ncc
    all_SI  += tmp_SI
    cnt += 1
    if cnt%10 == 0:
        print(idx, ': %.2f %.3f %.3f %.3f'%(tmp_psnr, tmp_ssim, tmp_ncc, tmp_SI), path_gt)
    records.append("%s %.2f %.3f %.3f %.3f"%(path_gt.split('/')[-1], tmp_psnr, tmp_ssim, tmp_ncc, tmp_SI))
print(cnt,all_ssim*1.0/(cnt),all_psnr*1.0/(cnt),all_ncc*1.0/(cnt), all_SI*1.0/(cnt), test_method,pred_path)
with open('T_%s_%s.txt'%(baseline_name,test_method), 'w') as f:
    f.write("Name PSNR SSIM NCC SI\n")
    for item in records:
        f.write("%s\n" % item)
f.close()
