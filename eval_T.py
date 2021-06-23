from imageio import imread, imsave
from glob import glob
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import cv2
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="crop", help="crop or padding")
parser.add_argument("--pred_dir", default="../../results/CoRRN/testset/testset_32/", help="path to prediction results")
parser.add_argument("--baseline", default="CoRRN", help="baseline_name")

ARGS = parser.parse_args()


gt_path = '../../data/testset/final_png_32/*T*'
pred_path= ARGS.pred_dir + '/*T*'
test_method = ARGS.method
baseline_name = ARGS.baseline

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

    if method=='padding':
        imsave(path_json.replace('.json','.jpg'),mask)
        imsave(path_json.replace('.json','T.jpg'),gt_t*mask[:h,:w])
        # print("crop_hw ", len(crop_hw))
        psnr = compare_psnr(gt_t*mask[:h,:w], pred_t*mask[:h,:w])
        ssim = compare_ssim(gt_t*mask[:h,:w], pred_t*mask[:h,:w],multichannel=True)
        ncc = compare_NCC(gt_t*mask[:h,:w], pred_t*mask[:h,:w])
        SI = compare_SI(gt_t*mask[:h,:w], pred_t*mask[:h,:w])
    elif method=="crop":
        psnrs,ssims,areas, nccs,SIs=[],[],[],[], []

        for i in range(len(crop_hw)):
            (h1,w1),(h2,w2) = crop_hw[i]
          #  print(path_json, h, w, h1, h2 ,w1, w2)
            imsave(path_json.replace('.json','T%d.jpg')%i,gt_t[h1:min(h,h2),w1:min(w,w2)])
            psnr=compare_psnr(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2])
            ssim=compare_ssim(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2],multichannel=True)
            ncc =compare_NCC(gt_t[h1:h2,w1:w2],pred_t[h1:h2,w1:w2])
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
