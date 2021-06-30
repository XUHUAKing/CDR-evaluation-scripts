import argparse
import os
from skimage.measure import compare_psnr, compare_ssim
import imageio
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./val/T')
parser.add_argument('-d1','--dir1', type=str, default='./pred')
parser.add_argument('-o','--out', type=str, default='ssim.txt')

opt = parser.parse_args()

def rgb_to_gray(rgb):
        R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
        gray = 0.2989 * R + 0.5870 * G + 0.1140 * B 
        gray = np.uint8(gray * 255.)
        return gray


# crawl directories
f = open(opt.out,'w')
files = sorted(os.listdir(opt.dir0)) # gt
files1 = sorted(os.listdir(opt.dir1)) # pred

dist_list = []
missing_files = []
for idx, file in enumerate(files):
        if(os.path.exists(os.path.join(opt.dir1,files1[idx]))):
                file1 = files1[idx].replace("_M_", "_T_")
                if file1 != file:
                        print("!!!WARNING: unmatched pair - %s vs %s"%(file, file1))
                # Load images
                img0 = imageio.imread(os.path.join(opt.dir0,file)) 
                img1 = imageio.imread(os.path.join(opt.dir1,files1[idx]))
                if len(img0.shape) == 3:
                        img0 = rgb_to_gray(img0 / 255.) # gray image from [0, 1]
                if len(img1.shape) == 3:
                        img1 = rgb_to_gray(img1 / 255.) # gray image from [0, 1]

                # Compute distance
                dist01 = compare_ssim(img0, img1)
                print('%s %.3f'%(file,dist01))
                f.writelines('%s %.6f\n'%(file,dist01))
                dist_list.append(dist01)
        else:
                missing_files.append(file)
print("np.mean(dist_list)", np.mean(dist_list))
if len(missing_files) > 0:
        print("warnings - missing following files in your output dir: ")
        print(missing_files)
f.close()
