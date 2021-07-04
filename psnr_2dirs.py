import argparse
import os
from skimage.measure import compare_psnr, compare_ssim
import imageio
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./val/T')
parser.add_argument('-d1','--dir1', type=str, default='./pred')
parser.add_argument('-o','--out', type=str, default='psnr.txt')

opt = parser.parse_args()


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
                dist01 = compare_psnr(img0,img1)
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
