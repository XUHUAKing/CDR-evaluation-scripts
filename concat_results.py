# import cv2
import glob
import os.path

import numpy as np

output_dir = "concat_results"
names = [
    ["gt/M", "gt/T", "CoRRN/T", "IBCLN/T", "Kim_et_al/T"],
    ["zhang/T", "CEILNet/T", "Arvan_et_al/T", "Li_et_al/T", "Yang_et_al/T"],
]

os.makedirs(output_dir, exist_ok=True)

# get image sequences ready
row_images = []
for row in names:
    col_images = []
    for folder_name in row:
        filenames = sorted(glob.glob(folder_name+"/*.png"))
        col_images.append(filenames)
    row_images.append(col_images)

num_images = len(row_images[0][0])
curr_id = 0
# start concatenating results
for i in range(num_images):
    vis_row = []
    for row in row_images:
        vis_col = []
        for col in row:
            img = cv2.imread(col[i])
            vis_col.append(img)
        # concat horizontally
        vis_row.append(cv2.hconcat(vis_col))
    # concat vertically
    vis_img = cv2.vconcat(vis_row)
    # save image using gt/M name
    basename = os.path.basename(filenames[i])
    cv2.imwrite(os.path.join(output_dir, basename), vis_img)
