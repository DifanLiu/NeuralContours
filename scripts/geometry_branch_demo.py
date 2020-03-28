import os
from utils.base_util import pr2tensor, show_tensor
import cv2
import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-sc', type=float, required=True)
parser.add_argument('-r', type=float, required=True)
parser.add_argument('-v', type=float, required=True)
parser.add_argument('-ar', type=float, required=True)
parser.add_argument('-model_name', required=True)
parser.add_argument('-save_name', required=True)
args = parser.parse_args()
sc_thre = args.sc
r_thre = args.r
v_thre = args.v
ar_thre = args.ar
model_name = args.model_name
save_name = args.save_name

example_folder = os.path.join('data/example', model_name)

# ------------read suggestive contour------------
sc_fn = os.path.join(example_folder, 'suggestive_contour.png')
sc_data = 1.0 - cv2.imread(sc_fn, cv2.IMREAD_GRAYSCALE) / 255.0
sc_tensor = pr2tensor(sc_data)  # 1 * 1 * 1024 * 1024

# ------------read sc info------------
sci_fn = os.path.join(example_folder, 'suggestive_contour_info.txt')
with open(sci_fn, 'r') as txt_obj:
    one_line = txt_obj.readline().strip()
max_dwkr = float(one_line.split(' ')[0])
sc_fs = float(one_line.split(' ')[1])

# ------------read sc feature------------
scf_fn = os.path.join(example_folder, 'suggestive_contour_feature.png')
scf_data = cv2.imread(scf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * max_dwkr
scf_tensor = pr2tensor(scf_data, to_fill_value=max_dwkr)

# ------------read nv feature------------
openglnv_fn = os.path.join(example_folder, 'nv.png')
openglnv_data = cv2.imread(openglnv_fn, cv2.IMREAD_UNCHANGED) / 65535.0
openglnv_tensor = pr2tensor(openglnv_data, to_fill_value=1.0)

# ------------read apr------------
apr_fn = os.path.join(example_folder, 'apparent_ridge.png')
apr_data = 1.0 - cv2.imread(apr_fn, cv2.IMREAD_GRAYSCALE) / 255.0
apr_tensor = pr2tensor(apr_data)

# ------------read apr info------------
apri_fn = os.path.join(example_folder, 'apparent_ridge_info.txt')
with open(apri_fn, 'r') as txt_obj:
    one_line = txt_obj.readline().strip()
apr_max_info = float(one_line.split(' ')[0])

# ------------read apr feature------------
aprf_fn = os.path.join(example_folder, 'apparent_ridge_feature.png')
aprf_data = cv2.imread(aprf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * apr_max_info
aprf_tensor = pr2tensor(aprf_data, to_fill_value=apr_max_info)

# ------------read ridge------------
ridge_fn = os.path.join(example_folder, 'ridge.png')
ridge_data = 1.0 - cv2.imread(ridge_fn, cv2.IMREAD_GRAYSCALE) / 255.0
ridge_tensor = pr2tensor(ridge_data)

# ------------read valley------------
valley_fn = os.path.join(example_folder, 'valley.png')
valley_data = 1.0 - cv2.imread(valley_fn, cv2.IMREAD_GRAYSCALE) / 255.0
valley_tensor = pr2tensor(valley_data)

# ------------read ridge info------------
ri_fn = os.path.join(example_folder, 'ridge_info.txt')
with open(ri_fn, 'r') as txt_obj:
    one_line = txt_obj.readline().strip()
r_max_info = float(one_line.split(' ')[0])

# ------------read valley info------------
vi_fn = os.path.join(example_folder, 'valley_info.txt')
with open(vi_fn, 'r') as txt_obj:
    one_line = txt_obj.readline().strip()
v_max_info = float(one_line.split(' ')[0])

# ------------read ridge feature------------
rf_fn = os.path.join(example_folder, 'ridge_feature.png')
rf_data = cv2.imread(rf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * r_max_info
rf_tensor = pr2tensor(rf_data, to_fill_value=r_max_info)

# ------------read valley feature------------
vf_fn = os.path.join(example_folder, 'valley_feature.png')
vf_data = cv2.imread(vf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * v_max_info
vf_tensor = pr2tensor(vf_data, to_fill_value=v_max_info)

# ------------read base image------------
base_data = np.zeros((768, 1024))
sup_fn = os.path.join(example_folder, 'base.png')
base_data += 1.0 - cv2.imread(sup_fn, cv2.IMREAD_GRAYSCALE) / 255.0
base_data = np.clip(base_data, 0.0, 1.0)
base_tensor = pr2tensor(base_data)

# ------------create comparison tensor------------
comparison_data = np.zeros((768, 1024))
comparison_tensor = pr2tensor(comparison_data)

# ------------create feature size tensor------------
fs_data = np.zeros((768, 1024))
fs_data.fill(sc_fs)
fs_tensor = pr2tensor(fs_data, to_fill_value=sc_fs)

# ------------------------------start differentiable filtering------------------------------
hw = 1024
# ------sc------
sc_thre_image = sc_thre * torch.ones(1, 1, hw, hw, dtype=torch.float32).cuda()
thek = scf_tensor * fs_tensor * fs_tensor - sc_thre_image * openglnv_tensor
thek = torch.max(thek, comparison_tensor)
sc_mask = thek / (openglnv_tensor * 0.03 + thek + 1e-12)
sc_mask = sc_mask.clamp(0.0, 1.0)

# ------ridge------
r_thre_image = r_thre * torch.ones(1, 1, hw, hw, dtype=torch.float32).cuda()
r_mask = 1.0 - r_thre_image / (fs_tensor * rf_tensor + 1e-12)
r_mask = r_mask.clamp(0.0, 1.0)

# ------valley------
v_thre_image = v_thre * torch.ones(1, 1, hw, hw, dtype=torch.float32).cuda()
v_mask = 1.0 - v_thre_image / (fs_tensor * vf_tensor + 1e-12)
v_mask = v_mask.clamp(0.0, 1.0)

# ------apr------
apr_thre_image = ar_thre * torch.ones(1, 1, hw, hw, dtype=torch.float32).cuda()
apr_mask = 1.0 - apr_thre_image / (fs_tensor * fs_tensor * aprf_tensor + 1e-12)
apr_mask = apr_mask.clamp(0.0, 1.0)

# ------add all the lines together------

final_pr = base_tensor + sc_mask * sc_tensor + r_mask * ridge_tensor + v_mask * valley_tensor + apr_mask * apr_tensor

final_pr = final_pr.clamp(0.0, 1.0)

show_tensor(final_pr, crop=True, ofn=save_name)



