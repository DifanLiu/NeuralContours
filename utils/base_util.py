import torch
import numpy as np
import cv2
import os
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def show_img(cv2_array, ofn=None):
    if ofn is None:
        cv2.imshow('image', cv2_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(ofn, cv2_array)


def pr2tensor(the_pr, to_fill_value=0.0):
    padding_np = np.zeros((128, 1024))
    padding_np.fill(to_fill_value)
    the_pr = np.concatenate((padding_np, the_pr, padding_np), axis=0)
    return torch.from_numpy(the_pr[np.newaxis, np.newaxis, :]).type(torch.FloatTensor).cuda()


def show_tensor(the_tensor, ofn=None, crop=False):
    temp = the_tensor.detach().cpu().squeeze().numpy()
    if crop:
        show_img(np.uint8(255.0 * (1.0 - temp))[128:896], ofn)
    else:
        show_img(np.uint8(255.0 * (1.0 - temp)), ofn)


def fetch_IT_input(folder_pos):
    multi_smooth_list = ['smooth_0', 'smooth_1', 'smooth_2', 'smooth_3', 'smooth_4', 'smooth_5']

    depth_path = os.path.join(folder_pos, 'depth.png')
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 65535.0
    dim1 = depth_img.shape[0]
    dim2 = depth_img.shape[1]
    depth_img = depth_img[np.newaxis, :, :] - 0.906764  # subtract average

    nv_img = np.zeros((len(multi_smooth_list), dim1, dim2))
    for smooth_id, smooth_str in enumerate(multi_smooth_list):
        nv_path = os.path.join(folder_pos, '%s.png' % smooth_str)
        nv_img[smooth_id, :, :] = cv2.imread(nv_path, cv2.IMREAD_UNCHANGED) / 65535.0 - 0.974258

    output_np = np.concatenate((nv_img, depth_img), axis=0)
    output_np = output_np[np.newaxis, :, :, :]
    output = torch.from_numpy(output_np).type(torch.cuda.FloatTensor)
    return output


def ridge_detection(img_pro, folder_pos, conf):
    img_cv2 = np.uint8(img_pro * 255.0)
    img_cv2[img_cv2 >= conf['rd_b_t']] = 255
    Hxx, Hxy, Hyy = hessian_matrix(img_cv2, sigma=conf['rd_sigma_para'], order='xy')
    i1, i2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

    i1[i1 <= conf['rd_threshold']] = 0.0
    i1[i1 >= conf['rd_threshold']] = 1.0

    pr = np.uint8(i1 * 255.0)
    kernel = np.ones((conf['rd_kernel_size'], conf['rd_kernel_size']), np.uint8)
    pr = cv2.erode(pr, kernel, iterations=1) / 255.0

    base_fn = os.path.join(folder_pos, 'base.png')
    base_pr = 1.0 - cv2.imread(base_fn, cv2.IMREAD_GRAYSCALE) / 255.0
    pr = np.maximum(pr, base_pr)
    output = pr2tensor(pr)

    return output


def fetch_G_input(folder_pos):
    # ------------read nv------------
    nv_fn = os.path.join(folder_pos, 'smooth_0.png')
    nv_data = cv2.imread(nv_fn, cv2.IMREAD_UNCHANGED) / 65535.0 - 0.974258
    nv_tensor = pr2tensor(nv_data, to_fill_value=1.0 - 0.974258)

    # ------------read depth------------
    depth_fn = os.path.join(folder_pos, 'depth.png')
    depth_data = cv2.imread(depth_fn, cv2.IMREAD_UNCHANGED) / 65535.0 - 0.906764
    depth_tensor = pr2tensor(depth_data, to_fill_value=1.0 - 0.906764)

    # ------------read suggestive contour------------
    sc_fn = os.path.join(folder_pos, 'suggestive_contour.png')
    sc_data = 1.0 - cv2.imread(sc_fn, cv2.IMREAD_GRAYSCALE) / 255.0
    sc_tensor = pr2tensor(sc_data)

    # ------------read sc info------------
    sci_fn = os.path.join(folder_pos, 'suggestive_contour_info.txt')
    with open(sci_fn, 'r') as txt_obj:
        one_line = txt_obj.readline().strip()
    max_dwkr = float(one_line.split(' ')[0])
    sc_fs = float(one_line.split(' ')[1])

    # ------------read sc feature------------
    scf_fn = os.path.join(folder_pos, 'suggestive_contour_feature.png')
    scf_data = cv2.imread(scf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * max_dwkr
    scf_tensor = pr2tensor(scf_data, to_fill_value=max_dwkr)

    # ------------read opengl nv feature------------
    openglnv_fn = os.path.join(folder_pos, 'nv.png')
    openglnv_data = cv2.imread(openglnv_fn, cv2.IMREAD_UNCHANGED) / 65535.0
    openglnv_tensor = pr2tensor(openglnv_data, to_fill_value=1.0)

    # ------------read apr------------
    apr_fn = os.path.join(folder_pos, 'apparent_ridge.png')
    apr_data = 1.0 - cv2.imread(apr_fn, cv2.IMREAD_GRAYSCALE) / 255.0
    apr_tensor = pr2tensor(apr_data)

    # ------------read apr info------------
    apri_fn = os.path.join(folder_pos, 'apparent_ridge_info.txt')
    with open(apri_fn, 'r') as txt_obj:
        one_line = txt_obj.readline().strip()
    apr_max_info = float(one_line.split(' ')[0])

    # ------------read apr feature------------
    aprf_fn = os.path.join(folder_pos, 'apparent_ridge_feature.png')
    aprf_data = cv2.imread(aprf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * apr_max_info
    aprf_tensor = pr2tensor(aprf_data, to_fill_value=apr_max_info)

    # ------------read ridge------------
    ridge_fn = os.path.join(folder_pos, 'ridge.png')
    ridge_data = 1.0 - cv2.imread(ridge_fn, cv2.IMREAD_GRAYSCALE) / 255.0
    ridge_tensor = pr2tensor(ridge_data)

    # ------------read valley------------
    valley_fn = os.path.join(folder_pos, 'valley.png')
    valley_data = 1.0 - cv2.imread(valley_fn, cv2.IMREAD_GRAYSCALE) / 255.0
    valley_tensor = pr2tensor(valley_data)

    # ------------read ridge info------------
    ri_fn = os.path.join(folder_pos, 'ridge_info.txt')
    with open(ri_fn, 'r') as txt_obj:
        one_line = txt_obj.readline().strip()
    r_max_info = float(one_line.split(' ')[0])

    # ------------read valley info------------
    vi_fn = os.path.join(folder_pos, 'valley_info.txt')
    with open(vi_fn, 'r') as txt_obj:
        one_line = txt_obj.readline().strip()
    v_max_info = float(one_line.split(' ')[0])

    # ------------read ridge feature------------
    rf_fn = os.path.join(folder_pos, 'ridge_feature.png')
    rf_data = cv2.imread(rf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * r_max_info
    rf_tensor = pr2tensor(rf_data, to_fill_value=r_max_info)

    # ------------read valley feature------------
    vf_fn = os.path.join(folder_pos, 'valley_feature.png')
    vf_data = cv2.imread(vf_fn, cv2.IMREAD_UNCHANGED) / 65535.0 * v_max_info
    vf_tensor = pr2tensor(vf_data, to_fill_value=v_max_info)

    # ------------comparison data------------
    comparison_data = np.zeros((768, 1024))
    comparison_tensor = pr2tensor(comparison_data)

    # ------------feature size------------
    fs_data = np.zeros((768, 1024))
    fs_data.fill(sc_fs)
    fs_tensor = pr2tensor(fs_data, to_fill_value=sc_fs)

    return nv_tensor, depth_tensor, sc_tensor, openglnv_tensor, scf_tensor, fs_tensor, comparison_tensor, ridge_tensor, rf_tensor, valley_tensor, vf_tensor, apr_tensor, aprf_tensor


