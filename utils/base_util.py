import torch
import numpy as np
import cv2


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



