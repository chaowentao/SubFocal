# -*- coding: utf-8 -*-
''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)

If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80

'''

import numpy as np
import os
import time
from LF_func.func_pfm import write_pfm, read_pfm
from LF_func.func_makeinput import make_epiinput
from LF_func.func_makeinput import make_input
from LF_func.func_model_sub_js import define_SubFocal

import matplotlib.pyplot as plt
import cv2
import imageio

if __name__ == '__main__':

    # Input : input_Cam000-080.png
    # Depth output : image_name.pfm

    dir_output = 'SubFocal_sub0.5_js_0.1_e10_export'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # GPU setting ( rtx 3090 - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    '''
    /// Setting 1. LF Images Directory

    LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                                   "A Dataset and Evaluation Methodology for 
                                   Depth Estimation on 4D Light Fields".
                                   http://hci-lightfield.iwr.uni-heidelberg.de/

    '''
    LFdir = 'synthetic'

    if (LFdir == 'synthetic'):
        dir_LFimages = [
            'hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots',
            'hci_dataset/stratified/pyramids',
            'hci_dataset/stratified/stripes', 'hci_dataset/training/boxes',
            'hci_dataset/training/cotton', 'hci_dataset/training/dino',
            'hci_dataset/training/sideboard'
        ]

        image_w = 512
        image_h = 512

    # number of views ( 0~8 for 9x9 )
    AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    path_weight = '/root/SubFocal/LF_checkpoint/SubFocal_sub_0.5_js_0.1_ckp/iter0010_valmse0.768_bp1.93.hdf5'

    img_scale = 1  # 1 for small_baseline(default) <3.5px,
    # 0.5 for large_baseline images   <  7px

    img_scale_inv = int(1 / img_scale)
    ''' Define Model ( set parameters )'''

    model_learning_rate = 0.001
    model_512 = define_SubFocal(round(img_scale * image_h),
                                round(img_scale * image_w), AngualrViews,
                                model_learning_rate)
    ''' Model Initialization '''

    model_512.load_weights(path_weight)
    dum_sz = model_512.input_shape[0]
    dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    tmp_list = []
    for i in range(81):
        tmp_list.append(dum)
    dummy = model_512.predict(tmp_list, batch_size=1)

    avg_attention = []
    """  Depth Estimation  """
    for image_path in dir_LFimages:
        val_list = make_input(image_path, image_h, image_w, AngualrViews)

        start = time.time()

        # predict
        val_output_tmp, _ = model_512.predict(val_list, batch_size=1)

        runtime = time.time() - start
        print("runtime: %.5f(s)" % runtime)

        # save .pfm file
        imageio.imsave(dir_output + '/%s.jpg' % (image_path.split('/')[-1]),
                       val_output_tmp[0, :, :])
        write_pfm(val_output_tmp[0, :, :],
                  dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        print('pfm file saved in %s/%s.pfm' %
              (dir_output, image_path.split('/')[-1]))
    """ Calculate error for pre-trained model """
    output_stack = []
    gt_stack = []
    for image_path in dir_LFimages:
        output = read_pfm(dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        gt = read_pfm(image_path + '/gt_disp_lowres.pfm')
        gt_490 = gt[15:-15, 15:-15]
        output_stack.append(output)
        gt_stack.append(gt_490)
    output = np.stack(output_stack, 0)

    gt = np.stack(gt_stack, 0)

    output = output[:, 15:-15, 15:-15]

    # train_diff = np.abs(output - gt)
    # train_bp = (train_diff >= 0.07)

    # training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    # training_bad_pixel_ratio = 100 * np.average(train_bp)

    # print('Pre-trained Model average MSE*100 = %f' %
    #       training_mean_squared_error_x100)
    # print('Pre-trained Model average Badpix0.07 = %f' %
    #       training_bad_pixel_ratio)

    train_diff = np.abs(output - gt)
    train_bp = (train_diff >= 0.07)
    train_mse = np.square(train_diff)
    mse_list = []
    bp_list = []
    for i in range(len(train_mse)):
        mse = 100 * np.mean(train_mse[i])
        bp = 100 * np.mean(train_bp[i])
        mse_list.append(mse)
        bp_list.append(bp)
        print("%s: mse: %f bp: %f" % (dir_LFimages[i], mse, bp))
    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    training_bad_pixel_ratio = 100 * np.average(train_bp)
    print("mean mse: %f mean bp: %f" % (np.mean(mse_list), np.mean(bp_list)))
    print('Pre-trained Model average MSE*100 = %f' %
          training_mean_squared_error_x100)
    print('Pre-trained Model average Badpix0.07 = %f' %
          training_bad_pixel_ratio)
