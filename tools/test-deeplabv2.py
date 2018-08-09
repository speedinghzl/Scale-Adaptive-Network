#!/usr/bin/env python

# --------------------------------------------------------
# Scale-aware deeplab
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zilong Huang
# --------------------------------------------------------

import findcaffe
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
import cv2
from seg_layer.layer import SimpleTransformer


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='net',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--test_ids', dest='test_ids',
                        help='solver prototxt',
                        default='list/test_id.txt', type=str)
    parser.add_argument('--images_dir', dest='images_dir',
                        help='images dir',
                        default='../../dataset/data/images', type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='images dir',
                        default=None, type=str)                    
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--mode', dest='mode',
                        help='faeture mode',
                        default=0, type=int)
                        
                        
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class TestWrapper(object):
    """A simple wrapper around Caffe's Test.
    This wrapper gives us control over he snapshotting process
    """
    def __init__(self, net_prototxt, model_weights=None, mode=0):
        """Initialize the TestWrapper."""

        self.net = caffe.Net(net_prototxt,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)
        self.params = {'batch_size': 1,
              'mean': (104.008, 116.669, 122.675),
              'mirror': False,
              'crop_size': (481, 481)}
        self.transformer = SimpleTransformer(self.params)
        self.mode = mode

    def process(self, image_path):
        im, data = self.transform(image_path)
        # self.net.blobs['data'].reshape(1, *(data.shape))
        self.net.blobs['data'].data[0, ...] = data
        output = self.net.forward()
        # print 'mean ', np.mean(output['interp_1s'][0][:]), np.mean(output['interp_2s'][0][:]), np.mean(output['interp_3s'][0][:])
        # print 'scale ', np.linalg.norm(output['interp_1s'][0][:]), np.linalg.norm(output['interp_2s'][0][:]), np.linalg.norm(output['interp_3s'][0][:])
        # print 'max ', np.max(output['interp_1s'][0][:]), np.max(output['interp_2s'][0][:]), np.max(output['interp_3s'][0][:])
        # print 'min ', np.min(output['interp_1s'][0][:]), np.min(output['interp_2s'][0][:]), np.min(output['interp_3s'][0][:])
        if self.mode == 0:
            output = output['interp'][0]
        elif self.mode == 1:      
            output = output['interp_1s'][0]
        elif self.mode == 2:       
            output = output['interp_2s'][0]
        elif self.mode == 3:       
            output = output['interp_3s'][0]

        img_h, im_w, _ = im.shape
        result = np.argmax(output,0)
        print result.shape
        result = cv2.resize(result, (500, 500), interpolation = cv2.INTER_NEAREST)
        #result = np.ceil(result/3)
        print(im.shape)
        print(result.shape)
        return result[0:img_h, 0:im_w]

    def transform(self, image_path):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)

        f_scale = self.params['crop_size'][0]/500.0
        im_scale = cv2.resize(im, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)

        im_processed = self.transformer.pre_test_image(im_scale)
        return im, im_processed


if __name__ == '__main__':
    args = parse_args()

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    Test = TestWrapper(args.net, args.pretrained_model, args.mode)
    test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == '']
    image_dir = args.images_dir
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img_id in test_ids:
        test_img_path = os.path.join(image_dir, img_id+'.jpg')
        print(test_img_path)
        save_img_path = os.path.join(save_dir, img_id+'.png')
        cv2.imwrite(save_img_path, Test.process(test_img_path))
