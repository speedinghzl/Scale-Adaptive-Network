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
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
              'crop_size': (505, 505)}
        self.transformer = SimpleTransformer(self.params)
        self.mode = mode

    def process(self, image_path):
        im, data = self.transform(image_path)
        self.net.blobs['data'].reshape(1, *(data.shape))
        self.net.blobs['data'].data[0, ...] = data
        output = self.net.forward()
        # print 'mean ', np.mean(output['interp_1s'][0][:]), np.mean(output['interp_2s'][0][:]), np.mean(output['interp_3s'][0][:])
        # print 'scale ', np.linalg.norm(output['interp_1s'][0][:]), np.linalg.norm(output['interp_2s'][0][:]), np.linalg.norm(output['interp_3s'][0][:])
        # print 'max ', np.max(output['interp_1s'][0][:]), np.max(output['interp_2s'][0][:]), np.max(output['interp_3s'][0][:])
        # print 'min ', np.min(output['interp_1s'][0][:]), np.min(output['interp_2s'][0][:]), np.min(output['interp_3s'][0][:])
        output_fl = output['interp'][0]
        output_1s = output['interp_1s'][0]
        output_2s = output['interp_2s'][0]
        output_3s = output['interp_3s'][0]

        print output_1s.shape
        img_h, im_w, _ = im.shape
        result = np.argmax(output_fl,0)
        output_fl = np.max(output_fl[1:,:,:], 0)
        output_1s = np.max(output_1s[1:,:,:], 0)
        output_2s = np.max(output_2s[1:,:,:], 0)
        output_3s = np.max(output_3s[1:,:,:], 0)
        #result = np.ceil(result/3)
        print output_1s.shape
        print(im.shape)
        print(result.shape)
        return self.handle_three_branches(im, result[0:img_h, 0:im_w], output_fl[0:img_h, 0:im_w], output_1s[0:img_h, 0:im_w], output_2s[0:img_h, 0:im_w], output_3s[0:img_h, 0:im_w])

    def transform(self, image_path):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im_processed = self.transformer.preprocess(im)
        return im, im_processed

    def handle_three_branches(self, im, result, final, op_1s, op_2s, op_3s):
        fig, axes = plt.subplots(2, 3)
        (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
        fig.set_size_inches(16, 8, forward=True)

        ax1.set_title('im')
        ax1.imshow(im)

        # make a color map of fixed colors
        cmap = colors.ListedColormap([(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0), (0,0,0.5), (0.5,0,0.5), (0,0.5,0.5)])
        bounds=[0,1,2,3,4,5,6,7]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax2.set_title('prediction')
        ax2.imshow(result, cmap=cmap, norm=norm)

        ax3.set_title('branch_1s')
        im1 = ax3.imshow(op_1s)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax3)

        ax4.set_title('branch_2s')
        im2 = ax4.imshow(op_2s)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax4)

        ax5.set_title('branch_3s')
        im3 = ax5.imshow(op_3s)
        divider5 = make_axes_locatable(ax5)
        cax5 = divider5.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax5)

        ax6.set_title('final output')
        fl = ax6.imshow(final, vmin=0, vmax=1)
        divider6 = make_axes_locatable(ax6)
        cax6 = divider6.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(fl, cax=cax6)

        #plt.show()
        return fig

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
        #cv2.imwrite(save_img_path, Test.process(test_img_path))
        Test.process(test_img_path).savefig(save_img_path, dpi=100)
