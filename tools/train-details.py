#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
import findcaffe
import caffe
import argparse
import pprint
import numpy as np
import sys
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, pretrained_model=None):
        """Initialize the SolverWrapper."""

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

    def train_model(self):
        """Network training loop."""
        #self.solver.solve()
        while True:
            self.solver.step(1)
            blobs = self.solver.net.blobs 
            print 'max ', np.max(blobs['fc2_part'].data[:]), np.max(blobs['fc2_part'].data[:])
            print 'min ', np.min(blobs['fc2_part'].data[:]), np.min(blobs['fc2_part'].data[:])

            print 'mean ', np.mean(blobs['fc2_part_fc2_part_0_split_0'].diff[:]), np.mean(blobs['fc2_part_fc2_part_0_split_1'].diff[:]), np.mean(blobs['res5c'].diff[:])
            print 'scale ', np.linalg.norm(blobs['fc2_part_fc2_part_0_split_0'].diff[:]), np.linalg.norm(blobs['fc2_part_fc2_part_0_split_1'].diff[:]), np.linalg.norm(blobs['res5c'].diff[:])
            print 'max ', np.max(blobs['fc2_part_fc2_part_0_split_0'].diff[:]), np.max(blobs['fc2_part_fc2_part_0_split_1'].diff[:]), np.max(blobs['res5c'].diff[:])
            print 'min ', np.min(blobs['fc2_part_fc2_part_0_split_0'].diff[:]), np.min(blobs['fc2_part_fc2_part_0_split_1'].diff[:]), np.min(blobs['res5c'].diff[:])
            print '-------------'
            # print 'mean ', np.mean(blobs['gate_3s'].data[:]), np.mean(blobs['gate_2s'].data[:]), np.mean(blobs['gate_1s'].data[:])
            # print 'scale ', np.linalg.norm(blobs['gate_3s'].data[:]), np.linalg.norm(blobs['gate_2s'].data[:]), np.linalg.norm(blobs['gate_1s'].data[:])
            # print 'max ', np.max(blobs['gate_3s'].data[:]), np.max(blobs['gate_2s'].data[:]), np.max(blobs['gate_1s'].data[:])
            # print 'min ', np.min(blobs['gate_3s'].data[:]), np.min(blobs['gate_2s'].data[:]), np.min(blobs['gate_1s'].data[:])
            print '*************' *10


class Arg(object):
    pass


if __name__ == '__main__':
    args = parse_args()

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    sw = SolverWrapper(args.solver,
                       pretrained_model=args.pretrained_model)

    print 'Solving...'
    sw.train_model()
    print 'done solving'
