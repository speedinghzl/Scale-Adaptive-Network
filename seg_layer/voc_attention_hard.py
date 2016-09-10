# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import cv2
import caffe

import numpy as np
import os.path as osp

from random import shuffle
from PIL import Image
import random, copy
from voc import load_pascal_annotation

class ImageSegDataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a Detection model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label_1s', 'label_2s', 'label_3s', 'label', 'attention']

        # === Read input parameters ===
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        SimpleTransformer.check_params(params)
        # store input as class variables
        self.batch_size = params['batch_size']
        self.input_shape = params['crop_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, self.input_shape[0], self.input_shape[1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])

        top[2].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])
            
        top[3].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])
            
        top[4].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])

        top[5].reshape(
            self.batch_size, 1, self.input_shape[0], self.input_shape[1])

        print_info("ImageSegDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label_1s, label_2s, label_3s, label, label_at = self.batch_loader.perpare_next_data()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label_1s
            top[2].data[itt, ...] = label_2s
            top[3].data[itt, ...] = label_3s
            top[4].data[itt, ...] = label
            top[5].data[itt, ...] = label_at

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.root_folder = params['root_folder']
        self.source = params['source']
        self.voc_dir = params['voc_dir']
        # get list of image indexes.
        self.indexlist = [line.strip().split() for line in open(self.source)]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(params)

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def perpare_next_data(self):
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_path, label_file_path = index
        image = cv2.imread(self.root_folder+image_file_path, cv2.IMREAD_COLOR)
        label = cv2.imread(self.root_folder+label_file_path, cv2.IMREAD_GRAYSCALE)
        img_id = osp.splitext(osp.basename(label_file_path))[0] 
        annotation = load_pascal_annotation(img_id, self.voc_dir, False)

        self._cur += 1
        return self.transformer.preprocess(image, label, annotation)


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, params):
        import pickle
        SimpleTransformer.check_params(params)
        self.mean = params['mean']
        self.is_mirror = params['mirror']
        self.crop_h, self.crop_w = params['crop_size']
        self.scale = params['scale']
        self.phase = params['phase']
        self.ignore_label = params['ignore_label']

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def load_object(self, filename):
        with open(filename, 'rb') as input:
            obj = pickle.load(input)
        return obj

    def get_threshold(self, cls_id):
        return self.scale_threshold[self.classes[cls_id]]

    def generate_scale_label(self, image, label, annotation):
        boxes = annotation['boxes']
        gt_classes = annotation['gt_classes']
        annos = zip(boxes, gt_classes)
        #base, ran = self.generate_scale_range(boxes)
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        #label_1s, label_2s, label_3s = copy.copy(label), copy.copy(label), copy.copy(label)
        label_1s, label_2s, label_3s = np.full(label.shape, self.ignore_label, dtype=np.uint8), np.full(label.shape, self.ignore_label, dtype=np.uint8), np.full(label.shape, self.ignore_label, dtype=np.uint8)
        # label_1s, label_2s, label_3s, label_at = np.zeros_like(label), np.zeros_like(label), np.zeros_like(label), np.zeros_like(label)
        # label_1s[label==255] = 255; label_2s[label==255] = 255; label_3s[label==255] = 255
        label_at = np.full(label.shape, self.ignore_label, dtype=np.uint8)

        def fitness(annos_item):
            x1, y1, x2, y2 = annos_item[0]
            sq = (x2 - x1) * (y2 - y1)
            return sq
        annos = sorted(annos, key=fitness, reverse=True)

        for box, cls_id in annos:
            box = np.array(box) * f_scale
            x1, y1, x2, y2 = box
            sq = (x2 - x1) * (y2 - y1)
            s1 = label_1s[y1:y2, x1:x2]
            s2 = label_2s[y1:y2, x1:x2]
            s3 = label_3s[y1:y2, x1:x2]
            s0 = label[y1:y2, x1:x2]
            at = label_at[y1:y2, x1:x2]

            index = (s0==cls_id)
            if sq < 12544:
                s1[:] = s0[:]
                # s2[index] = 255
                # s3[index] = 255
                at[index] = 1
            elif sq >= 12544 and sq < 50176:
                # s1[index] = 255
                s2[:] = s0[:]
                # s3[index] = 255
                at[index] = 2
            elif sq >= 50176:
                # s1[index] = 255
                # s2[index] = 255
                s3[:] = s0[:]
                at[index] = 3

        #at[s0==255] = label[s0==255]
        # cv2.imshow('image  ', image)
        # cv2.imshow('label 1', label_1s)
        # cv2.imshow('label 2', label_2s)
        # cv2.imshow('label 3', label_3s)
        # cv2.waitKey()
        # a = random.randint(0,10000)
        # cv2.imwrite('temp/'+ str(a) + '.png', label)
        # cv2.imwrite('temp/'+ str(a) + '_s.png', label_1s)
        # cv2.imwrite('temp/'+ str(a) + '_m.png', label_2s)
        # cv2.imwrite('temp/'+ str(a) + '_l.png', label_3s)
        return image, label_1s, label_2s, label_3s, label, label_at

    def preprocess(self, image, label, boxes):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """
        # image = cv2.convertTo(image, cv2.CV_64F)
        image, label_1s, label_2s, label_3s, label, label_at = self.generate_scale_label(image, label, boxes)
        image = np.asarray(image, np.float32)
        image -= self.mean
        image *= self.scale
        
        img_h, img_w = label_1s.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_1s_pad = cv2.copyMakeBorder(label_1s, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            label_2s_pad = cv2.copyMakeBorder(label_2s, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            label_3s_pad = cv2.copyMakeBorder(label_3s, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            label_at_pad = cv2.copyMakeBorder(label_at, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_1s_pad, label_2s_pad, label_3s_pad, label_pad, label_at_pad = image, label_1s, label_2s, label_3s, label, label_at

        img_h, img_w = label_1s_pad.shape
        if self.phase == 'Train':
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = (img_h - self.crop_h) / 2
            w_off = (img_w - self.crop_w) / 2

        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)
        label_1s = np.asarray(label_1s_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)
        label_2s = np.asarray(label_2s_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)
        label_3s = np.asarray(label_3s_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)
        label    = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)
        label_at = np.asarray(label_at_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w].copy(), np.float32)

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label_1s = label_1s[:, ::flip]
            label_2s = label_2s[:, ::flip]
            label_3s = label_3s[:, ::flip]
            label    = label[:, ::flip]
            label_at    = label_at[:, ::flip]

        return image, label_1s, label_2s, label_3s, label, label_at

    @classmethod
    def check_params(cls, params):
        if 'crop_size' not in params:
            params['crop_size'] = (505, 505)
        if 'mean' not in params:
            params['mean'] = [128, 128, 128]
        if 'scale' not in params:
            params['scale'] = 1.0
        if 'mirror' not in params:
            params['mirror'] = False
        if 'phase' not in params:
            params['phase'] = 'Train'
        if 'ignore_label' not in params:
            params['ignore_label'] = 255

def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['source'],
        params['batch_size'],
        params['crop_size'])


if __name__ == '__main__':
    params = {'batch_size': 2,
              'mean': (104.008, 116.669, 122.675),
              'root_folder': 'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/',
              'source': 'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/list/train_3s.txt',
              'mirror': True,
              'crop_size': (505, 505)}
    t = SimpleTransformer(params)

    image = Image.open(r'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/images/2008_000003.jpg')
    label = Image.open(r'D:/v-zihuan/segmentation_with_scale/experiment/voc_part/data/part_mask_scale_3/2008_000003.png')
    t.preprocess(image, label)
