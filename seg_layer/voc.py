# imports
import scipy.misc


import numpy as np
import os.path as osp

from xml.dom import minidom
import cv2




def load_pascal_annotation(index, pascal_root, is_only_human=True):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = []
    gt_classes = []
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        if is_only_human and (not cls == 15):
           continue
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        
        boxes.append([x1, y1, x2, y2])
        gt_classes.append(cls)


    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'flipped': False,
            'index': index}


if __name__ == '__main__':
    a = [[1,3,4,5], [1,2,3,4], [3,2,5,6]]
    def fitness(item):
        x1, y1, x2, y2 = item
        sq = (x2 - x1) * (y2 - y1)
        print sq
        return sq
    a = sorted(a, key=fitness, reverse=True)
    print a