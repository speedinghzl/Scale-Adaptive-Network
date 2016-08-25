# imports
import scipy.io


import numpy as np
import os.path as osp

part2ind = {}
part2ind['head'] = 1
part2ind['leye'] = 1
part2ind['reye'] = 1
part2ind['lear'] = 1
part2ind['rear'] = 1
part2ind['lebrow'] = 1
part2ind['rebrow'] = 1
part2ind['nose'] = 1
part2ind['mouth'] = 1
part2ind['hair'] = 1

part2ind['torso'] = 2
part2ind['neck'] = 2
part2ind['llarm'] = 4
part2ind['luarm'] = 3
part2ind['lhand'] = 4
part2ind['rlarm'] = 4
part2ind['ruarm'] = 3
part2ind['rhand'] = 4

part2ind['llleg'] = 6
part2ind['luleg'] = 5
part2ind['lfoot'] = 6
part2ind['rlleg'] = 6
part2ind['ruleg'] = 5
part2ind['rfoot'] = 6

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (cmin, rmin, cmax, rmax)

def show_result(gt):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    fig, axes = plt.subplots(1, 1)
    ax1 = axes
    fig.set_size_inches(10, 8, forward=True)

    # make a color map of fixed colors
    cmap = colors.ListedColormap([(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0), (0,0,0.5), (0.5,0,0.5), (0,0.5,0.5)])
    bounds=[0,1,2,3,4,5,6,7]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)
    plt.show()

def load_human_annotation(index, human_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    filename = osp.join(human_root, 'Annotations_Part', index + '.mat')
    # print 'Loading: {}'.format(filename)

    anno = scipy.io.loadmat(filename)['anno']
    # anno = scipy.io.loadmat(filename)[0]
    ins_list = anno[0][0][1][0]

    instances = []
    boxes = []

    for ins in ins_list:
        if not ins[1] == 15:
            continue
        instances.append(ins[2])
        boxes.append(bbox2(ins[2]))
    
    return {'instances': instances,
            'boxes': boxes,
            'flipped': False,
            'index': 15}

def load_human_part_annotation(index, human_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    filename = osp.join(human_root, 'Annotations_Part', index + '.mat')
    # print 'Loading: {}'.format(filename)

    anno = scipy.io.loadmat(filename)['anno']
    # anno = scipy.io.loadmat(filename)[0]
    ins_list = anno[0][0][1][0]

    instances = []
    boxes = []

    for ins in ins_list:
        if not ins[1] == 15 or len(ins[3]) == 0:
            continue
        human_part = np.zeros_like(ins[2])
        part_list = ins[3][0]
        for part in part_list:
            ind = part2ind[part[0][0]]
            human_part[part[1] == 1] = ind
        instances.append(human_part)
        boxes.append(bbox2(human_part))
    
    return {'instances': instances,
            'boxes': boxes,
            'flipped': False,
            'index': 15}


if __name__ == '__main__':
    img1 = np.zeros((20,20))
    img1[5:15,10:20] = 1.

    print bbox2(img1)