# [Semantic Image Segmentation by Scale-Adaptive Networks](https://speedinghzl.github.io/docs/san.pdf)
By [Zilong Huang](http://speedinghzl.github.io), [Chunyu Wang](https://chunyuwang.netlify.com/), [Xinggang Wang](http://www.xinggangw.info/index.htm), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu) and [Jingdong Wang](https://jingdongwang2017.github.io/).

This code is a implementation of the experiments in the paper **[Semantic Image Segmentation by Scale-Adaptive Networks](https://speedinghzl.github.io/docs/san.pdf)**, which is accepted by Transactions on Image Processing. The code is developed based on the Caffe framework.

### License

SAN is released under the MIT License (refer to the LICENSE file for details).

## Installing dependencies

* **caffe (deeplabv2 version)**: deeplabv2 caffe installation instructions are available at `https://bitbucket.org/aquariusjay/deeplab-public-ver2`. Note, you need to compile **caffe** with python wrapper and support for python layers. Then add the caffe python path into [tools/findcaffe.py](https://github.com/speedinghzl/Scale-Adaptive-Network/blob/master/tools/findcaffe.py#L21).

## Training the SAN model

* Run:

```bash
      $ python tools/train.py --solver YOUR_SOLVER --weight IMAGENET_PRETRAINED_MODEL --gpu GPU_ID
```
The corresponding solver files and input image lists are put in **config** and **list** floders.

## Acknowledgment
The work was mainly done during an internship at [MSRA](https://www.msra.cn/).
