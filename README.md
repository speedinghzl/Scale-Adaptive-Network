# Semantic Image Segmentation by Scale-Adaptive Networks

This code is a implementation of the experiments in the paper Semantic Image Segmentation by Scale-Adaptive Networks. The code is developed based on the Caffe framework.

### License

SAN is released under the MIT License (refer to the LICENSE file for details).

## Installing dependencies

* **caffe (deeplabv2 version)**: deeplabv2 caffe installation instructions are available at `https://bitbucket.org/aquariusjay/deeplab-public-ver2`. Note, you need to compile **caffe** with python wrapper and support for python layers. Then add the caffe python path into [training/tools/findcaffe.py](https://github.com/speedinghzl/DSRG/blob/master/training/tools/findcaffe.py#L21).

## Training the SAN model
