# ArtisticStyleConversion_TensorFlow

Image style conversion by\
Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge, "Image style transfer using convolutional neural networks"\
https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html

## Need Packages
Python 3.5\
tensorflow 1.11.0\
numpy 1.15.4\
scipy 1.1.0

[for GPU]\
tensorflow-gpu 1.11.0\
CUDA 9.0\
cuDNN v7.4.1
##### Note: This is environment in my experiment. Some other environments maybe work.

## Model
This style conversion uses VGG-19 Network.\
Go to http://www.vlfeat.org/matconvnet/pretrained/ and download __imagenet-vgg-verydeep-19.mat__\
Put it as below
```buildoutcfg
ArtisticStyleConversion/
┣ .gitignore
┣ .git/
┣ models/
┃  ┗ imagenet-vgg-verydeep19.mat
┗ images/
```

## How to use
Execute as below
```buildoutcfg
python ArtisticStyleCoversion.py [content-image-format] [style-image-format] [result-image-format(optional)] [process-output(optional)]
```
* __\[content-image-format\] : path of content-image__
* __\[style-image-format\]&emsp;&nbsp;&nbsp;: path of style-image__
* __\[result-image-format\]&emsp;&thinsp;: result-image name (default: result.png)__
* __\[process-output\]&emsp;&emsp;&emsp;&thinsp;: bool of process image output (default: 0)__

## Example
```buildoutcfg
python ArtisticStyleConversion.py images\eiffel.jpg images\starrynight.jpg result.png 1
```
