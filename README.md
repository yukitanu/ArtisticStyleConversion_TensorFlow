# ArtisticStyleConversion_TensorFlow

Image style conversion by\
[Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge, "Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

## Requirements
- Python3
- tensorflow 1.11.0
- numpy 1.15.4
- scipy 1.1.0

[for GPU]
- tensorflow-gpu 1.11.0
- CUDA 9.0
- cuDNN v7.4.1

## Model
This style conversion uses VGG-19 Network.
Go to http://www.vlfeat.org/matconvnet/pretrained/ and download __imagenet-vgg-verydeep-19.mat__, and put it as below.
```
ArtisticStyleConversion/
├ .gitignore
├ .git/
├ models/
|  └ imagenet-vgg-verydeep19.mat
└ images/
```

## How to use
```shell-session
python ArtisticStyleConversion.py [-h] [--output OUTPUT]
                                  [--output_process OUTPUT_PROCESS]
                                  content_image style_image
```
- `content_image`: Path to content image (str)
- `style_image`: Path to style image (str)
- `--output`: Path to result image (str)
- `--output_process`: Whether to save the intermediate images (bool)

## Example
```shell-session
python ArtisticStyleConversion.py images/eiffel.jpg images/starrynight.jpg result.png 1
```
