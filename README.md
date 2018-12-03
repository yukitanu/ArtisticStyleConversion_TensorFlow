# ArtisticStyleConversion_TensorFlow

Image style conversion by\
"Image style transfer using convolutional neural networks"\
Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge.\
https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html

## Need Packages
Python ver.3.5\
TensorFlow ver.1.11.0\
numpy ver.1.15.4\
scipy ver.1.1.0
#####Note: This is my environment. Some other environments maybe work.

## Need Download
This style conversion uses VGG-19 Network.\
Go to http://www.vlfeat.org/matconvnet/pretrained/ and download __imagenet-vgg-verydeep-19__.\
Then, put it into ArtisticStyleConversion_TensorFlow/models/

## Images
Put style-image and content-image and write these paths as CONTENT_IMG, STYLE_IMG in ArtisticStyleConversion.py.\
Output image path is OUTPUT_DIR/OUTPUT_IMG.
