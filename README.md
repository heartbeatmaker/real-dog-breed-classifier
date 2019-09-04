Based on https://github.com/skmatz/dog-breed-classifier



# Dog Breed Classifier

## Overview
Detect dogs in real time and classify breeds.  

## Usage
```
pip3 install -r requirements.txt

1. 'app.py' deals with videos or webcam
python3 app.py [label=path/to/label] [model=path/to/model] [video=path/to/video]

2. 'process_image.py' deals with images
python3 process_image.py [video=path to image directory]
ex) python process_image.py --video ./screenshot_output/
```

## Model File
Download from below.
* [model_8.h5](https://www.dropbox.com/s/ol3w28b8onl23xa/model_8.h5?dl=0)  
* [model_16.h5](https://www.dropbox.com/s/btpeb738uk3mikq/model_16.h5?dl=0)  
* [yolo.h5](https://www.dropbox.com/s/kozt3gbk5l5ucde/yolo.h5?dl=0)  


## Dataset
* [Dog Breed Identification - Kaggle](https://www.kaggle.com/c/dog-breed-identification)
