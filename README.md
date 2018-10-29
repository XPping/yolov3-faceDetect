# yolov3-faceDetect
Used yolov3 to implement face detection  

# Environment
Python3.5+Pytorch4.0  

# Database
FDDB database  
COCO database

# Model
We use yolov3 pretrained in COCO databse to implement face detection in FDDB database. 
We change the 255 of YOLOLayer output which turned out to be 255(3*(1+4+num_class=80)=255) to 18, in face detection num_class=1.
You can load coco detection pretrained weight and face detection pretrained weight from https://drive.google.com/open?id=1u6v0Pg2TAgbpD-s0_Jy8fbB7hR2SHYtf

# Attention
1. We load dataset information from ArgumentParser instead of coco.data.  
2. We change the label rectangle format from (centerx1, centerx2, width, height) to (top-left-x, top-left-y, bottom-right-x, bottom-right-y). Because we generally rect object using top-left and bottom-right as ccordinate. In dataloader.py, we will convert ccordinate to yolov3 needed format.

# Result
In FDDB val dataset, we can get compute average precision is 98%. But face detection doesn't work well. Maybe YOLOLayer we changed not well or overfit used yolov3 for FDDB database.  
![image](https://github.com/XPping/yolov3-faceDetect/raw/master/result/2002_07_19_big_img_209.jpg) 
![image](https://github.com/XPping/yolov3-faceDetect/raw/master/result/2002_07_21_big_img_744.jpg) 
![image](https://github.com/XPping/yolov3-faceDetect/raw/master/result/2002_07_25_big_img_722.jpg) 

# Reference code
https://github.com/eriklindernoren/PyTorch-YOLOv3  

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
```
@article{FDDB,
  title={FDDB: A Benchmark for Face Detection in Unconstrained Settings},
  author={Vidit Jain and Erik Learned-Miller},
  institution =  {University of Massachusetts, Amherst},
  year = {2010},
  number = {UM-CS-2010-009}
```
