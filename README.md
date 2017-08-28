R-CNN

Fast R-CNN

Faster R-CNN

ION


HyperNet


SDP-CRC


YOLO


G-CNN


SSD


目标检测，通俗的理解其目的是定位目标，并确定目标位置及大小。相比于图像目标检查，视频检测是比单张图片检测多了时间上下文的信息；相比于传统目标检测，深度学习目标检测算法越来越成熟，那么到底在目标检测上有哪些可用的深度学习算法，哪些算法又比较适合我们的工作或者研究呢？深思考小编带你一起了解目标检测算法！
目标检测算法论文集
[overfeat] http://arxiv.org/abs/1312.6229
[RCNN] http://arxiv.org/abs/1311.2524
[SPP-Net] http://arxiv.org/pdf/1406.4729.pdf
[Fast-RCNN] http://arxiv.org/abs/1504.08083
[R-FCN]  https://arxiv.org/abs/1605.06409
[Faster-RCNN] http://arxiv.org/abs/1506.01497
[YOLO] http://arxiv.org/abs/1506.02640
[YOLO2] https://arxiv.org/pdf/1612.08242v1.pdf
[SSD] http://arxiv.org/pdf/1512.02325v1.pdf
[HyperNet] https://www.arxiv.org/abs/1604.00600
[MR-CNN] http://arxiv.org/abs/1505.01749
[Inside-Outside Net] http://120.52.73.9/
[LocNet] https://arxiv.org/abs/1511.07763
[G-CNN] https://arxiv.org/abs/1512.07729
[MASK-RCNN] http://arxiv.org/abs/1605.02319
 
目标检测算法性能大比拼
目标检测算法实际应用中的场景，无外乎关心MAP和检测耗时两个指标，针对上述一系列的算法，下表给出其对应的结果，其中+++表示训练数据为VOC07+VOC12+MS COCO。其余方法的训练数据均为VOC07 for VOC07 test，VOC07+VOC12 for VOC12test，MS COCO for MS COCO test.
方法	检测耗时	
VOC07
VOC12
MS   COCO
overfeat
----
----
----
----
RCNN
13s
66.0
53.3
----
SPP-Net
0.29s
59.2
----
----
fast-rcnn
0.32s
70.0
68.0
19.7
faster-rcnn+++
140ms
85.6
83.8
21.9
HyperNet
1140ms
76.3
71.4
----
MR-CNN
30s
78.2
73.9
----
R-FCN+++
0.17s
83.6
82.0
29.9
ION
0.8s
79.2
76.4
33.1
视频算法
--
--
--
--
YOLO
45FPS
63.4
57.9
----
Fast  YOLO
155FPS
52.7
----
----
SSD300
58FPS
72.1
70.3
20.8
SSD500
23FPS
75.1
73.1
24.4

目标检测算法源码大集合
[overfeat] https://github.com/sermanet/OverFeat
[RCNN] https://github.com/rbgirshick/rcnn
[SPP-Net] https://github.com/dsisds/caffe-SPPNet 
[Fast-RCNN] https://github.com/rbgirshick/fast-rcnn 
[R-FCN]  https://github.com/daijifeng001/R-FCN
[Faster-RCNN] https://github.com/rbgirshick/py-faster-rcnn
[YOLO] https://github.com/nilboy/tensorflow-yolo
[YOLO2] https://github.com/pjreddie/darknet
[SSD] https://github.com/weiliu89/caffe/tree/ssd
