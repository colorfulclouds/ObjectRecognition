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


fine-tuning


场景识别

http://places.csail.mit.edu



视觉风格识别

http://demo.vislab.berkeleyvision.org

论文地址：https://arxiv.org/pdf/1704.04861.pdf
个人实现项目：
https://github.com/rcmalli/keras-mobilenet
https://github.com/pby5/MobileNet

MobileNet 是谷歌在 2017 年 4 月发表的一项研究，它是一种高效、小尺寸的神经网络架构，适用于构建手机/移动设备上的低延迟深度学习应用，并可以完成多种不同任务。

Core ML 实现链接：https://github.com/hollance/MobileNet-CoreML

本实现是论文《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》中神经网络架构 MobileNet 的苹果 CoreML 框架实现。它使用了 MobileNet-Caffe 中的预训练内容。

想使用这个 app，请在 Xcode 9 中打开 MobileNetCoreML.xcodeproj，并在 iOS11 系统或同版本的模拟器中运行。目前，它只能用于识别猫的图片，实时视频的识别将在稍后加入（可以看看 Forge：https://github.com/hollance/Forge，一个用于 iOS10 的 Metal 神经网络工具包，和 MobileNet 共同使用可以处理视频）。


模型迁移

这个版本已经包含了完整的 MobileNet.mlmodel，所以你不必遵循这一章节的所有步骤。当然，如果你希望尝试，以下是如何将原版 Caffe 模型转换到.mlmodel 文件中的方法：

1. 从 Caffemodel 文件中下载模型，放到本项目中的根目录下。（注意，你不必下载 mobilenet_deploy.prototxt，它在本项目中已经存在。该实现作者还在尾部加入了一个原版缺失的 Softmax 层）

原 Caffe 实现连接：https://github.com/shicai/MobileNet-Caffe

2. 在终端加入如下代码：
$ virtualenv -p /usr/bin/python2.7 env
$ source env/bin/activate
$ pip install tensorflow
$ pip install keras==1.2.2
$ pip install coremltools
使用/usr/bin/python2.7 设置虚拟环境非常重要，如果你在使用其他版本的 Python，则转换脚本会崩溃——Fatal Python error: PyThreadState_Get: no current thread；同时，你需要使用 Keras 1.2.2 版本，而不是更新的 2.0 版。

3. 运行 coreml.py 脚本进行转换：
$ python coreml.py
这会生成 MobileNet.mlmodel 文件。

4. 通过禁止 virtualenv 进行清理：
$ deactivate

论文：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

论文链接：https://arxiv.org/abs/1704.04861v1

https://github.com/tensorflow/models/tree/master/inception#geitting-started


https://github.com/balancap/SSD-tensorflow


https://github.com/Zehaos/MobileNet


https://github.com/rbgirshick/py-faster-rcnn  安装caffe2


http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/detection.ipynb