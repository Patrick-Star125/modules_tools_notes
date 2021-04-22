![image-20210314173107582](C:\Users\86189\AppData\Roaming\Typora\typora-user-images\image-20210314173107582.png)

![image-20210314173123095](C:\Users\86189\AppData\Roaming\Typora\typora-user-images\image-20210314173123095.png)

### 肤色提取

Detection类的skin()函数

skin（img）传入图片，返回值--肤色提取结果

### 边缘检测

Edges类的edgedetection ()函数

edgedetection (img)，return 边缘检测结果

### 方向判断

Tracking类的pointtracking()函数

pointtracking(img，new_point,old_point)

传入img，（x1，y1），（x2，y2）

在图像上显示方向判定结果



#### Optical_flow

作为主函数运行，图像预处理部分放入per_treatment ()函数

在进行光流法前，需设置特征点检测参数feature_params

以及lk算法参数lk_params

使用goodFeaturesToTrack函数将第一帧图片存入，得到关键点坐标

之后开启视频流

先对图像预处理，再使用calcOpticalFlowPyrLK函数传入当前帧图像和上一帧图像，以及上一帧点的坐标

进入循环，将跟踪的点标记在图像上，传入pointtracking函数对方向判定

最后将上一帧图像和点赋给old_gray和p0（存储上一帧图像和点坐标）





