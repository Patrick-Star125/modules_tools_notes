import cv2 as cv
import numpy as np
from handdetection.skin_extraction import Detection

class Edges:

    #计算图像水平与垂直方向上sobel算子梯度图
    def sobel(self):
        sobelx = cv.Sobel(self.gray,cv.CV_64F,1,0)
        sobely = cv.Sobel(self.gray,cv.CV_64F,0,1)
        sobelx = cv.convertScaleAbs(sobelx)
        sobely = cv.convertScaleAbs(sobely)
        dst = cv.addWeighted(sobelx,0.5,sobely,0.5,0)
        return dst

    #边缘检测
    def edgedetction(self,img):
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sobelxy = self.sobel()
        dst = self.luminance(sobelxy)
        return dst

    #亮度改变
    def luminance(self,img):

        blank = np.zeros(img.shape, img.dtype)
        dst = cv.addWeighted(img, 1.8, blank, 1 - 1.8, 1.3)

        return dst

'''
以下是该类实例化测试
'''
def operating(img):
    detection = Detection()
    edges = Edges()
    skin = detection.skin(img)
    dst = edges.edgedetction(skin)
    return dst


if __name__ == '__main__':
    cap_path = 'D:\\cap\\5.旋转.mp4'

    '''
    #img = cv.imread(img_path)


    '''
    cap = cv.VideoCapture(0)
    ret,img = cap.read()

    w = int(cap.get(3))
    h = int(cap.get(4))
    print(h, w)

    fourrc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('hand.avi', fourrc, 25.0, (h, w), False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        dst = operating(frame)
        out.write(frame)
        cv.imshow('cap', dst)
        if cv.waitKey(25) & 0xff == ord('q'):
            break
        prvs = dst

    cap.release()