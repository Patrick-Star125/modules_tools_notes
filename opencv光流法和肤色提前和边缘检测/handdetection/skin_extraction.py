import cv2 as cv
import numpy as np


class Detection:

    def __init__(self):

        return

    #肤色提取
    def skin(self,img):
        frame = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        (y, cr, cb) = cv.split(frame)
        cr1 = cv.GaussianBlur(cr, (5, 5), 0)  # 高斯模糊去噪
        # 二值化的目的是将红色区域二值化变成白色，其他区域变为黑色
        ret, skin = cv.threshold(cr1, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)

        # cv.imshow('skin',skin)
        res = cv.bitwise_and(img, img, mask=skin)  # 将色彩空间与原图进行与运算，保留检测到的色彩空间
        return res

    #canny边缘检测
    def Canny(self,img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        edges = cv.Canny(binary, 70, 100)
        # cv.imshow('edges',h)
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        mask = np.ones(img.shape, np.uint8)
        dst = cv.polylines(mask, contours, -1, (255, 255, 255), 4)  # 画线

        return dst

    #形态学处理
    def morphology(self,img):
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        openging = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel, iterations=3)
        closing = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel, iterations=3)
        return openging,closing


def operating(img):
    skin = detection.skin(img)
    dst = detection.Canny(skin)

    return dst

if __name__ == '__main__':
    cap_path = 'D:\\cap\\5.旋转.mp4'
    detection = Detection()
    '''
    #img = cv.imread(img_path)
    
    
    '''
    cap = cv.VideoCapture(cap_path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    print(h,w)
    fourrc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('hand.avi',fourrc,25.0,(h,w),False)
    while True:
        ret, frame = cap.read()
        if  not ret:
            break
        dst = operating(frame)
        out.write(frame)
        cv.imshow('cap',dst)
        if cv.waitKey(25)&0xff == ord('q'):
            break

    cap.release()