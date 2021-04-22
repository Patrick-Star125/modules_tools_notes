import cv2 as cv
import numpy as np
from handdetection.skin_extraction import Detection
from handdetection.Edgedetection import Edges
from handdetection.tracking import Tracking



#图像预处理
def per_treatment(img):
    detection = Detection()
    edges = Edges()
    skin = detection.skin(img)#肤色提取
    dst = edges.edgedetction(skin)#边缘检测
    cv.imshow('edge_img',dst)
    return dst



if __name__ == '__main__':

#特征点参数设置
    feature_params = dict(maxCorners=20,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7 )

#lk算法参数设置
    lk_params = dict(winSize = (15,15),
                     maxLevel = 5,
                     criteria = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10, 0.03))


    # cap =cv.VideoCapture('C:\\Users\86151\Desktop\比赛资料\服创比赛说明材料\【A12】基于手势识别的会议控制系统【长安计算】-5 种基本动作示例视频\\2.平移.mp4')
    cap =cv.VideoCapture(0)
    color = np.random.randint(0,255,(100,3))
    ptracking = Tracking()
    ret, old_frame = cap.read()


    old_gray = per_treatment(old_frame)

    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    mask = np.zeros_like(old_frame)
    flag = -1

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    w = int(cap.get(3))
    h = int(cap.get(4))
    out = cv.VideoWriter('result.avi', fourcc, 20.0, (w,h))


    while(1):
        x_sum = 0
        y_sum = 0
        diffx_sum = 0
        diffy_sum = 0

        ret,frame = cap.read()
        if not ret:
            break


        frame_gray = per_treatment(frame)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]

        good_old = p0[st==1]




        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            x_sum += a
            y_sum += b

            diffx_sum += a-c
            diffy_sum += b-d

            mask = cv.line(mask,(a,b),(c,d),color[i].tolist(),2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
            ptracking.pointtracking(frame,(a,b),(c,d))



        img = cv.add(frame,mask)
        out.write(img)
        cv.imshow('frame',img)

        k = cv.waitKey(25)

        if k==27:
            break

        old_gray = frame_gray.copy()

        p0 = good_new.reshape(-1,1,2)

    out.release()
    cap.release()
    cv.destroyAllWindows()
