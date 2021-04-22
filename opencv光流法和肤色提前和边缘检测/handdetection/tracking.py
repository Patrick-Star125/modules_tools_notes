import cv2 as cv

#队列
class Queue:
    def __init__(self,length):

        self.q = list()
        self.len = length
        self.head = 0
        self.tail = 0


    def isempty(self):
        if self.q == []:
            return 1
        else:
            return 0

    def isfull(self):
        if self.tail == self.len:
            return 1
        else:
            return 0


    def clear(self):
        self.q.clear()


    def dequeue(self):
        if self.isempty():
            self.head = 0
            self.tail = 0
            return 0

        else :
            self.head += 1
            return self.q.pop(0)

    def enqueue(self,elem):
        if self.isfull():
            print('Queue is full')

        else:
            self.q.append(elem)
            self.tail += 1

    def getqueue(self):
        return self.q

#点跟踪
class Tracking:

    #创建队列，用于存储偏移量
    def __init__(self):
        self.sumx = Queue(20)
        self.sumy = Queue(20)
        self.delta_x = 0 #偏移量初始化
        self.delta_y = 0
        self.diffx = 0 #偏移量之和
        self.diffy = 0

    #求算偏移量，传入两帧的坐标（x，y）
    def displacenment(self,new_point,old_point):
        x1,y1 = old_point
        x2,y2 = new_point
        self.delta_x = x2-x1
        self.delta_y = y2-y1
        return (self.delta_x,self.delta_y)


    #进行方向判断
    def pointtracking(self,frame,new_point,old_point):

        deltax,deltay = self.displacenment(new_point,old_point)

        #对偏移量求和，判断20帧内点的在x，y轴上的位移量，队列满则进行偏移量求和
        if self.sumx.isfull():
            for i in range(self.sumx.len):
                self.diffx += self.sumx.dequeue()
        else:
            self.sumx.enqueue(deltax)

        if self.sumy.isfull():
            for i in range(self.sumy.len):
                self.diffy += self.sumy.dequeue()
        else:
            self.sumy.enqueue(deltay)

        #左右平移判断，y方向小于50则是水平平移，否则为垂直移动
        if abs(self.diffy) < 50:
            if self.diffx > 50:
                cv.putText(frame, "left", (100, 100), 0, 0.5, (0, 0, 255), 2)

            elif self.diffx < -50:
                cv.putText(frame, "right", (100, 100), 0, 0.5, (0, 0, 255), 2)

            self.diffx = 0
            self.diffy = 0

        else:
            if self.diffy > 0:
                cv.putText(frame, "down", (100, 100), 0, 0.5, (0, 0, 255), 2)
            else:
                cv.putText(frame, "up", (100, 100), 0, 0.5, (0, 0, 255), 2)

            self.diffx = 0
            self.diffy = 0




