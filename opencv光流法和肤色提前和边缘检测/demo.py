import cv2 as cv


class Buffer:
    def __init__(self,length):
        self.len = length
        self.q = []
        self.number = 0

    def isempty(self):
        if self.q == []:
            return 1
        else:
            return 0
    def isfull(self):
        if self.number == self.len:
            return 1
        else:
            return 0


    def enqueue(self,elem):
        if self.isfull():
            pass
        else:
            self.number += 1
            self.q.append(elem)

    def dequeue(self):
        if self.isempty():
            return 0
        else:
            self.number -= 1
            return self.q.pop(0)

    # 读取缓存最早图片
    def readBuffer(self):
        val = self.dequeue()
        return val
    # 读取缓存区图片
    def readBuffers(self):
        return self.q
    # 清理缓存
    def clearBuffer(self):
        self.q.clear()
        return self.q
    # 存入图片
    def writeBuffer(self,elem):
        self.enqueue(elem)


if __name__ == '__main__':
    buf = Buffer(30) #创建缓存
    # cap =cv.VideoCapture('C:\\Users\86151\Desktop\比赛资料\服创比赛说明材料\【A12】基于手势识别的会议控制系统【长安计算】-5 种基本动作示例视频\\2.平移.mp4')
    cap =cv.VideoCapture(0)
    img_path = 'D:\\cap\\buf\\'
    img_num = 0
    img_buf = [] #图片列表
    while True:
        ret, frame = cap.read()


        if not ret:
            break
        img_name = img_path + str(img_num) + '.jpg'
        img_num += 1
        cv.imwrite(img_name, frame)
        cv.imshow('frame', frame)


        if buf.isfull():

            img_buf.extend(buf.readBuffers())#把缓存区的图片载入，这里可以根据需要修改
            buf.clearBuffer()#清理缓存
            pass
        else:
            buf.writeBuffer(frame)#将图片存入缓存，也可以存图片地址（比较麻烦）

        k = cv.waitKey(25)
        if k==27:
            break


    cv.destroyAllWindows()
    cap.release()