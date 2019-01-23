import cv2
import numpy as np
import timeit
import scipy.misc
from abc import ABCMeta, abstractmethod, abstractproperty


class ROIimg(object):
    __metaclass__ = ABCMeta

    def __init__(self, frame, kx, ky, style):
        self.height, self.width, _ = frame.shape
        self.ROIwidth = int(self.width / kx)
        self.ROIheight = int(self.height / ky)
        styleTable = {'left-up': (0, 0), 'left-down': (0, self.height-self.ROIheight), 'right-up': (self.width-self.ROIwidth, 0), 'right-down': (self.width-self.ROIwidth, self.height-self.ROIheight)}
        self.x0, self.y0 = styleTable[style]
        self.sensorSwitch = 0  # 检测开关
        self.saveSwitch = 0  # 记录开关
        self.fx = 0
        self.fy = 0
        self.gusture = np.zeros((self.ROIheight, self.ROIwidth), np.uint8)  # 手势胶片

    def setFrame(self, frame):
        # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
        frame = cv2.flip(frame, 1)  # 第二个参数大于0就表示是沿y轴翻转
        self.frame = frame

    def show(self):
        cv2.imshow('frame', self.frame)

    def setROI(self):
        # 画出有效范围框
        cv2.rectangle(self.frame, (self.x0, self.y0), (self.x0+self.ROIwidth, self.y0+self.ROIheight), (0, 255, 0))
        # 提取ROI像素、预降噪
        self.ROI = cv2.GaussianBlur(self.frame[self.y0:self.y0+self.ROIheight, self.x0:self.x0+self.ROIwidth], (7, 7), 0)
        self.setInterestDetect()
        self.setBinary()
        if self.sensorSwitch:
            self.setFocusPart()
            self.setFocusPoint()
            if (timeit.default_timer() - self.start_time) >= 1:
                self.setLineGusture()
                self.getROI()
        else:
            if self.saveSwitch:
                self.saveGusture()

    def getROI(self):
        blue, green, red = cv2.split(self.ROI)
        ROIimage = cv2.merge([blue & self.gusture, green & self.gusture, red & self.gusture])
        self.frame[self.y0:self.y0+self.ROIheight, self.x0:self.x0+self.ROIwidth] = cv2.addWeighted(self.frame[self.y0:self.y0+self.ROIheight, self.x0:self.x0+self.ROIwidth], 0.3, ROIimage, 0.7, 0)

    # 轮廓面积计算函数
    def areaCal(self, contours):
        area = 0
        for i in range(len(contours)):
            area += cv2.contourArea(contours[i])
        return area

    def checkSensorSwitch(self):
        # 轮廓检测
        _, self.contours, _ = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours):
            if self.sensorSwitch == 1:
                if not (self.areaCal(self.contours) > 1000):
                    self.sensorSwitch = 0
            else:
                if self.areaCal(self.contours):
                        self.start_time = timeit.default_timer()
                        self.sensorSwitch = 1
        else:
            self.sensorSwitch = 0

    def checkSaveSwitch(self):
        if self.sensorSwitch:
            self.saveSwitch = 1
        else:
            self.saveSwitch = 0

    @abstractmethod
    def setInterestDetect(self):
        pass

    @abstractmethod
    def setBinary(self):
        pass

    @abstractmethod
    def setFocusPart(self):
        pass

    @abstractmethod
    def setFocusPoint(self):
        pass

    @abstractmethod
    def setLineGusture(self):
        pass

    @abstractmethod
    def saveGusture(self):
        pass


class Gusture(ROIimg):
    def setLineGusture(self):
        cv2.line(self.gusture, (self.fxlast, self.fylast), (self.fx, self.fy), 255, 3)
        cv2.imshow('gusture', self.gusture)

    def saveGusture(self):
        # size = self.gusture.shape  # 获取gusture图片大小（长，宽，通道数）
        # gusture = cv2.resize(gusture, (int(size[1]*28/200), int(size[0]*28/200)), cv2.INTER_LINEAR)
        # scipy.misc.imsave('D:/Desktop/智能嵌入式设计/第二讲 计算机视觉/test_pics/gusture.jpg', self.gusture)
        self.gusture = np.zeros((self.ROIheight, self.ROIwidth), np.uint8)
        self.checkSaveSwitch()


class Focus_CM(Gusture):
    def setFocusPoint(self):
        moments = cv2.moments(self.topPart)
        if moments['m00'] != 0:
            self.fxlast = self.fx
            self.fylast = self.fy
            self.fx = int(moments['m10']/moments['m00'])  # 图像重心横坐标
            self.fy = int(moments['m01']/moments['m00'])  # 图像重心纵坐标
            print('x:', self.fx, 'y:', self.fy)
            self.checkSaveSwitch()


class TopPart(Focus_CM):
    def setFocusPart(self):
        img = self.binary
        valid = int(self.ROIheight / 5)
        # 从上往下截取长度valid的较大面积肤色
        top = self.ROIheight
        self.checkSensorSwitch()
        contour = self.contours[0]
        for k in range(len(contour)):
            if top > contour[k, 0, 1]:
                top = int(contour[k, 0, 1])
        for i in range(self.ROIheight-top-valid):
            for k in range(self.ROIwidth):
                img[i+top+valid, k] = 0
        self.topPart = img
        cv2.imshow('fingertip', self.topPart)


class Binary(TopPart):
    def setBinary(self):
        gray = cv2.cvtColor(self.interestDetect, cv2.COLOR_BGR2GRAY)  # 灰度化
        blur = cv2.GaussianBlur(gray, (7, 7), 0)  # 降噪
        _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#        cv2.imshow('thresh', self.binary)

        self.checkSensorSwitch()
        if self.sensorSwitch:
            # 删除较小轮廓
            def saltErase(img, contours):
                # 按面积排序
                areas = np.zeros(len(contours))
                idx = 0
                for contour in contours:
                    areas[idx] = cv2.contourArea(contour)
                    idx = idx + 1
                areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
                imgClear = np.zeros(img.shape, dtype=np.uint8)
#                for idx in areas_s:
#                    if areas[idx] < 800:
#                        break
#                     # 绘制轮廓图像，通过将thickness设置为-1可以填充整个区域，否则只绘制边缘
#                    cv2.drawContours(imgClear, contours, idx, [255, 255, 255], -1)
                cv2.drawContours(imgClear, contours, areas_s[0], [255, 255, 255], -1)
                imgClear = imgClear & img
                return imgClear

            self.binary = saltErase(self.binary, self.contours)
            cv2.imshow('binary', self.binary)
            self.checkSensorSwitch()


class SkinDetect_RGB(Binary):
    def setInterestDetect(self):
        # =============================================================================
        # 直方图均值化
        #         lut = np.zeros(256, dtype = self.ROI.dtype )#创建空的查找表
        #         hist= cv2.calcHist([self.ROI], #计算图像的直方图
        #             [0], #使用的通道
        #             None, #没有使用mask
        #             [256], #it is a 1D histogram
        #             [0.0,255.0])
        #         minBinNo, maxBinNo = 0, 255
        #         #计算从左起第一个不为0的直方图柱的位置
        #         for binNo, binValue in enumerate(hist):
        #             if binValue != 0:
        #                 minBinNo = binNo
        #                 break
        #         #计算从右起第一个不为0的直方图柱的位置
        #         for binNo, binValue in enumerate(reversed(hist)):
        #             if binValue != 0:
        #                 maxBinNo = 255-binNo
        #                 break
        #         #生成查找表，方法来自参考文献1第四章第2节
        #         for i,v in enumerate(lut):
        #             if i < minBinNo:
        #                 lut[i] = 0
        #             elif i > maxBinNo:
        #                 lut[i] = 255
        #             else:
        #                 lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)
        #         self.ROI = cv2.LUT(self.ROI, lut)
        # =============================================================================
        img2 = np.zeros((self.ROIheight, self.ROIwidth, 3), np.uint8)
        for Y in range(0, self.ROIwidth):
            for X in range(0, self.ROIheight):
                Red = int(self.ROI[X, Y, 2])
                Green = int(self.ROI[X, Y, 1])
                Blue = int(self.ROI[X, Y, 0])
                if (Red >= 60 and Green >= 40 and Blue >= 20 and Red >= Blue and (Red - Green) >= 10 and max(max(Red, Green), Blue) - min(min(Red, Green), Blue) >= 10):
                    img2[X, Y] = self.ROI[X, Y]  # 抠图效果
                else:
                    img2[X, Y] = 0
        self.interestDetect = img2
        cv2.imshow('interestDetect', self.interestDetect)


# =============================================================================
# class SkinDetect_YCrCb(Binary):
#     def setInterestDetect(self):
#         img2 = np.zeros((self.ROIheight, self.ROIwidth, 3), np.uint8)
#         imgYcc = cv2.cvtColor(self.ROI, cv2.COLOR_BGR2YCR_CB)
#         for Y in range(0, self.ROIwidth):
#             for X in range(0, self.ROIheight):
#                 y = int(imgYcc[X, Y, 0])
#                 cr = int(imgYcc[X, Y, 1])
#                 cb = int(imgYcc[X, Y, 2])
#                 if 86 <= cb <= 117 and 90 <= cr <= 160:
#                     img2[X, Y] = self.ROI[X, Y]  # 抠图效果
#                 else:
#                     img2[X, Y] = 0
#         self.interestDetect = img2
#         cv2.imshow('nterestDetect', self.interestDetect)
# =============================================================================


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    roi_1 = SkinDetect_RGB(frame, 3, 2, 'right-up')  # kx, ky, style

    while(cap.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord('1'):  # 按1退出
            break
        ret, frame = cap.read()

        roi_1.setFrame(frame)
        roi_1.setROI()
        roi_1.show()

    cap.release()
    cv2.destroyAllWindows()
