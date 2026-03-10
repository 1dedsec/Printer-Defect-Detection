import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
# from MvImport.MvCameraControl_header import MV_CC_DEVICE_INFO_LIST
from open_camera import Ui_MainWindow  # 导入创建的GUI类
import sys
import threading
import msvcrt
from ctypes import *
from PyQt5.Qt import *
from PIL import Image, ImageTk

sys.path.append("../MvImport")
from MvCameraControl_class import *
import time
import os


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    sendAddDeviceName = pyqtSignal()  # 定义一个添加设备列表的信号。
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    g_bExit = False
    camera_information = False  # 获取相机标志
    opencamera_flay = False  # 打开相机标志
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        # self.connect_and_emit_sendAddDeviceName()
        self.init()
        self.label.setScaledContents(True)  # 图片自适应
        self.label_2.setScaledContents(True)  # 图片自适应

    def init(self):
        # 获取相机相信
        self.pushButton.clicked.connect(self.get_camera_information)
        # 打开摄像头
        self.pushButton_2.clicked.connect(self.openCamera)
        # 拍照
        self.pushButton_3.clicked.connect(self.taking_pictures)
        # 关闭摄像头
        self.pushButton_4.clicked.connect(self.closeCamera)
        # Connect the sendAddDeviceName signal to a slot.
        # self.sendAddDeviceName.connect(self.camera_information)
        # Emit the signal.
        # self.sendAddDeviceName.emit()

        # 获得所有相机的列表存入cmbSelectDevice中

    def get_camera_information(self):
        '''选择所有能用的相机到列表中，
             gige相机需要配合 sdk 得到。
        '''
        # 得到相机列表
        # tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # QMessageBox.critical(self, '错误', '读取设备驱动失败！')
            # sys.exit()
        if self.deviceList.nDeviceNum == 0:
            QMessageBox.critical(self, "错误", "没有发现设备 ！ ")
            # print("find no device!")
            # sys.exit()
        else:
            QMessageBox.information(self, "提示", "发现了 %d 个设备 !" % self.deviceList.nDeviceNum)
        # print("Find %d devices!" % self.deviceList.nDeviceNum)
        if self.deviceList.nDeviceNum == 0:
            return None

        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

        self.camera_information = True

    # 打开摄像头。
    def openCamera(self, camid=0):
        if self.camera_information == True:
            self.g_bExit = False
            # ch:选择设备并创建句柄 | en:Select device and create handle
            stDeviceList = cast(self.deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                # print("create handle fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "创建句柄失败 ! ret[0x%x]" % ret)
                # sys.exit()
            # ch:打开设备 | en:Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                # print("open device fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "打开设备失败 ! ret[0x%x]" % ret)
                # sys.exit()
            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        # print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                        QMessageBox.warning(self, "警告", "报文大小设置失败 ! ret[0x%x]" % ret)
                else:
                    # print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                    QMessageBox.warning(self, "警告", "报文大小获取失败 ! ret[0x%x]" % nPacketSize)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                # print("set trigger mode fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "设置触发模式失败 ! ret[0x%x]" % ret)
                # sys.exit()
                # ch:获取数据包大小 | en:Get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                # print("get payload size fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "获取有效负载大小失败 ! ret[0x%x]" % ret)
                # sys.exit()
            nPayloadSize = stParam.nCurValue

            # ch:开始取流 | en:Start grab image
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                # print("start grabbing fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "开始抓取图像失败 ! ret[0x%x]" % ret)
                # sys.exit()

            data_buf = (c_ubyte * nPayloadSize)()
            self.opencamera_flay = True
            try:
                hThreadHandle = threading.Thread(target=self.work_thread, args=(self.cam, data_buf, nPayloadSize))
                hThreadHandle.start()
            except:
                # print("error: unable to start thread")
                QMessageBox.critical(self, "错误", "无法启动线程 ! ")

        else:
            QMessageBox.critical(self, '错误', '获取相机信息失败！')
            return None

    # 关闭相机
    def closeCamera(self):
        if self.opencamera_flay == True:
            self.g_bExit = True
            # ch:停止取流 | en:Stop grab image
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                # print("stop grabbing fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "停止抓取图像失败 ! ret[0x%x]" % ret)
                # sys.exit()

            # ch:关闭设备 | Close device
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                # print("close deivce fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "停止设备失败 ! ret[0x%x]" % ret)
            # ch:销毁句柄 | Destroy handle
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                # print("destroy handle fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "销毁处理失败 ! ret[0x%x]" % ret)

            self.label.clear()  # 清除label组件上的图片
            self.label_2.clear()  # 清除label组件上的图片
            self.label.setText("摄像头")
            self.label_2.setText("显示图片")
            self.camera_information = False
            self.opencamera_flay = False
        else:
            QMessageBox.critical(self, '错误', '未打开摄像机！')
            return None

    def work_thread(self, cam=0, pData=0, nDataSize=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                image = np.asarray(pData)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
                # print(image.shape)
                image = image.reshape((5472, 3648))  # 根据自己分辨率进行转化
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 这一步获取到的颜色不对，因为默认是BRG，要转化成RGB，颜色才正常
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2BGR)  # Bayer格式（raw data）向RGB或BGR颜色空间的转换
                # print(image.shape)
                # pyrD1 = cv2.pyrDown(image)  # 向下取样
                # pyrD2 = cv2.pyrDown(pyrD1)  # 向下取样
                image_height, image_width, image_depth = image.shape  # 读取图像高宽深度
                self.image_show = QImage(image.data, image_width, image_height, image_width * image_depth,
                                         QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(self.image_show))
            if self.g_bExit == True:
                del pData
                break

    # 拍照
    def taking_pictures(self):
        if self.opencamera_flay == True:
            FName = fr"images/cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
            print(FName)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image_show))
            # self.showImage.save(FName + ".jpg", "JPG", 100)
            self.image_show.save('./1.bmp')
        else:
            QMessageBox.critical(self, '错误', '摄像机未打开！')
            return None

    # 重写关闭函数
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', "确认退出吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            # 用过sys.exit(0)和sys.exit(app.exec_())，但没起效果
            os._exit(0)
        else:
            event.ignore()


if __name__ == '__main__':
    from PyQt5 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率

    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
