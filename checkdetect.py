import argparse
import time
from pathlib import Path
import sys

from ctypes import *
import datetime
import numpy
import threading

from PIL import Image

sys.path.append("IMV/MVSDK")
from IMVApi import *

from pathlib import Path
import torch
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPicture, QPainter
import time
from PyQt5.QtWidgets import QTableWidgetItem
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.general import increment_path
from utils.myutil import Globals


def displayDeviceInfo(deviceInfoList):
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0, deviceInfoList.nDevNum):
        pDeviceInfo = deviceInfoList.pDevInfo[i]
        strType = ""
        strVendorName = ""
        strModeName = ""
        strSerialNumber = ""
        strCameraname = ""
        strIpAdress = ""
        for str in pDeviceInfo.vendorName:
            strVendorName = strVendorName + chr(str)
        for str in pDeviceInfo.modelName:
            strModeName = strModeName + chr(str)
        for str in pDeviceInfo.serialNumber:
            strSerialNumber = strSerialNumber + chr(str)
        for str in pDeviceInfo.cameraName:
            strCameraname = strCameraname + chr(str)
        for str in pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress:
            strIpAdress = strIpAdress + chr(str)
        if pDeviceInfo.nCameraType == typeGigeCamera:
            strType = "Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType = "U3V"
        print("[%d]  %s   %s    %s      %s     %s           %s" % (
            i + 1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))


def run(
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        view_img=True,  # show results
        show_label=None,
        use_camera=False,
):
    nConnectionNum = source + 1
    # 创建设备句柄
    cam = MvCamera()
    nRet = cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(int(nConnectionNum) - 1)))
    if IMV_OK != nRet:
        print("Create devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 打开相机
    nRet = cam.IMV_Open()
    if IMV_OK != nRet:
        print("Open devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 通用属性设置:设置触发模式为off
    nRet = IMV_OK
    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
    if IMV_OK != nRet:
        print("Set triggerSource value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    # nRet = cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
    if IMV_OK != nRet:
        print("Set triggerSelector value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    nRet = cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
    if IMV_OK != nRet:
        print("Set triggerMode value failed! ErrorCode[%d]" % nRet)
        sys.exit()

    # 开始拉流
    nRet = cam.IMV_StartGrabbing()
    if IMV_OK != nRet:
        print("Start grabbing failed! ErrorCode", nRet)
        sys.exit()
    while True:
        # 主动取图
        frame = IMV_Frame()
        stPixelConvertParam = IMV_PixelConvertParam()
        t0 = time.time()
        nRet = cam.IMV_GetFrame(frame, 10000)
        if IMV_OK != nRet:
            print("[%d]getFrame fail! Timeout:[10000]ms" % source)
            time.sleep(1)
            continue
        else:
            print("[" + str(source) + "] getFrame success BlockId = [" + str(
                frame.frameInfo.blockId) + "], get frame time: " + str(
                datetime.datetime.now()))

        if None == byref(frame):
            print("pFrame is NULL!")
            continue
        # 给转码所需的参数赋值

        if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
        else:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3

        pDstBuf = (c_ubyte * nDstBufSize)()
        memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))

        stPixelConvertParam.nWidth = frame.frameInfo.width
        stPixelConvertParam.nHeight = frame.frameInfo.height
        stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pSrcData = frame.pData
        stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
        stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
        stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
        stPixelConvertParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
        stPixelConvertParam.eDstPixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pDstBuf = pDstBuf
        stPixelConvertParam.nDstBufSize = nDstBufSize

        # 释放驱动图像缓存
        # release frame resource at the end of use

        nRet = cam.IMV_ReleaseFrame(frame)
        if IMV_OK != nRet:
            print("Release frame failed! ErrorCode[%d]\n", nRet)
            sys.exit()

        # 如果图像格式是 Mono8 直接使用
        # no format conversion required for Mono8
        if stPixelConvertParam.ePixelFormat == IMV_EPixelType.gvspPixelMono8:
            imageBuff = stPixelConvertParam.pSrcData
            userBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)

            memmove(userBuff, imageBuff, stPixelConvertParam.nDstBufSize)
            grayByteArray = bytearray(userBuff)

            cvImage = numpy.array(grayByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth)

        else:
            # 转码 => BGR24
            # convert to BGR24
            stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
            # stPixelConvertParam.nDstBufSize=nDstBufSize

            nRet = cam.IMV_PixelConvert(stPixelConvertParam)
            if IMV_OK != nRet:
                print("image convert to failed! ErrorCode[%d]" % nRet)
                del pDstBuf
                sys.exit()
            rgbBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)
            memmove(rgbBuff, stPixelConvertParam.pDstBuf, stPixelConvertParam.nDstBufSize)
            colorByteArray = bytearray(rgbBuff)
            cvImage = numpy.array(colorByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth,
                                                          3)
            if None != pDstBuf:
                del pDstBuf
                pass
        im0 = cvImage.copy()
        # result = cv2.matchTemplate(im0, template, cv2.TM_SQDIFF_NORMED)
        # # 归一化处理
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        # # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # # 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置定点取min_loc,对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
        # strmin_val = str(min_val)
        # # 绘制矩形边框，将匹配区域标注出来
        # cv2.rectangle(im0, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
        # print("MatchingValue=", strmin_val)

        if view_img:
            im1 = im0.astype("uint8")
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            # im1 = cv2.resize(im1, (show_label.size().height, show_label.size().width), interpolation=cv2.INTER_CUBIC)
            im1 = QtGui.QImage(im1.data, im1.shape[1], im1.shape[0], im1.shape[1] * 3, QtGui.QImage.Format_RGB888)
            im1 = im1.scaled(show_label.size(), Qt.IgnoreAspectRatio)
            show_label.setPixmap(QPixmap(im1))
        # 关闭摄像头->退出
        if not Globals.camera_running and use_camera:
            break

    print(f'Done. ({time.time() - t0:.3f}s)')

    # 停止拉流
    nRet = cam.IMV_StopGrabbing()
    if IMV_OK != nRet:
        print("Stop grabbing failed! ErrorCode", nRet)
        sys.exit()

    nRet = cam.IMV_ClearFrameBuffer()
    if IMV_OK != nRet:
        print("Stop grabbing failed! ErrorCode", nRet)
        sys.exit()

    # 关闭相机
    nRet = cam.IMV_Close()
    if IMV_OK != nRet:
        print("Close camera failed! ErrorCode", nRet)
        sys.exit()

    # 销毁句柄
    if cam.handle:
        nRet = cam.IMV_DestroyHandle()

    print("---Press any key to exit---")
    # msvcrt.getch()
