# -- coding: utf-8 --
import os
import sys

sys.path.append("HaikangPython/MvImport")
from MvCameraControl_class import *
from HaikangPython.MultipleCameras.CamOperation_class import *

global deviceList
global tlayerType
global obj_cam_operation
global devList
global nOpenDevSuccess

from ctypes import *

from pathlib import Path
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import time
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def To_hex_str(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


def displayDeviceInfo(deviceList, devList):
    nips = []
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            nips.append(nip4)
            devList.append(
                "Gige[" + str(i) + "]:" + str(nip1) + "." + str(nip2) + "." + str(nip3) + "." + str(nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
            devList.append("USB[" + str(i) + "]" + str(strSerialNumber))
    return nips


def run(
        layout_list=None,
        status_list=None,
        camera_list=None,
        weights='yolov5s.pt',  # model path or triton URL
        table=None,
):
    data = ''  # dataset.yaml path
    imgsz = (2560, 2560)  # inference size (height, width)
    device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt = False  # save results to *.txt
    project = 'result'  # save results to project/name
    name = ''  # save results to project/name
    exist_ok = True  # existing project/name ok, do not increment
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.cuda.empty_cache()
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model_location = DetectMultiBackend('weights/bestlocation.pt', device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    stride_loc, names_loc, pt_loc = model_location.stride, model_location.names, model_location.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    status = False
    # Dataloader
    bs = 1  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    model_location.warmup(imgsz=(1 if pt_loc or model_location.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 调用海康相机SDK

    nOpenDevSuccess = 0
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    devList = []
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print('enum devices fail! ret = ' + ToHexStr(ret))

    # 显示相机个数

    if deviceList.nDeviceNum == 0:
        print('find no device!')

    print("Find %d devices!" % deviceList.nDeviceNum)

    nips = displayDeviceInfo(deviceList, devList)
    index_list = np.array(nips)
    b = np.argsort(index_list)
    print(index_list)
    print(b)
    # ch:打开相机 | en:open device

    obj_cam_operation = []
    for i in range(0, deviceList.nDeviceNum):
        camObj = MvCamera()
        strName = str(devList[i])
        obj_cam_operation.append(CameraOperation(camObj, deviceList, i))
        ret = obj_cam_operation[nOpenDevSuccess].Open_device()
        if 0 != ret:
            obj_cam_operation.pop()
            continue
        else:
            print(str(devList[i]))
            nOpenDevSuccess = nOpenDevSuccess + 1
        print("nOpenDevSuccess = ", nOpenDevSuccess)

    # ch:开始取流 | en:Start grab images

    for i in range(0, nOpenDevSuccess):
        ret = obj_cam_operation[b[i]].Set_trigger_mode("continuous")
        # if 0 != ret:
        #     print('camera:' + str(i) + ',Set_trigger_mode fail! ret = ' + To_hex_str(ret))
        # nRet = MvCamera.MV_CC_SetIntValueEx(obj_cam_operation[b[i]].obj_cam, "Width", 2736)
        # nRet = MvCamera.MV_CC_SetIntValueEx(obj_cam_operation[b[i]].obj_cam, "Height", 1824)

        ret = obj_cam_operation[b[i]].Start_grabbing(model, model_location, i, layout_list[i], status_list[i],
                                                     camera_list[i], table)
        if 0 != ret:
            print('camera:' + str(i) + ',start grabbing fail! ret = ' + To_hex_str(ret))
