# -- coding: utf-8 --
import sys

sys.path.append("../MvImport")
from MvCameraControl_class import *
from CamOperation_class import *


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


if __name__ == "__main__":
    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global tlayerType
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    global obj_cam_operation
    obj_cam_operation = []
    global nOpenDevSuccess
    nOpenDevSuccess = 0
    global devList

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        tkinter.messagebox.showerror('show error', 'enum devices fail! ret = ' + ToHexStr(ret))

    # 显示相机个数

    if deviceList.nDeviceNum == 0:
        tkinter.messagebox.showinfo('show info', 'find no device!')

    print("Find %d devices!" % deviceList.nDeviceNum)

    devList = []
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
    ret = 0
    for i in range(0, nOpenDevSuccess):
        ret = obj_cam_operation[i].Set_trigger_mode("continuous")
        # if 0 != ret:
        #     print('camera:' + str(i) + ',Set_trigger_mode fail! ret = ' + To_hex_str(ret))
        ret = obj_cam_operation[i].Start_grabbing(i)
        if 0 != ret:
            print('camera:' + str(i) + ',start grabbing fail! ret = ' + To_hex_str(ret))
