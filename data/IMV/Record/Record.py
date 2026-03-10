# -- coding: utf-8 --

import threading
import sys
import msvcrt
from ctypes import *

sys.path.append("../MVSDK")
from IMVApi import *

winfun_ctype = WINFUNCTYPE

pFrame = POINTER(IMV_Frame)
FrameInfoCallBack = winfun_ctype(None, pFrame, c_void_p)

# 为线程定义一个函数
def frameGrabbingProc(cam):
    devHandle=cam.handle
    stRecordFrameInfoParam=IMV_RecordFrameInfoParam()
    frame=IMV_Frame()
    imageCount=0
    if None==devHandle:
        return
    
    for i in range(0,500):
        nRet=cam.IMV_GetFrame(frame,500)
        if IMV_OK!=nRet:
            print("Get frame failed!ErrorCode[%d]" % nRet)
            continue
        memset(byref(stRecordFrameInfoParam), 0, sizeof(stRecordFrameInfoParam))
        stRecordFrameInfoParam.pData = frame.pData
        stRecordFrameInfoParam.nDataLen = frame.frameInfo.size
        stRecordFrameInfoParam.nPaddingX = frame.frameInfo.paddingX
        stRecordFrameInfoParam.nPaddingY = frame.frameInfo.paddingY
        stRecordFrameInfoParam.ePixelFormat = frame.frameInfo.pixelFormat

        nRet=cam.IMV_InputOneFrame(stRecordFrameInfoParam)

        if IMV_OK==nRet:
            imageCount=imageCount+1
            print("record frame %d successfully!" % imageCount)
        else:
            print("record failed! ErrorCode[%d]" % nRet)

        nRet=cam.IMV_ReleaseFrame(frame)

        if IMV_OK!=nRet:
            print("Release frame failed! ErrorCode[%d]" % nRet)
    return

def displayDeviceInfo(deviceInfoList):  
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0,deviceInfoList.nDevNum):
        pDeviceInfo=deviceInfoList.pDevInfo[i]
        strType=""
        strVendorName=""
        strModeName = ""
        strSerialNumber=""
        strCameraname=""
        strIpAdress=""
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
            strType="Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType="U3V"
        print ("[%d]  %s   %s    %s      %s     %s           %s" % (i+1, strType,strVendorName,strModeName,strSerialNumber,strCameraname,strIpAdress))

if __name__ == "__main__":
    deviceList=IMV_DeviceList()
    interfaceType=IMV_EInterfaceType.interfaceTypeAll
    stRecordParam=IMV_RecordParam()
    nWidth=c_uint()
    nHeight=c_uint()

    # 枚举设备
    nRet=MvCamera.IMV_EnumDevices(deviceList,interfaceType)
    if IMV_OK != nRet:
        print("Enumeration devices failed! ErrorCode",nRet)
        sys.exit()
    if deviceList.nDevNum == 0:
        print ("find no device!")
        sys.exit()

    print("deviceList size is",deviceList.nDevNum)

    displayDeviceInfo(deviceList)

    nConnectionNum = input("Please input the camera index: ")

    if int(nConnectionNum) > deviceList.nDevNum:
        print ("intput error!")
        sys.exit()

    cam=MvCamera()
    # 创建设备句柄
    nRet=cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex,byref(c_void_p(int(nConnectionNum)-1)))
    if IMV_OK != nRet:
        print("Create devHandle failed! ErrorCode",nRet)
        sys.exit()
        
    # 打开相机
    nRet=cam.IMV_Open()
    if IMV_OK != nRet:
        print("Open devHandle failed! ErrorCode",nRet)
        sys.exit()
    
    # 宽和高必须是2的整数倍
    if True!=cam.IMV_FeatureIsValid("Width"):
        print("Width feature is invalid!")
    nRet=cam.IMV_GetIntFeatureValue("Width",nWidth)
    if IMV_OK != nRet:
        print("Get width feature value failed! ErrorCode[%d]",nRet)
        sys.exit()

    if True!=cam.IMV_FeatureIsValid("Height"):
        print("Height feature is invalid!")
    nRet=cam.IMV_GetIntFeatureValue("Height",nHeight)
    if IMV_OK != nRet:
        print("Get Height feature value failed! ErrorCode[%d]",nRet)
        sys.exit()

    memset(byref(stRecordParam),0,sizeof(stRecordParam))

    stRecordParam.nWidth=nWidth
    stRecordParam.nHeight=nHeight
    stRecordParam.fFameRate=20
    stRecordParam.nQuality = 30
    stRecordParam.recordFormat = IMV_EVideoType.typeVideoFormatAVI
    stRecordParam.pRecordFilePath ="record.avi".encode('ascii')
    nRet=cam.IMV_OpenRecord(stRecordParam)
    if IMV_OK!=nRet:
        print("OpenRecord failed! ErrorCode[%d]" % nRet)
        sys.exit()

    # 开始拉流
    nRet=cam.IMV_StartGrabbing()
    if IMV_OK != nRet:
        print("Start grabbing failed! ErrorCode",nRet)
        sys.exit()
    
    try:
        hThreadHandle = threading.Thread(target=frameGrabbingProc, args=(cam,))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")

    hThreadHandle.join()
    
    # 关闭录像
    cam.IMV_CloseRecord()

    # 停止拉流
    nRet=cam.IMV_StopGrabbing()
    if IMV_OK != nRet:
        print("Stop grabbing failed! ErrorCode",nRet)
        sys.exit()
    
    # 关闭相机
    nRet=cam.IMV_Close()
    if IMV_OK != nRet:
        print("Close camera failed! ErrorCode",nRet)
        sys.exit()
    
    # 销毁句柄
    if(cam.handle):
        nRet=cam.IMV_DestroyHandle()
    
    print("---Press any key to exit---")
    msvcrt.getch()