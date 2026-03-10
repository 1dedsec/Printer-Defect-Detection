# -- coding: utf-8 --
import ctypes
import inspect
import os
import sys
import threading
import time
import tkinter.messagebox
from ctypes import *
from pathlib import Path

import numpy
import pymssql
import cv2
import numpy as np
import torch
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

sys.path.append('DS77/Python/')
from API.VzenseDS_api import *
# sys.path.append("./ZoeDepth")
# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config
# sys.path.append("../RCF-PyTorch-master")
# from RCFmodels import RCF

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

sys.path.append("../MvImport")
from MvCameraControl_class import *
from trackers.multi_tracker_zoo import create_tracker
import halcon as ha

CameraParameters = ['area_scan_polynomial', 0.0279315, 138.987, 372942, -5.97038e+08, 0.360096, -2.7727, 2.25651e-06,
                    2.4e-06, 3043.65, -2540.04, 5472, 3648]
CameraPose = [-0.214615, 0.388094, 1.37056, 1.70252, 3.61043, 359.908, 0]
mtx = np.array([[2.24559893e+04, 0.00000000e+00, 3.16100345e+03],
                [0.00000000e+00, 2.25351060e+04, 1.89683094e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-1.26245447e+00, 9.02913663e+01, -2.88855662e-03, -2.21958443e-02, -2.12143653e+03]])
#                                       3             4              5             6
cls_names = ["脏污", "划痕", "油渍", "异物", "污渍", "伤", "长划痕标记"]

location_name = ["面一", "面二左上", "面二右上", "面二左下", "面二右下", "面三左上", "面三右上", "面三左下", "面三右下",
                 "面四左上", "面四右上", "面四左下", "面四右下"]

WD2X = lambda x: 0.5603 * x + 3.2838
WD2Y = lambda x: 0.4213 * x + 2.0886
WD2areaP = lambda x: (WD2X(x) / 5472) * (WD2Y(x) / 3648)

IDs = [((4980, 1111), (5021, 1140)), ((4642, 1093), (4717, 1116)), ((4544, 1086), (4588, 1129)),
       ((3903, 1055), (3939, 1091)), ((3903, 1055), (3939, 1091)), ((3903, 1055), (3939, 1091))]
centers = [((xyxy[1][0] + xyxy[0][0]) / 2, (xyxy[1][1] + xyxy[0][1]) / 2) for xyxy in IDs]
area_log = [[], [], [], [], [], []]


def init_VZ():
    camera = VzenseTofCam()
    camera_count = camera.VZ_GetDeviceCount()
    retry_count = 100
    while camera_count == 0 and retry_count > 0:
        retry_count = retry_count - 1
        camera_count = camera.VZ_GetDeviceCount()
        time.sleep(1)
        print("scaning......   ", retry_count)

    device_info = VzDeviceInfo()

    if camera_count > 1:
        ret, device_infolist = camera.VZ_GetDeviceInfoList(camera_count)
        if ret == 0:
            device_info = device_infolist[0]
            for info in device_infolist:
                print('cam uri:  ' + str(info.uri))
        else:
            print(' failed:', ret)
            exit()
    elif camera_count == 1:
        ret, device_info = camera.VZ_GetDeviceInfo()
        if ret == 0:
            print('cam uri:' + str(device_info.uri))
        else:
            print(' failed:', ret)
            exit()
    else:
        print("there are no camera found")
        exit()

    print("uri: " + str(device_info.uri))
    ret = camera.VZ_OpenDeviceByUri(device_info.uri)
    if ret == 0:
        imgpath = r"C:\Users\31604\Desktop\Project\yolov5_pyqt5-master\config\cam_params.json"
        ret = camera.VZ_SetParamsByJson(imgpath)
        if ret == 0:
            print("SetParamsByJson successful")
        else:
            print('VZ_SetParamsByJson failed: ' + str(ret))

        ret = camera.VZ_StartStream()
        if ret == 0:
            print("start stream successful")
        else:
            print("VZ_StartStream failed:", ret)
    return camera


def testArea(contour):
    return cv2.contourArea(contour)


def predict_depth(zoe, image):
    depth_numpy = zoe.infer_pil(image)  # as numpy
    return depth_numpy


def calculate_intersection_over_union(box1, box2):
    # box1 和 box2 分别表示两个边界框，格式为 (x_min, y_min, x_max, y_max)

    # 计算交集区域的坐标
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # 计算交集区域的面积
    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # 计算两个边界框的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算交集区域的面积占两个边界框总面积的比例
    iou1 = intersection_area / area_box1
    iou2 = intersection_area / area_box2
    if iou1 > 0.6 or iou2 > 0.6:
        return True
    else:
        return False


def get_contour(contours, total_area):
    length = len(contours)
    contours_sorted = sorted(contours, key=testArea, reverse=True)  # 在原地降序排序，改变 my_list 的顺序
    max_contours = [contours_sorted[0]]
    index = 1
    # while index + 1 < length and index < 4:
    #     if cv2.contourArea(contours_sorted[index]) / cv2.contourArea(contours_sorted[index + 1]) < 5:
    #         max_contours.append(contours_sorted[index + 1])
    #         index += 1
    #     else:
    #         break
    while index < length:
        if cv2.contourArea(contours_sorted[index]) / total_area > 0.02:
            # print("比例:", cv2.contourArea(contours_sorted[index]) / total_area)
            max_contours.append(contours_sorted[index])
            index += 1
        else:
            break
    return max_contours


def merge_scratch(dp_img, mark_list, scratch_list, annotator):
    iou_thresh = 0.7
    flag_list = np.zeros(len(scratch_list))
    # 检测到有长划痕的mark
    if len(mark_list) > 0:
        for xyxy in mark_list:
            total_length = 0
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # subdp_img = dp_img[int(y1):int(y2), int(x1):int(x2)]
            # mean_dp = np.mean(subdp_img)
            area_P = WD2areaP(450)
            # 结合工作距离算出平均边长
            length_P = np.sqrt(area_P)
            # 对每一个小划痕判断iou是否被mark包含
            for xyxyc in scratch_list:
                x1c, y1c, x2c, y2c = int(xyxyc[0]), int(xyxyc[1]), int(xyxyc[2]), int(xyxyc[3])
                if calculate_intersection_over_union([x1, y1, x2, y2], [x1c, y1c, x2c, y2c]):
                    total_length += max(x2c - x1c, y2c - y1c)
                    idx = scratch_list.index(xyxyc)
                    # 已被合并的小划痕不再处理
                    flag_list[idx] = 1
                    annotator.box_label(xyxyc, color=colors(1, True))
            length = length_P * total_length
            # 坐标{(x1, y1), (x2, y2)}
            annotator.box_label(xyxy, f"总长度 {length:.3f}", color=colors(6, True))
            # for center in centers:
            #     if x1 < center[0] < x2:
            #         if y1 < center[1] < y2:
            #             idx = centers.index(center)
            #             area_log[idx].append(f"{length:.3f}")
    # 对剩余的独立的小划痕单独计算长度
    for idx in range(len(flag_list)):
        if flag_list[idx] == 0:
            xyxy = scratch_list[idx]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # subdp_img = dp_img[int(y1):int(y2), int(x1):int(x2)]
            # mean_dp = np.mean(subdp_img)
            length = np.sqrt(WD2areaP(450)) * max(x2 - x1, y2 - y1)
            annotator.box_label(xyxy, f"长度 {length:.3f}", color=colors(1, True))
            # for center in centers:
            #     if x1 < center[0] < x2:
            #         if y1 < center[1] < y2:
            #             idx = centers.index(center)
            #             area_log[idx].append(f"{length:.3f}")


def multi_scale_edge(dp_img, img):
    # img = my_guidedFilter_threeChannel(img, img, 3, 0.01)
    H, W, _ = img.shape
    print("面积", H * W)
    if Globals.visible:
        cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # 转换为灰度图像
    # sharpen_image = gray_image
    # sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # for _ in range(3):
    #     sharpen_image = cv2.filter2D(sharpen_image, cv2.CV_32F, sharpen_op)
    #     sharpen_image = cv2.convertScaleAbs(sharpen_image)

    res1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    inverted_binary_image = cv2.bitwise_not(res1)
    nonzero = np.count_nonzero(inverted_binary_image)
    if nonzero < 100:
        pass
        kernel_size_d = 1
        kernel_size_e = 1
    elif nonzero < 200:
        res1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
        inverted_binary_image = cv2.bitwise_not(res1)
        kernel_size_d = 3
        kernel_size_e = 3
    elif nonzero < 300:
        res1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        inverted_binary_image = cv2.bitwise_not(res1)
        kernel_size_d = 3
        kernel_size_e = 3
    else:
        res1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        inverted_binary_image = cv2.bitwise_not(res1)
        kernel_size_d = 3
        kernel_size_e = 3
    # inverted_binary_image = cv2.erode(inverted_binary_image, np.ones((3, 3), np.uint8), iterations=1)
    # ph_area = 0
    # for row in range(H):
    #     for col in range(W):
    #         areaP = WD2areaP(dp_img[row, col])
    #         ph_area += areaP * (1 if inverted_binary_image[row, col] != 0 else 0)
    # print("单位面积", areaP)
    # print("像素总数:", np.count_nonzero(inverted_binary_image))
    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 设置阈值，小于该阈值的区域将被设为黑色
    threshold_area = H * W / 10  # 你可以根据需要调整阈值
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓区域
        area = cv2.contourArea(contour)

        # 如果白色区域面积小于阈值，将其设为黑色
        if area < threshold_area:
            cv2.drawContours(inverted_binary_image, [contour], -1, 0, thickness=cv2.FILLED)
    nonzero = np.count_nonzero(inverted_binary_image)
    print("nonzero1:", np.count_nonzero(inverted_binary_image))
    contours, hierarchy = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_img = np.zeros_like(gray_image)
    cv2.drawContours(edge_img, contours, -1, 255, thickness=cv2.FILLED)

    # 定义膨胀和腐蚀的内核大小
    kernel_d = np.ones((kernel_size_d, kernel_size_d), np.uint8)
    kernel_e = np.ones((kernel_size_e, kernel_size_e), np.uint8)
    # 对当前轮廓进行膨胀腐蚀操作
    result = cv2.erode(cv2.dilate(edge_img, kernel_d, iterations=1), kernel_e, iterations=1)
    # result = cv2.bitwise_not(result)
    # # 将当前轮廓的结果添加到最终结果中
    ph_area = 0
    for row in range(H):
        for col in range(W):
            areaP = WD2areaP(450)
            ph_area += areaP * (1 if result[row, col] != 0 else 0)
    print("nonzero2:", np.count_nonzero(result))
    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        # 绘制轮廓
        print("只有一个轮廓")
        max_contour = contours[0]
    else:
        print("有多个轮廓")
        max_contour = None
        max_area = 0
        # 遍历轮廓列表
        for contour in contours:
            # 计算当前轮廓的面积
            area = cv2.contourArea(contour)

            # 如果当前轮廓的面积大于最大面积，更新最大面积和最大轮廓
            if area > max_area:
                max_area = area
                max_contour = contour
    if max_contour is not None:
        cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 1)
    else:
        nonzero = H * W * 0.6
        # print("最大面积：", max_area)
    if Globals.visible:
        cv2.imshow("origin", img)
        cv2.imshow("threshold", inverted_binary_image)
        cv2.imshow("edge", result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return ph_area / 1.7


def multi_scale_edge_scratch(dp_img, img):
    img = img.copy()
    img = my_guidedFilter_threeChannel(img, img, 3, 0.01)
    H, W, _ = img.shape
    nonzero = 0
    if Globals.visible:
        cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
        cv2.namedWindow("edge_max", cv2.WINDOW_NORMAL)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
    # 定义窗口大小和步进大小
    # window_size = gray_image.shape[:2]
    # step_size = gray_image.shape[:2]
    # 执行滑动窗口阈值分割
    # result = sliding_window_thresholding(gray_image, window_size, step_size)
    # ret, res1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRIANGLE+cv2.THRESH_BINARY)  # 阈值分割, thresh=T
    inverted_binary_image = cv2.bitwise_not(res1)
    inverted_binary_image = cv2.erode(inverted_binary_image, np.ones((3, 3), np.uint8), iterations=1)
    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 设置阈值，小于该阈值的区域将被设为黑色
    threshold_area = W * H * 0.0001  # 你可以根据需要调整阈值

    # 遍历轮廓
    for contour in contours:
        # 计算轮廓区域
        area = cv2.contourArea(contour)

        # 如果白色区域面积小于阈值，将其设为黑色
        if area < threshold_area:
            cv2.drawContours(inverted_binary_image, [contour], -1, 0, thickness=cv2.FILLED)

    contours, hierarchy = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        cv2.destroyAllWindows()
        return 0
    edge_img = np.zeros_like(gray_image)
    cv2.drawContours(edge_img, contours, -1, 255, thickness=cv2.FILLED)
    edge_max_img = np.zeros_like(gray_image)
    # 定义膨胀和腐蚀的内核大小
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 对当前轮廓进行膨胀腐蚀操作
    # edge_img = cv2.dilate(cv2.erode(edge_img, kernel, iterations=1), kernel, iterations=1)
    result = cv2.erode(cv2.dilate(edge_img, kernel, iterations=1), kernel, iterations=1)
    # result = cv2.bitwise_not(result)
    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ph_area = 0
    if len(contours) == 0:
        cv2.destroyAllWindows()
        return 0
    elif len(contours) == 1:
        # 绘制轮廓
        # print("只有一个轮廓")
        max_contour = contours[0]
        cv2.drawContours(edge_max_img, [max_contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 1)
        nonzero = np.count_nonzero(edge_max_img)
        print(nonzero)
    else:
        # print("有多个轮廓")
        # 遍历轮廓列表
        max_contour = get_contour(contours, W * H)
        for contour in max_contour:
            cv2.drawContours(edge_max_img, [contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
        nonzero = np.count_nonzero(edge_max_img)
    edge_max_img = cv2.dilate(edge_max_img, np.ones((3, 3), np.uint8), iterations=1)
    for row in range(H):
        for col in range(W):
            areaP = WD2areaP(dp_img[row, col])
            ph_area += areaP * (1 if edge_max_img[row, col] != 0 else 0)
    print("nonzero3:", nonzero)
    if Globals.visible:
        cv2.imshow("origin", img)
        cv2.imshow("threshold", inverted_binary_image)
        cv2.imshow("edge", result)
        cv2.imshow("edge_max", edge_max_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if nonzero > H * W * 0.8:
        return 0
    return ph_area / 1.8


def cal_area(dp_img, im0, xyxy, cls, WD=500):
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    sub_img = im0[int(y1):int(y2), int(x1):int(x2)]
    # dp_img = dp_img[int(y1):int(y2), int(x1):int(x2)]
    W, H, _ = sub_img.shape
    # print(dp_img)
    if cls == 'scratch':
        # print("scratch")
        area = multi_scale_edge(dp_img, sub_img)
    else:
        # print("others")
        area = multi_scale_edge(dp_img, sub_img)

    # x, y, w, h = cv2.boundingRect(contour)
    # area = w * h
    # if area < W * H * 0.7:
    #     area = W * H
    #     contour = np.array([[x1, y1], [x1, y1 + H], [x1 + W, y1], [x1 + W, y1 + H]])
    #     p1 = (x1, y1)
    #     p2 = (x1 + H, y1 + W)
    # else:
    #     p1 = (x1 + x, y1 + y)
    #     p2 = (x1 + x + w, y1 + y + h)
    #
    # obj_area = (area * obj_dist) / (focal * (5472 * 3648))
    # return contour, obj_area, p1, p2
    # X_D = WD2X(WD)
    # Y_D = WD2Y(WD)
    # Sx = X_D / 5472
    # Sy = Y_D / 3648
    # area_p = Sx * Sy
    # area_W = area_p * area / 2

    return area


def my_guidedFilter_oneChannel(srcImg, guidedImg, rad=9, eps=0.01):
    srcImg = srcImg / 255.0
    guidedImg = guidedImg / 255.0
    img_shape = np.shape(srcImg)

    #    dstImg=np.zeros(img_shape,dtype=float)
    #
    #    P_mean=np.zeros(img_shape,dtype=float)
    #    I_mean=np.zeros(img_shape,dtype=float)
    #    I_square_mean=np.zeros(img_shape,dtype=float)
    #    I_mul_P_mean=np.zeros(img_shape,dtype=float)
    #    var_I=np.zeros(img_shape,dtype=float)
    #    cov_I_P=np.zeros(img_shape,dtype=float)
    #
    #    a=np.zeros(img_shape,dtype=float)
    #    b=np.zeros(img_shape,dtype=float)
    #    a_mean=np.zeros(img_shape,dtype=float)
    #    b_mean=np.zeros(img_shape,dtype=float)

    P_mean = cv2.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv2.boxFilter(guidedImg, -1, (rad, rad), normalize=True)

    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)

    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)

    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)

    a_mean = cv2.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv2.boxFilter(b, -1, (rad, rad), normalize=True)

    dstImg = np.multiply(a_mean, guidedImg) + b_mean

    return dstImg * 255.0


def my_guidedFilter_threeChannel(srcImg, guidedImg, rad=9, eps=0.01):
    img_shape = np.shape(srcImg)

    dstImg = np.zeros(img_shape, dtype=float)

    for ind in range(0, img_shape[2]):
        dstImg[:, :, ind] = my_guidedFilter_oneChannel(srcImg[:, :, ind],
                                                       guidedImg[:, :, ind], rad, eps)

    dstImg = dstImg.astype(np.uint8)

    return dstImg


def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if not center:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = int(bbox1[0] - bbox1[2] / 2.0), int(bbox1[1] - bbox1[3] / 2.0)
        xmax1, ymax1 = int(bbox1[0] + bbox1[2] / 2.0), int(bbox1[1] + bbox1[3] / 2.0)
        xmin2, ymin2 = int(bbox2[0] - bbox2[2] / 2.0), int(bbox2[1] - bbox2[3] / 2.0)
        xmax2, ymax2 = int(bbox2[0] + bbox2[2] / 2.0), int(bbox2[1] + bbox2[3] / 2.0)

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou


def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


def execute_num(boxes_count):
    stain = 0
    scratch = 0
    for box in boxes_count.values():
        if box[0] == 0 and box[1] > 5:
            stain += 1
        if box[0] == 1 and box[1] > 5:
            scratch += 1
    return stain, scratch


class CameraOperation():

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False,
                 b_save_jpg=False,
                 buf_save_image=None, n_save_image_size=0, frame_rate=0, exposure_time=0, gain=0):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.windows = ["image1", "image2"]

    def To_hex_str(self, num):
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

    def Open_device(self):
        if False == self.b_open_device:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_closed = False
                return ret
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                print("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self, model, model_location, index, label_layout, label_status, label_camera, table):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread,
                                                        args=(
                                                            self, model, model_location, index, label_layout,
                                                            label_status,
                                                            label_camera,
                                                            table))
                self.h_thread_handle.start()
            except:
                print('error: unable to start thread')
            return ret

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            # 退出线程
            ret = 0
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                self.b_start_grabbing = True
                self.b_exit = False
                return ret
            self.b_start_grabbing = False
            self.b_exit = True
            return ret

    def Close_device(self):
        if True == self.b_open_device:
            # 退出线程
            if True == self.b_thread_closed:
                self.b_thread_closed = False
                Stop_thread(self.h_thread_handle)
            ret = self.obj_cam.MV_CC_StopGrabbing()
            ret = self.obj_cam.MV_CC_CloseDevice()
            return ret

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True

    def Set_trigger_mode(self, strMode):
        if True == self.b_open_device:
            if "continuous" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
                if ret != 0:
                    return ret
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "On")
                if ret != 0:
                    return ret
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    return ret
                return ret

    def Trigger_once(self, nCommand):
        if True == self.b_open_device:
            if 1 == nCommand:
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                return ret

    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            self.gain = stFloatParam_gain.fCurValue
            return ret

    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            return -1
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            return ret

    def Work_thread(self, model, model_location, index, show_label, status_label, camera_label, table):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ID = 0
        conf_thres = 0.15  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference
        visualize = False  # visualize features
        line_thickness = 8  # bounding box thickness (pixels)
        hide_conf = False  # hide confidences
        boxes_count = {}
        img_buff = None
        buf_cache = None
        numArray = None
        save_dir = increment_path(Path('result'), exist_ok=True)  # increment run
        stride, names, pt = model.stride, model.names, model.pt
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        status = False
        # trackers = create_tracker('ocsort', 'trackers/ocsort/configs/ocsort.yaml', None, "0", True)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        # camera = init_VZ()
        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                if None == buf_cache:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                self.st_frame_info = stOutFrame.stFrameInfo

                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                print("Camera[%d]:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    index, self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                # print(f"self.st_frame_info.enPixelType:{PixelType_Gvsp_Mono8 == self.st_frame_info.enPixelType}")
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()
            else:
                print("Camera[" + str(index) + "]:no data, ret = " + self.To_hex_str(ret))
                continue

            # 转换像素结构体赋值
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType

            # RGB直接显示
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                numArray = CameraOperation.Color_numpy(self, buf_cache, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
            else:
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                if ret != 0:
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                # numArray = CameraOperation.Color_numpy(self, img_buff, self.st_frame_info.nWidth,
                #                                        self.st_frame_info.nHeight)
                numArray = CameraOperation.Color_numpy(self, img_buff, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
            cv2.imwrite(f"./pixel{index}.bmp", numArray)
            # numArray = ha.read_image("./pixel.bmp")
            # CarParamVirtualFixed = ha.change_radial_distortion_cam_par('adaptive', CameraParameters, [0, 0, 0, 0, 0])
            # Map = ha.gen_radial_distortion_map(CameraParameters, CarParamVirtualFixed, 'bilinear')
            # numArrayRef = ha.map_image(numArray, Map)
            numArray = ha.read_image(f"./pixel{index}.bmp")
            Pointer, Type, Width, Height = ha.get_image_pointer1(numArray)
            Width = Width[0]
            Height = Height[0]
            ImageRect = ha.gen_rectangle1(0, 0, Height - 1, Width - 1)
            ImageBorder = ha.gen_contour_region_xld(ImageRect, 'border')
            ImageBorderWCS = ha.contour_to_world_plane_xld(ImageBorder, CameraParameters, CameraPose, 1)
            MinY, MinX, MaxY, MaxX = ha.smallest_rectangle1_xld(ImageBorderWCS)
            PoseForEntireImage = ha.set_origin_pose(CameraPose, MinX, MinY, 0.01)
            WorldPixelX, WorldPixelY = ha.image_points_to_world_plane(CameraParameters, PoseForEntireImage,
                                                                      [Height / 2, Height / 2, Height / 2 + 1],
                                                                      [Width / 2, Width / 2 + 1, Width / 2], 1)
            WorldLength1 = ha.distance_pp(WorldPixelY[0], WorldPixelX[0], WorldPixelY[1], WorldPixelX[1])[0]
            WorldLength2 = ha.distance_pp(WorldPixelY[0], WorldPixelX[0], WorldPixelY[2], WorldPixelX[2])[0]
            ScaleForSimilarPixelSize = (WorldLength1 + WorldLength2) / 2
            ExtentX = MaxX[0] - MinX[0]
            ExtentY = MaxY[0] - MinY[0]
            WidthRectifiedImage = ExtentX / ScaleForSimilarPixelSize
            HeightRectifiedImage = ExtentY / ScaleForSimilarPixelSize
            Map = ha.gen_image_to_world_plane_map(CameraParameters, PoseForEntireImage, Width, Height,
                                                  WidthRectifiedImage, HeightRectifiedImage, ScaleForSimilarPixelSize,
                                                  'bilinear')
            numArrayRef = ha.map_image(numArray, Map)
            numArrayRef = ha.himage_as_numpy_array(numArrayRef)
            numArrayRef = cv2.resize(numArrayRef, (5472, 3648), interpolation=cv2.INTER_CUBIC)
            # h1, w1 = numArray.shape[:2]
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
            # # 纠正畸变
            # # dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # # dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1, h1), 5)
            # numArray = cv2.remap(numArray, mapx, mapy, cv2.INTER_LINEAR)
            # Mono8直接显示
            # if PixelType_Gvsp_Mono8 == self.st_frame_info.enPixelType:
            #     numArray = CameraOperation.Mono_numpy(self, buf_cache, self.st_frame_info.nWidth,
            #                                           self.st_frame_info.nHeight)
            #
            # # RGB直接显示
            # elif PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
            #     numArray = CameraOperation.Color_numpy(self, buf_cache, self.st_frame_info.nWidth,
            #                                            self.st_frame_info.nHeight)
            #
            # # 如果是黑白且非Mono8则转为Mono8
            # elif self.Is_mono_data(self.st_frame_info.enPixelType):
            #     nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight
            #     stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
            #     stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            #     stConvertParam.nDstBufferSize = nConvertSize
            #     ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
            #     if ret != 0:
            #         tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
            #         continue
            #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
            #     numArray = CameraOperation.Mono_numpy(self, img_buff, self.st_frame_info.nWidth,
            #                                           self.st_frame_info.nHeight)
            #
            # # 如果是彩色且非RGB则转为RGB后显示
            # elif self.Is_color_data(self.st_frame_info.enPixelType):
            #     nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
            #     stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            #     stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            #     stConvertParam.nDstBufferSize = nConvertSize
            #     ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
            #     if ret != 0:
            #         tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
            #         continue
            #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
            #     numArray = CameraOperation.Color_numpy(self, img_buff, self.st_frame_info.nWidth,
            #                                            self.st_frame_info.nHeight)
            # current_image = numArray[:, :, 0]

            # 深度相机操作
            # ret, frameready = camera.VZ_GetFrameReady(c_uint16(1000))
            # if ret != 0:
            #     print("VZ_GetFrameReady failed:", ret)
            #     continue
            # frametmp = None
            # dp_img = None
            # if frameready.depth:
            #     ret, depthframe = camera.VZ_GetFrame(VzFrameType.VzDepthFrame)
            #     if ret == 0:
            #         frametmp = numpy.ctypeslib.as_array(depthframe.pFrameData,
            #                                             (1, depthframe.width * depthframe.height * 2))
            #         frametmp.dtype = numpy.uint16
            #         frametmp.shape = (depthframe.height, depthframe.width)
            #         frametmp = cv2.resize(frametmp, (5472, 3648), interpolation=cv2.INTER_CUBIC)
            #         frametmp = frametmp[1106:2356, 707:2433]
            #         dp_img = cv2.resize(frametmp, (5472, 3648), interpolation=cv2.INTER_CUBIC)
            #     else:
            #         print("get depth frame failed:", ret)

            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            path = "result"
            im = numArrayRef.copy()
            im0s = numArrayRef.copy()
            s = ""
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                im = im.unsqueeze(0)
                # im = torch.cat((im, im, im), dim=1)
                im = im.permute(0, 3, 1, 2)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            #
            # with torch.no_grad():
            #     # Inference
            #     with dt[1]:
            #         visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #         pred = model_location(im, augment=augment, visualize=visualize)
            #
            # # NMS
            # with dt[2]:
            #     pred = non_max_suppression(pred, 0.4, 0.45, classes, agnostic_nms, max_det=max_det)
            #
            # for i, det in enumerate(pred):  # per image
            #     im0 = im0s.copy()
            #     label_location = " "
            #     max_conf = 0
            #     if len(det):
            #         # Rescale boxes from img_size to im0 size
            #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            #
            #         # Write results
            #         for *xyxy, conf, cls in reversed(det):
            #             c = int(cls)  # integer class
            #             if conf > max_conf:
            #                 max_conf = conf
            #                 label_location = location_name[c]
            #     if index == 0:
            #         camera_label.setText(f"{label_location}")
            #     elif index == 1:
            #         camera_label.setText(f"{label_location}")
            #     elif index == 2:
            #         camera_label.setText(f"顶部")

            with torch.no_grad():
                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            table.setRowCount(0)
            table.setHorizontalHeaderLabels(["监测信息"])
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0 = path, im0s.copy()
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    label_location = "无位置信息"
                    # Write results
                    count = 1
                    mark_list = []
                    scratch_list = []
                    for *xyxy, conf, cls in reversed(det):
                        flag = False
                        c = int(cls)  # integer class
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # print(f"ID:{count} 类别 {names[c]}坐标:{(x1, y1), (x2, y2)}")
                        # print(f"ID:{count}")
                        if c == 0:
                            if int(xyxy[2]) - int(xyxy[0]) < 70 and int(xyxy[3]) - int(xyxy[1]):
                                continue
                        elif c == 1:
                            scratch_list.append(xyxy)
                        elif c == 6:
                            mark_list.append(xyxy)
                        else:
                            area = cal_area(None, im0, xyxy, names[c])
                            if area == 0:
                                continue
                            # for center in centers:
                            #     if x1 < center[0] < x2:
                            #         if y1 < center[1] < y2:
                            #             idx = centers.index(center)
                            #             area_log[idx].append(f"{area:.3f}")
                            label = cls_names[c] if hide_conf else f'ID{count} {cls_names[c]} {area:.3f}'
                            # label = f'ID{count}'
                            table.insertRow(table.rowCount())
                            rowIdx = table.rowCount() - 1
                            item1 = QtWidgets.QTableWidgetItem(f"ID:{count} {names[c]},面积{area:.3f}")
                            item1.setFont(font)
                            table.setItem(rowIdx, 0, item1)
                            count += 1
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    if len(scratch_list):
                        merge_scratch(None, mark_list, scratch_list, annotator)
                    if not status:
                        status = True
                        status_label.setText(u"NG")
                        status_label.setStyleSheet("color:red")
                    # bboxes = det.cpu().numpy().copy()
                    # trackBox = trackers.update(bboxes, im0)
                    # IDs = boxes_count.keys()
                    # for bbox in trackBox:
                    #     bbox = [int(bbox[i]) for i in range(6)]
                    #     if str(bbox[4]) not in IDs:
                    #         boxes_count[str(bbox[4])] = list([bbox[5], 1])  # [ID]:[类型,nums]
                    #     else:
                    #         boxes_count[str(bbox[4])][1] += 1
                else:
                    if status:
                        status = False
                        status_label.setText(u"OK")
                        status_label.setStyleSheet("color:green")

                height, width = im0.shape[:2]
                # Stream results
                im0 = annotator.result()
                im1 = im0.astype("uint8")
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./area1.0/img_1.0area{ID + 1}.png", im1)
                cv2.imwrite("result_img.png", im1)
                ID += 1
                # im1 = cv2.resize(im1, (show_label.size().height, show_label.size().width), interpolation=cv2.INTER_CUBIC)
                im1 = QtGui.QImage(im1.data, im1.shape[1], im1.shape[0], im1.shape[1] * 3, QtGui.QImage.Format_RGB888)
                im1 = im1.scaled(show_label.size(), Qt.IgnoreAspectRatio)
                show_label.setPixmap(QPixmap(im1))

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            LOGGER.info(f"第{ID}帧")
            time.sleep(1)
            while Globals.stop_sign:
                pass
            if Globals.export_file[index]:
                conn = pymssql.connect(host='127.0.0.1', user='alo', password='12345678', database='cameradb',
                                       charset="cp936")
                if conn:
                    print(f"连接成功{conn}-----")
                    cursor = conn.cursor()
                    stain, scratch = execute_num(boxes_count)
                    sql_write = f'UPDATE Table_camera SET STAIN = {stain}, SCRATCH = {scratch} WHERE CAMERAID = {index + 1}'
                    cursor.execute(sql_write)
                    conn.commit()
                    conn.close()
                    print("修改数据库成功！")
                    print(f"stain:{stain},scratch:{scratch}")
                Globals.export_file[index] = False

            if not Globals.camera_running:
                # ui.close()
                # app.quit()
                # print("记录得到的面积:", area_log)
                for log in area_log:
                    print("面积记录开始！！")
                    for item in log:
                        print(float(item))
                    print("面积记录结束！！")
                show_label.setText("no camera")
                status_label.setText(u"OK")
                status_label.setStyleSheet("color:green")
                if index == 0:
                    camera_label.setText(f" ")
                elif index == 1:
                    camera_label.setText(f" ")
                elif index == 2:
                    camera_label.setText(f" ")
                print('stopping grabbing !!')
                # ch:停止取流 | en:Stop grab image
                ret = self.obj_cam.MV_CC_StopGrabbing()
                if ret != 0:
                    print("stop grabbing fail! ret[0x%x]" % ret)
                    sys.exit()

                # ch:关闭设备 | Close device
                ret = self.obj_cam.MV_CC_CloseDevice()
                if ret != 0:
                    print("close deivce fail! ret[0x%x]" % ret)
                    sys.exit()

                # ch:销毁句柄 | Destroy handle
                ret = self.obj_cam.MV_CC_DestroyHandle()
                if ret != 0:
                    print("destroy handle fail! ret[0x%x]" % ret)
                    sys.exit()
                break

    def Save_jpg(self, buf_cache):
        if (None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            tkinter.messagebox.showinfo('show info', 'save jpg success!')
        except:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Save_Bmp(self, buf_cache):
        if (0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)
        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save bmp fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tkinter.messagebox.showinfo('show info', 'save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray
