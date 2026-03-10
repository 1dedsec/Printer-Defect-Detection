# -- coding: utf-8 --
import ctypes

import sys

from ctypes import *
from pathlib import Path
import cv2
import numpy as np
import torch
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


cls_names = ["污渍", "划痕", "油渍"]


def cal_area(im0, xyxy):
    focal = 25
    obj_dist = 400
    im0 = my_guidedFilter_threeChannel(im0, im0, 9, 0.01)
    x1, y1, x2, y2 = xyxy
    sub_img = im0[y1:y2, x1:x2]
    dst_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2LAB)
    dst_img0 = dst_img[:, :, 0]
    print(np.shape(dst_img0))
    result_1 = cv2.Canny(dst_img0, 100, 80, apertureSize=3)

    cv2.imshow("threshold1=64,threshold2=256时的边缘检测结果", result_1)

    contours, _ = cv2.findContours(result_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化变量来存储最大轮廓和最大面积
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

    x, y, w, h = cv2.boundingRect(max_contour)
    p1 = (x1 + x, y1 + y)
    p2 = (x1 + x + w, y1 + y + h)

    obj_area = max_area * (obj_dist / focal) ** 2
    return obj_area, p1, p2


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


if __name__ == '__main__':
    img = cv2.imread("JPEGImages/Image_20230808172551196.bmp", cv2.IMREAD_COLOR)
    weights = './weights/best87.pt',  # model path or triton URL
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
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    status = False
    # Dataloader
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    ID = 0
    conf_thres = 0.55  # confidence threshold
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

    path = "result"
    im = img.copy()
    im0s = img.copy()
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
            max_conf = 0
            # Write results
            for *xyxy, conf, cls in reversed(det):
                flag = False
                c = int(cls)  # integer class
                label = cls_names[c] if hide_conf else f'{cls_names[c]} {conf:.2f}'
                if c == 0:
                    if int(xyxy[2]) - int(xyxy[0]) < 70 and int(xyxy[3]) - int(xyxy[1]):
                        continue
                obj_area, p1, p2 = cal_area(im0, xyxy)
                annotator.box_label(xyxy, label, obj_area, p1, p2, color=colors(c, True))

            bboxes = det.cpu().numpy().copy()

        height, width = im0.shape[:2]
        # Stream results
        im0 = annotator.result()
        im1 = im0.astype("uint8")
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", im1)

    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
