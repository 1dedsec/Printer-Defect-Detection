import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette
import time
from PyQt5.QtWidgets import QTableWidgetItem
from UI.widget import Report_UI
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# sys.path.append("./ZoeDepth")
# from RCFmodels import RCF
# Tensor
# Local file
# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config
from PIL import Image

WD2X = lambda x: 0.5603 * x + 3.2838
WD2Y = lambda x: 0.4213 * x + 2.0886
WD2areaP = lambda x: (WD2X(x) / 5472) * (WD2Y(x) / 3648)


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


@smart_inference_mode()
def run(
        weights='yolov5s.pt',  # model path or triton URL
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        data='',  # dataset.yaml path
        imgsz=(3840, 3840),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_img=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='result',  # save results to project/name
        name='',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=6,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        show_label=None,
        status_label=None,
        use_camera=False,
        table=None,
):
    table.setRowCount(0)
    table.setHorizontalHeaderLabels(["监测信息"])
    font = QtGui.QFont()
    font.setFamily("Agency FB")
    font.setPointSize(18)
    status_label.setFixedWidth(120)
    status_label.setFixedHeight(120)
    status_label.setAlignment(Qt.AlignCenter)

    status_label.setText(u"OK")
    # pe.setColor(QPalette.Background,Qt.blue)<span style="font-family: Arial, Helvetica, sans-serif;">#设置背景颜色，和上面一行的效果一样
    source = str(source)
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    webcam = source.isnumeric()
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    status = False
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # model = RCF().cuda()
        #
        # if os.path.isfile('./RCF-PyTorch-master/bsds500_pascal_model.pth'):
        #     print("=> loading checkpoint from '{}'".format('./RCF-PyTorch-master/bsds500_pascal_model.pth'))
        #     checkpoint = torch.load('./RCF-PyTorch-master/bsds500_pascal_model.pth')
        #     model.load_state_dict(checkpoint)
        #     print("=> checkpoint loaded")
        # else:
        #     print("=> no checkpoint found at '{}'".format('./RCF-PyTorch-master/bsds500_pascal_model.pth'))
        # Process predictions
        # ZoeD_NK
        # conf = get_config("zoedepth_nk", "infer")
        # model_zoe_nk = build_model(conf)
        # ##### sample prediction
        # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # zoe = model_zoe_nk.to(DEVICE)
        zoe = None
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
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
                count = 1
                # im1 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                # dp_img = zoe.infer_pil(im1)
                # print(dp_img)
                # Write results
                count = 1
                mark_list = []
                scratch_list = []
                for *xyxy, conf, cls in reversed(det):
                    flag = False
                    c = int(cls)  # integer class
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # print(f"ID:{count} 类别 {names[c]}坐标:{(x1, y1), (x2, y2)}")
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
                        # label = cls_names[c] if hide_conf else f'ID{count} {cls_names[c]} {area:.3f}'
                        label = f'ID{count}'
                        table.insertRow(table.rowCount())
                        rowIdx = table.rowCount() - 1
                        item1 = QtWidgets.QTableWidgetItem(f"ID:{count} {names[c]},面积{area:.3f}")
                        item1.setFont(font)
                        table.setItem(rowIdx, 0, item1)
                        count += 1
                        annotator.box_label(xyxy, label, color=colors(c, True))
                if len(scratch_list):
                    merge_scratch(None, mark_list, scratch_list, annotator)
                if webcam and not status:
                    status = True
                    status_label.setText(u"NG")
                    status_label.setStyleSheet("color:red")
            else:
                if webcam and status:
                    status = False
                    status_label.setText(u"OK")
                    status_label.setStyleSheet("color:green")

            height, width = im0.shape[:2]
            # Stream results
            im0 = annotator.result()
            im1 = im0.astype("uint8")
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            # im1 = cv2.resize(im1, (height, width), interpolation=cv2.INTER_CUBIC)
            im1 = QtGui.QImage(im1[:], im1.shape[1], im1.shape[0], im1.shape[1] * 3, QtGui.QImage.Format_RGB888)
            show_label.setScaledContents(True)
            show_label.setPixmap(QtGui.QPixmap(im1))

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
