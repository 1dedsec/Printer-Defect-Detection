import cv2
import numpy as np
#
# original = cv2.imread("JPEGImages/Image_20230808171721375.bmp")
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("lena_gray", cv2.WINDOW_NORMAL)
# cv2.namedWindow("threshold1=64,threshold2=256时的边缘检测结果", cv2.WINDOW_NORMAL)
# # cv2.namedWindow("threshold1=16,threshold2=128时的边缘检测结果", cv2.WINDOW_NORMAL)
# # cv2.namedWindow("threshold1=8,threshold2=64时的边缘检测结果", cv2.WINDOW_NORMAL)
# result_1 = cv2.Canny(original, 8, 100, 15)
# # result_2 = cv2.Canny(original, 16, 128)
# # result_3 = cv2.Canny(original, 8, 64)
# cv2.imshow("lena_gray", original)
# cv2.imshow("threshold1=64,threshold2=256时的边缘检测结果", result_1)
# # cv2.imshow("threshold1=16,threshold2=128时的边缘检测结果", result_2)
# # cv2.imshow("threshold1=8,threshold2=64时的边缘检测结果", result_3)
# cv2.waitKey()
# cv2.destroyAllWindows()
import os, cv2

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:46:20 2019

@author: 不用先生
"""

import cv2
import numpy as np

input_fn = "C:/Users/31604/Desktop/Project/yolov5_pyqt5-master/data/12mm小污渍实例/instance1.bmp"


def robert_filter(image):
    h = image.shape[0]
    w = image.shape[1]
    image_new = np.zeros(image.shape, np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            image_new[i][j] = np.abs(image[i][j].astype(int) - image[i + 1][j + 1].astype(int)) + np.abs(
                image[i + 1][j].astype(int) - image[i][j + 1].astype(int))
    return image_new


def laplacian_filter(image):
    h = image.shape[0]
    w = image.shape[1]
    image_new = np.zeros(image.shape, np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            image_new[i][j] = 4 * image[i][j].astype(int) - image[i + 1][j].astype(int) - image[i - 1][j].astype(int) - \
                              image[i][j + 1].astype(int) - image[i][j - 1].astype(int)
    return image_new


def my_guidedFilter_oneChannel(srcImg, guidedImg, rad=9, eps=0.01):
    srcImg = srcImg / 255.0
    guidedImg = guidedImg / 255.0
    img_shape = np.shape(srcImg)

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


def main():
    p11 = (676, 2058)
    p12 = (719, 2092)
    p21 = (1279, 1309)
    p22 = (1308, 1345)
    p31 = (2094, 2967)
    p32 = (2110, 2981)
    cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
    img = cv2.imread(input_fn)
    h, w, c = img.shape
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_img = gray_image[p11[1]:p12[1], p11[0]:p12[0]]
    # sub_img = my_guidedFilter_threeChannel(sub_img, sub_img, 9, 0.01)
    # img = robert_filter(img)
    # 转换为灰度图像

    # sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # sharpen_image = cv2.filter2D(gray_image, cv2.CV_32F, sharpen_op)
    # sharpen_image = cv2.convertScaleAbs(sharpen_image)
    # 确保灰度图像是8位无符号单通道图像
    if sub_img.dtype != np.uint8:
        sub_img = cv2.convertScaleAbs(sub_img)
    #  4  16
    # result_1 = cv2.Canny(gray_image, 4, 16, apertureSize=3)
    # 应用自适应阈值
    # threshold_value = 50  # 阈值，根据需要调整
    # ret, gray_image = cv2.threshold(sub_img, threshold_value, 113, cv2.THRESH_BINARY)
    # inverted_binary_image = cv2.bitwise_not(gray_image)
    # res1 = cv2.adaptiveThreshold(sub_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    res1 = cv2.adaptiveThreshold(sub_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
    inverted_binary_image = cv2.bitwise_not(res1)
    contours, hierarchy = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_img = np.zeros_like(sub_img)
    cv2.drawContours(edge_img, contours, -1, 255, thickness=cv2.FILLED)

    # 定义膨胀和腐蚀的内核大小
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 对当前轮廓进行膨胀腐蚀操作
    # edge_img = cv2.dilate(cv2.erode(edge_img, kernel, iterations=1), kernel, iterations=1)
    result = cv2.erode(cv2.dilate(edge_img, kernel, iterations=1), kernel, iterations=1)
    # result = cv2.bitwise_not(result)
    # # 将当前轮廓的结果添加到最终结果中
    # result = cv2.add(result, eroded)
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
    cv2.drawContours(sub_img, [max_contour], -1, (0, 255, 0), 1)
    # x, y, w, h = cv2.boundingRect(max_contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=1)
    # cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 1)

    cv2.imshow("origin", sub_img)
    cv2.imshow("threshold", inverted_binary_image)
    cv2.imshow("edge", result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
