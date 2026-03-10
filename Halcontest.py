import cv2
import numpy as np
import halcon as ha

CameraParameters = ['area_scan_polynomial', 0.0279315, 138.987, 372942, -5.97038e+08, 0.360096, -2.7727, 2.25651e-06,
                    2.4e-06, 3043.65, -2540.04, 5472, 3648]
CameraPose = [-0.214615, 0.388094, 1.37056, 1.70252, 3.61043, 359.908, 0]
# # 相机内参矩阵
# fx = 0.0215833
# fy = 0.0215833
# cx = 2619.92
# cy = 1839.53
#
# camera_matrix = np.array([[fx, 0, cx],
#                           [0, fy, cy],
#                           [0, 0, 1]])
#
# rotation_matrix = np.array([[0.999342, -0.004720, -0.035832],
#                             [0.004825, 0.999988, 0.001235],
#                             [0.035818, -0.001280, 0.999355]])
#
# translation_vector = np.array([-0.294264, -0.206406, 2.98153])
X = [63.8, 91.6, 119.4, 146.9, 174.1, 207.4, 231.1, 256.4, 282, 313.3, 335.7, 391.7, 454.8, 503.6, 564, 1084.6, 2820]
Y = [48, 68.8, 89.5, 110.4, 130.9, 151.4, 173.8, 192.7, 212, 235.5, 252.4, 294.4, 341.9, 378.6, 424, 815.4, 2120]  # y坐标
W_D = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 2000, 5000]
# p11 = (676, 2058)
# p12 = (719, 2092)
# p21 = (1279, 1309)
# p22 = (1308, 1345)
# p31 = (2094, 2967)
# p32 = (2110, 2981)
# index = W_D.index(500)
# x_D, y_D = X[index], Y[index]
# Sx = x_D / 5472
# Sy = y_D / 3648
# area_p = Sx * Sy
# area = area_p*2609/1.8
# # X1, Y1 = ha.image_points_to_world_plane(CameraParameters, CameraPose, 2058, 676, 'mm')
# # X2, Y2 = ha.image_points_to_world_plane(CameraParameters, CameraPose, 2092, 719, 'mm')
#
# # Distance1 = ha.distance_pp(X1, Y1, X1, Y2)
# # Distance2 = ha.distance_pp(X2, Y1, X2, Y2)
# # area = Distance1[0] * Distance2[0]
# # print(f"X1:{X1}X2:{X2}Y1:{Y1}Y2:{Y2}")
# print("面积为:", area)
