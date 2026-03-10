import math

XYZ2LABCH_10_D65 = [94.81, 100.00, 107.32]


def XYZ2LABCH(XYZ, lightsource_type, observer):
    L, a, b, C, H, X, Y, Z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    temX, temY, temZ = 0.0, 0.0, 0.0
    labch = [0, 0, 0, 0, 0]
    temX = XYZ[0] / XYZ2LABCH_10_D65[0]  # D65װXֵ
    temY = XYZ[1] / XYZ2LABCH_10_D65[1]  # D65װYֵ
    temZ = XYZ[2] / XYZ2LABCH_10_D65[2]  # D65װZֵ

    if temX > 0.008856:
        temX = math.pow(temX, 0.3333333)
    else:
        temX = (7.787 * temX) + 0.138
    if temY > 0.008856:
        temY = math.pow(temY, 0.3333333)
        L = 116 * temY - 16
    else:
        L = 903.3 * temY
        temY = (7.787 * temY) + 0.138
    if temZ > 0.008856:
        temZ = math.pow(temZ, 0.3333333)
    else:
        temZ = (7.787 * temZ) + 0.138
    a = 500.0 * (temX - temY)
    b = 200.0 * (temY - temZ)

    if L < 0:
        L = 0.00
    C = math.sqrt(a * a + b * b)
    if (a == 0) and (b > 0):
        H = 90
    elif (a == 0) and (b < 0):
        H = 270
    elif (a >= 0) and (b == 0):
        H = 0
    elif (a < 0) and (b == 0):
        H = 180
    else:
        H = math.atan(b / a)
        H = H * 57.3
        if (a > 0) and (b > 0):
            H = H
        elif a < 0:
            H = 180 + H
        else:
            H = 360 + H

    labch[0] = L
    labch[1] = a
    labch[2] = b
    labch[3] = C
    labch[4] = H
    return labch
