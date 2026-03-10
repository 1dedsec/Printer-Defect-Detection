# 查验导入数据是否是图片
def file_is_pic(suffix):
    if suffix == "png" or suffix == "jpg" or suffix == "bmp":
        return True
    return False


import threading


# 全局变量
class Globals:
    export_file = [False, False, False]
    camera_running = False
    stop_sign = False
    visible = False
    Glock = threading.Lock()
    nums = []
    boxes_count = [[],[],[]]
    sign = False
