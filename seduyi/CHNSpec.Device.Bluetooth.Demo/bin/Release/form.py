import os
import sys
import tkinter as tk
from ctypes import WinDLL
from tkinter import messagebox
import clr

# clr.AddReference("System.Windows.Forms")
# clr.AddReference("System.Drawing")
# from System.Windows.Forms import *
# from System.Drawing import *

clr.AddReference("CHNSpec.Device.Models.dll")
clr.AddReference("CHNSpec.Device.Bluetooth.dll")

from CHNSpec.Device.Models import DeviceInfo, ConnectionStatus, EnumMeasure_Mode, EnumCaliberateType
from CHNSpec.Device.Models.Enums import BLECode
from CHNSpec.Device.Bluetooth import DeviceCallback, BluetoothHelper


class Form1(tk.Tk):
    def __init__(self):
        super().__init__()

        # 蓝牙设备列表
        self.bluetoothList = []

        self.helper = BluetoothHelper()

        # true表示正在查找蓝牙设备，false表示已停止查找
        self.isDiscovering = False

        self.title("CHNSpec Device Bluetooth Demo")
        self.geometry("600x400")

        self.cmb_multilingual = tk.StringVar(value="Chinese")
        self.cmb_multilingual.trace_add("write", self.on_language_change)
        self.cmb_multilingual_options = ["Chinese", "English"]
        self.cmb_multilingual_menu = tk.OptionMenu(self, self.cmb_multilingual, *self.cmb_multilingual_options)
        self.cmb_multilingual_menu.grid(row=0, column=0, padx=10, pady=10)

        self.lab_state = tk.Label(self, text="未连接", font=("Helvetica", 16))
        self.lab_state.grid(row=1, column=0, padx=10, pady=10)

        self.listBox1 = tk.Listbox(self)
        self.listBox1.grid(row=2, column=0, padx=10, pady=10)

        self.btn_start = tk.Button(self, text="开始搜索", command=self.start_search)
        self.btn_start.grid(row=3, column=0, padx=10, pady=10)

        self.btn_end = tk.Button(self, text="停止搜索蓝牙", command=self.stop_search)
        self.btn_end.grid(row=4, column=0, padx=10, pady=10)

        self.btn_connect = tk.Button(self, text="连接/断开", command=self.connect_disconnect)
        self.btn_connect.grid(row=5, column=0, padx=10, pady=10)

        self.btn_measure = tk.Button(self, text="测量", command=self.measure)
        self.btn_measure.grid(row=6, column=0, padx=10, pady=10)

        self.btn_blackcalibration = tk.Button(self, text="黑校准", command=self.black_calibration)
        self.btn_blackcalibration.grid(row=7, column=0, padx=10, pady=10)

        self.btn_white_calibration = tk.Button(self, text="白校准", command=self.white_calibration)
        self.btn_white_calibration.grid(row=8, column=0, padx=10, pady=10)

        self.textBox1 = tk.Text(self, height=10, width=50)
        self.textBox1.grid(row=2, column=1, rowspan=8, padx=10, pady=10)

        # 蓝牙回调
        if self.helper.bleCode is not None:
            # 查找到的蓝牙设备
            self.helper.bleCode.Added = self.add_bluetooth_device

            # 停止查找蓝牙设备
            self.helper.bleCode.WatcherStopped = self.watcher_stopped

            # 查找完设备完成
            self.helper.bleCode.EnumerationCompleted = self.enumeration_completed

        # 订阅仪器连接状态
        DeviceCallback.ConnectionChangeCallback = self.on_connection_change

        # 订阅测量状态
        DeviceCallback.MeasureCallback = self.on_measure

        # 订阅校准状态
        DeviceCallback.CalibrateCallback = self.on_calibrate

    def on_language_change(self, *args):
        lang = self.cmb_multilingual.get()
        if lang == "Chinese":
            # Set Chinese language
            self.lab_state.config(text="未连接")
        else:
            # Set English language
            self.lab_state.config(text="Not connected")

    def add_bluetooth_device(self, data):
        if not self.winfo_exists():
            return

        device_info = DeviceInfo(
            Address=data.Address,
            DeviceId=data.DeviceId,
            IsPaired=data.IsPaired,
            Name=data.Name,
            State=ConnectionStatus.Disconnected,
            Type=data.Type
        )
        if device_info not in self.bluetoothList:
            self.bluetoothList.append(device_info)
            self.listBox1.insert(tk.END, device_info.Name)

    def watcher_stopped(self):
        pass

    def enumeration_completed(self):
        pass

    def on_connection_change(self, state):
        if self.cmb_multilingual.get() == "Chinese":
            self.lab_state.config(text="已连接" if state else "未连接")
        else:
            self.lab_state.config(text="Connected" if state else "Not connected")

    def on_measure(self, state, result):
        msg = ""
        if state:
            if self.cmb_multilingual.get() == "Chinese":
                spectrum = "\n".join(
                    f"测量模式：{item.measure_mode}\n光谱信息：{', '.join(map(str, item.spectral_data))}" for item in
                    result.spectrums
                )
                msg = f"测量成功\nUV模式：{result.uv_mode}\n口径大小：{result.caliber_size}\n开始波长：{result.wave_start}\n波长间隔：{result.wave_interval}\n波长数量：{result.wave_number}\n{spectrum}"
            else:
                spectrum = "\n".join(
                    f"Measurement mode：{item.measure_mode}\nspectral information：{', '.join(map(str, item.spectral_data))}"
                    for item in result.spectrums
                )
                msg = f"Measurement successful\nUV mode：{result.uv_mode}\nCaliber size：{result.caliber_size}\nStarting wavelength：{result.wave_start}\nWavelength spacing：{result.wave_interval}\nNumber of wavelengths：{result.wave_number}\n{spectrum}"
        else:
            if self.cmb_multilingual.get() == "Chinese":
                msg = "测量失败\n\n"
            else:
                msg = "Measurement failed\n\n"
        self.textBox1.insert(tk.END, msg + "\n\n")
        with open('./colormsg.txt', 'w') as f:
            f.write(msg)

    def on_calibrate(self, state):
        if self.cmb_multilingual.get() == "Chinese":
            messagebox.showinfo("提示", "校准成功" if state else "校准失败")
        else:
            messagebox.showinfo("Information", "Calibration successful" if state else "Calibration failed")

    def start_search(self):
        self.listBox1.delete(0, tk.END)
        self.bluetoothList.clear()
        self.helper.bleCode.StartBleDeviceWatcher("CM")

    def stop_search(self):
        self.helper.bleCode.StopBleDeviceWatcher()

    def connect_disconnect(self):
        selected_index = self.listBox1.curselection()
        if not selected_index:
            if self.cmb_multilingual.get() == "Chinese":
                messagebox.showinfo("提示", "请先选择仪器")
            else:
                messagebox.showinfo("Information", "Please select the instrument first")
            return
        device_info = self.bluetoothList[selected_index[0]]
        result = self.helper.OpenBluetooth(device_info.DeviceId, device_info.Name)
        if not result:
            messagebox.showinfo("提示",
                                "连接失败" if self.cmb_multilingual.get() == "Chinese" else "Connection failed")

    def measure(self):
        if self.helper.Send_MeasureCmd(EnumMeasure_Mode.SCI, 0):
            messagebox.showinfo("提示",
                                "测量下发成功" if self.cmb_multilingual.get() == "Chinese" else "Measurement sent successfully")

    def black_calibration(self):
        if self.helper.Send_CalibrateCmd(EnumCaliberateType.Black):
            messagebox.showinfo("提示",
                                "黑校准下发成功" if self.cmb_multilingual.get() == "Chinese" else "Black calibration sent successfully")

    def white_calibration(self):
        if self.helper.Send_CalibrateCmd(EnumCaliberateType.White):
            messagebox.showinfo("提示",
                                "白校准下发成功" if self.cmb_multilingual.get() == "Chinese" else "White calibration sent successfully")


if __name__ == "__main__":
    app = Form1()
    app.mainloop()
