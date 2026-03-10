using BLECode;
using CHNSpec.Device.Models;
using CHNSpec.Device.Models.Enums;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;

namespace CHNSpec.Device.Bluetooth.Demo
{
    public partial class Form1 : Form
    {
        /// <summary>
        /// 蓝牙设备列表
        /// </summary>
        List<DeviceInfo> bluetoothList = new List<DeviceInfo>();

        public BluetoothHelper helper = new BluetoothHelper();


        /// <summary>
        /// true表示正在查找蓝牙设备，false表示已停止查找
        /// </summary>
        public bool isDiscovering = false;


        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            cmb_multilingual.SelectedIndex = 0;

            #region 蓝牙回调

            if (helper.bleCode != null)
            {
                //查找到的蓝牙设备
                helper.bleCode.Added = AddBluetoothDevice;

                //停止查找蓝牙设备
                helper.bleCode.WatcherStopped = WatcherStopped;

                //查找完设备完成
                helper.bleCode.EnumerationCompleted = EnumerationCompleted;
            }
            #endregion


            //订阅仪器连接状态
            DeviceCallback.ConnectionChangeCallback = (state) =>
            {
                this.Invoke(new Action(() =>
                {
                    if (cmb_multilingual.SelectedIndex == 0)
                    {
                        lab_state.Text = state ? "已连接" : "未连接";
                    }
                    else
                    {
                        lab_state.Text = state ? "Connected" : "Not connected";
                    }
                }));

            };


            //订阅测量状态
            DeviceCallback.MeasureCallback = (state, result) =>
            {
                this.Invoke(new Action(() =>
                {
                    string msg;
                    if (state)
                    {
                        if (cmb_multilingual.SelectedIndex == 0)
                        {
                            string spectrum = string.Empty;
                            foreach (var item in result.spectrums)
                            {
                                spectrum += "测量模式：" + item.measure_mode.ToString() + Environment.NewLine;
                                spectrum += "光谱信息：" + string.Join(",", item.spectral_data) + Environment.NewLine;
                            }
                            msg = $"测量成功{ Environment.NewLine}UV模式：{result.uv_mode + Environment.NewLine}口径大小：{result.caliber_size + Environment.NewLine}开始波长：{result.wave_start + Environment.NewLine}波长间隔：{result.wave_interval + Environment.NewLine}波长数量：{result.wave_number + Environment.NewLine}{spectrum}";
                        }
                        else
                        {
                            string spectrum = string.Empty;
                            foreach (var item in result.spectrums)
                            {
                                spectrum += "Measurement mode：" + item.measure_mode.ToString() + Environment.NewLine;
                                spectrum += "spectral information：" + string.Join(",", item.spectral_data) + Environment.NewLine;
                            }
                            msg = $"Measurement successful{ Environment.NewLine}UV mode：{result.uv_mode + Environment.NewLine}Caliber size：{result.caliber_size + Environment.NewLine}Starting wavelength：{result.wave_start + Environment.NewLine}Wavelength spacing：{result.wave_interval + Environment.NewLine}Number of wavelengths：{result.wave_number + Environment.NewLine}{spectrum}";
                        }

                    }
                    else
                    {
                        if (cmb_multilingual.SelectedIndex == 0)
                        {
                            msg = "测量失败" + Environment.NewLine + Environment.NewLine;
                        }
                        else
                        {
                            msg = "Measurement failed" + Environment.NewLine + Environment.NewLine;
                        }
                    }
                    textBox1.Text += msg + Environment.NewLine + Environment.NewLine;
                    System.IO.File.WriteAllText(@"./colormsg.txt", msg);
                }));
            };

            //订阅校准状态
            DeviceCallback.CalibrateCallback = (state) =>
            {
                if (cmb_multilingual.SelectedIndex == 0)
                {
                    MessageBox.Show(state ? "校准成功" : "校准失败");
                }
                else
                {
                    MessageBox.Show(state ? "Calibration successful" : "Calibration failed");
                }
            };
        }



        /// <summary>
        /// 新增蓝牙设备
        /// </summary>
        private void AddBluetoothDevice(DeviceInfo data)
        {
            if (!IsHandleCreated) return;


            DeviceInfo deviceInfo = new DeviceInfo()
            {
                Address = data.Address,
                DeviceId = data.DeviceId,
                IsPaired = data.IsPaired,
                Name = data.Name,
                State = ConnectionStatus.Disconnected,
                Type = data.Type,
            };
            if (!bluetoothList.Contains(deviceInfo))
            {
                bluetoothList.Add(deviceInfo);
                this.Invoke(new Action(() =>
                {
                    if (!IsHandleCreated) return;

                    listBox1.Items.Add(deviceInfo.Name);
                }));
            }
        }

        /// <summary>
        /// 停止查找蓝牙设备
        /// </summary>
        private void WatcherStopped()
        {
            if (!IsHandleCreated) return;

            this.Invoke(new Action(() =>
            {

            }));
        }

        /// <summary>
        /// 查找完设备完成
        /// </summary>
        private void EnumerationCompleted()
        {
            if (!IsHandleCreated) return;
            this.Invoke(new Action(() =>
            {

            }));
        }




        /// <summary>
        /// 连接/断开
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_connect_Click(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex < 0)
            {
                if (cmb_multilingual.SelectedIndex == 0)
                {
                    MessageBox.Show("请先选择仪器");
                }
                else
                {
                    MessageBox.Show("Please select the instrument first");
                }
                return;
            }

            DeviceInfo deviceInfo = bluetoothList[listBox1.SelectedIndex];


            bool result = helper.OpenBluetooth(deviceInfo.DeviceId, deviceInfo.Name);
            if (!result)
            {
                MessageBox.Show("connection failed");
                return;
            }
        }

        /// <summary>
        /// 测量
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_measure_Click(object sender, EventArgs e)
        {
            if (helper.Send_MeasureCmd(EnumMeasure_Mode.SCI, 0))
            {
                if (cmb_multilingual.SelectedIndex == 0)
                {
                    MessageBox.Show("测量下发成功");
                }
                else
                {
                    MessageBox.Show("Measurement sent successfully");
                }
            }
        }

        /// <summary>
        /// 黑校准
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_blackcalibration_Click(object sender, EventArgs e)
        {
            if (helper.Send_CalibrateCmd(EnumCaliberateType.Black))
            {
                if (cmb_multilingual.SelectedIndex == 0)
                {
                    MessageBox.Show("校准下发成功");
                }
                else
                {
                    MessageBox.Show("Calibration sent successfully");
                }
            }

        }

        /// <summary>
        /// 白校准
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_white_calibration_Click(object sender, EventArgs e)
        {
            if (helper.Send_CalibrateCmd(EnumCaliberateType.White))
            {
                if (cmb_multilingual.SelectedIndex == 0)
                {
                    MessageBox.Show("校准下发成功");
                }
                else
                {
                    MessageBox.Show("Calibration sent successfully");
                }
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            textBox1.Text = string.Empty;
        }

        /// <summary>
        /// 开始搜索
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_start_Click(object sender, EventArgs e)
        {
            listBox1.Items.Clear();
            bluetoothList.Clear();

            helper.bleCode.StartBleDeviceWatcher("CM");
        }

        /// <summary>
        /// 停止搜索蓝牙
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_end_Click(object sender, EventArgs e)
        {
            helper.bleCode.StopBleDeviceWatcher();
        }

        /// <summary>
        /// 切换多语言 
        /// Switch multiple languages
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmb_multilingual_SelectedIndexChanged(object sender, EventArgs e)
        {
            string lang;
            switch (cmb_multilingual.SelectedIndex)
            {
                case 1:
                    lang = "en-US";
                    break;
                default:
                    lang = "zh-CN";
                    break;
            }

            JapanHelper.SetLang(lang, this);
        }
    }
}
