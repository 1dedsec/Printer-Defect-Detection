namespace CHNSpec.Device.Bluetooth.Demo
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btn_end = new System.Windows.Forms.Button();
            this.listBox1 = new System.Windows.Forms.ListBox();
            this.lab_state = new System.Windows.Forms.Label();
            this.btn_connect = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.btn_start = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.btn_white_calibration = new System.Windows.Forms.Button();
            this.btn_blackcalibration = new System.Windows.Forms.Button();
            this.btn_measure = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.button6 = new System.Windows.Forms.Button();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.cmb_multilingual = new System.Windows.Forms.ComboBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            resources.ApplyResources(this.groupBox1, "groupBox1");
            this.groupBox1.Controls.Add(this.btn_end);
            this.groupBox1.Controls.Add(this.listBox1);
            this.groupBox1.Controls.Add(this.lab_state);
            this.groupBox1.Controls.Add(this.btn_connect);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.btn_start);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.TabStop = false;
            // 
            // btn_end
            // 
            resources.ApplyResources(this.btn_end, "btn_end");
            this.btn_end.Name = "btn_end";
            this.btn_end.UseVisualStyleBackColor = true;
            this.btn_end.Click += new System.EventHandler(this.btn_end_Click);
            // 
            // listBox1
            // 
            resources.ApplyResources(this.listBox1, "listBox1");
            this.listBox1.FormattingEnabled = true;
            this.listBox1.Name = "listBox1";
            // 
            // lab_state
            // 
            resources.ApplyResources(this.lab_state, "lab_state");
            this.lab_state.Name = "lab_state";
            // 
            // btn_connect
            // 
            resources.ApplyResources(this.btn_connect, "btn_connect");
            this.btn_connect.Name = "btn_connect";
            this.btn_connect.UseVisualStyleBackColor = true;
            this.btn_connect.Click += new System.EventHandler(this.btn_connect_Click);
            // 
            // label2
            // 
            resources.ApplyResources(this.label2, "label2");
            this.label2.Name = "label2";
            // 
            // btn_start
            // 
            resources.ApplyResources(this.btn_start, "btn_start");
            this.btn_start.Name = "btn_start";
            this.btn_start.UseVisualStyleBackColor = true;
            this.btn_start.Click += new System.EventHandler(this.btn_start_Click);
            // 
            // groupBox2
            // 
            resources.ApplyResources(this.groupBox2, "groupBox2");
            this.groupBox2.Controls.Add(this.btn_white_calibration);
            this.groupBox2.Controls.Add(this.btn_blackcalibration);
            this.groupBox2.Controls.Add(this.btn_measure);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.TabStop = false;
            // 
            // btn_white_calibration
            // 
            resources.ApplyResources(this.btn_white_calibration, "btn_white_calibration");
            this.btn_white_calibration.Name = "btn_white_calibration";
            this.btn_white_calibration.UseVisualStyleBackColor = true;
            this.btn_white_calibration.Click += new System.EventHandler(this.btn_white_calibration_Click);
            // 
            // btn_blackcalibration
            // 
            resources.ApplyResources(this.btn_blackcalibration, "btn_blackcalibration");
            this.btn_blackcalibration.Name = "btn_blackcalibration";
            this.btn_blackcalibration.UseVisualStyleBackColor = true;
            this.btn_blackcalibration.Click += new System.EventHandler(this.btn_blackcalibration_Click);
            // 
            // btn_measure
            // 
            resources.ApplyResources(this.btn_measure, "btn_measure");
            this.btn_measure.Name = "btn_measure";
            this.btn_measure.UseVisualStyleBackColor = true;
            this.btn_measure.Click += new System.EventHandler(this.btn_measure_Click);
            // 
            // groupBox3
            // 
            resources.ApplyResources(this.groupBox3, "groupBox3");
            this.groupBox3.Controls.Add(this.button6);
            this.groupBox3.Controls.Add(this.textBox1);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.TabStop = false;
            // 
            // button6
            // 
            resources.ApplyResources(this.button6, "button6");
            this.button6.Name = "button6";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // textBox1
            // 
            resources.ApplyResources(this.textBox1, "textBox1");
            this.textBox1.Name = "textBox1";
            // 
            // label4
            // 
            resources.ApplyResources(this.label4, "label4");
            this.label4.Name = "label4";
            // 
            // cmb_multilingual
            // 
            resources.ApplyResources(this.cmb_multilingual, "cmb_multilingual");
            this.cmb_multilingual.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmb_multilingual.FormattingEnabled = true;
            this.cmb_multilingual.Items.AddRange(new object[] {
            resources.GetString("cmb_multilingual.Items"),
            resources.GetString("cmb_multilingual.Items1")});
            this.cmb_multilingual.Name = "cmb_multilingual";
            this.cmb_multilingual.SelectedIndexChanged += new System.EventHandler(this.cmb_multilingual_SelectedIndexChanged);
            // 
            // Form1
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.cmb_multilingual);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button btn_start;
        private System.Windows.Forms.Button btn_connect;
        private System.Windows.Forms.Label lab_state;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Button btn_blackcalibration;
        private System.Windows.Forms.Button btn_measure;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Button btn_white_calibration;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.ListBox listBox1;
        private System.Windows.Forms.Button btn_end;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox cmb_multilingual;
    }
}

