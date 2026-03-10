
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 300)

        open_dialog_button = QPushButton("Open Dialog", self)
        open_dialog_button.clicked.connect(self.open_dialog)

    def open_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Dialog")
        dialog.setGeometry(200, 200, 850, 1000)

        # 创建一个 QTableWidget 并放置在 QDialog 上
        table_widget = QTableWidget(dialog)
        table_widget.setColumnCount(4)

        # 填充表格单元格
        for row in range(5):
            for col in range(3):
                item = QTableWidgetItem(f"Row {row + 1}, Col {col + 1}")
                table_widget.setItem(row, col, item)

        # 创建一个布局并将表格小部件添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(table_widget)

        # 设置 QDialog 的布局
        dialog.setLayout(layout)

        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
