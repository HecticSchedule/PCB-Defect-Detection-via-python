# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File    : GUI.py
# @Time    : 2023/12/31 21:54
# @Author  : HecticSchedule
# @Version : python 3.9
# @Desc    :
"""

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys
from PCBDefectDetection import detect_pcb_defects


# 定义 PCB 缺陷检测 GUI 类
class PCBDefectDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()

        # 创建控件
        self.result_label = QLabel("PCB Defect Detection")  # 显示检测结果的标签
        self.result_image_label = QLabel()  # 显示检测结果图像的标签
        self.result_image_label.setAlignment(Qt.AlignCenter)  # 图像居中显示
        self.select_button = QPushButton("Select PCB Image")  # 选择 PCB 图像的按钮
        self.select_button.clicked.connect(self.select_image)  # 连接按钮点击事件到选择图像的方法

        # 设置窗口图标
        self.setWindowIcon(QIcon('../images/icon.ico'))
        # 设置窗口标题
        self.setWindowTitle("PCBDefectDetectionGUI")

        # 创建垂直布局管理器
        main_layout = QVBoxLayout(self)
        # 添加上方标签到布局
        main_layout.addWidget(self.result_label, alignment=Qt.AlignTop | Qt.AlignHCenter)
        # 创建水平布局管理器
        image_layout = QHBoxLayout()
        # 添加中间图像标签到水平布局
        image_layout.addWidget(self.result_image_label, alignment=Qt.AlignCenter)
        # 将水平布局添加到垂直布局
        main_layout.addLayout(image_layout)
        # 添加下方按钮到布局
        self.select_button.setFixedSize(700, 50)  # 调整按钮大小
        main_layout.addWidget(self.select_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

        # 设置窗口大小
        self.setFixedSize(770, 770)

        # 美化界面样式表
        self.setStyleSheet("""
            QWidget {
                background-color: #F0F0F0; /* 设置背景颜色 */
                font-family: "Microsoft YaHei"; /* 设置全局字体 */
            }
            QLabel {
                color: #333333; /* 设置文本颜色 */
                font-size: 18px; /* 设置字体大小 */
            }
            QPushButton {
                background-color: #4CAF50; /* 设置按钮背景颜色 */
                color: white; /* 设置按钮文本颜色 */
                font-size: 16px; /* 设置按钮字体大小 */
                font-family: "Helvetica"; /* 设置按钮字体 */
            }
            /* 添加其他样式属性 */
        """)

    # 选择图像的方法
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PCB Image", "", "Image files (*.png; *.jpg; *.jpeg)")
        if file_path:
            result_image, detected_defects = self.run_defect_detection(file_path)
            self.show_result(result_image, detected_defects)

    # 运行 PCB 缺陷检测的方法
    def run_defect_detection(self, image_path):
        print("PCB Image Path:", image_path)
        result_image, detected_defects = detect_pcb_defects(image_path)
        print("Detected Defects:", detected_defects)
        return result_image, detected_defects

    # 显示检测结果的方法
    def show_result(self, result_image, detected_defects):
        # 将 NumPy 数组转换为 QImage
        height, width, channel = result_image.shape
        q_image = QImage(result_image.data, width, height, width * channel, QImage.Format_RGB888)

        # 将 QImage 缩放到合适的大小
        scaled_image = q_image.scaled(800, 600, Qt.KeepAspectRatio)

        # 在 QLabel 中显示缩放后的图片
        self.result_image_label.setPixmap(QPixmap.fromImage(scaled_image))
        self.result_image_label.setScaledContents(True)

        # 显示检测到的缺陷
        result_text = f"检测到的缺陷: {' '.join(detected_defects)}"
        self.result_label.setText(result_text)

    # 调试窗口大小时的重写函数
    def resizeEvent(self, event):
        new_size = event.size()
        print(f"当前窗口大小：{new_size.width()} x {new_size.height()}")
        # 调用父类的 resizeEvent 方法以确保正常的事件处理
        super().resizeEvent(event)


# 主程序入口
def main():
    app = QApplication(sys.argv)
    gui = PCBDefectDetectionGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
