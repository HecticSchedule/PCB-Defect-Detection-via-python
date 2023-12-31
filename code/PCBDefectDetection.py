# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File    : PCBDefectDetection.py
# @Time    : 2023/12/4
# @Author  : HecticSchedule
# @Version : python3.9
# @Desc    : PCB defect detection though python.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# 获取PCB型号
def get_pcb_model(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 2:
        model_number = int(parts[0])
        defect_names = parts[1:-1]
        image_number = int(parts[-1].split('.')[0])
        return model_number, defect_names, image_number
    else:
        return None, None, None


# 加载模板图片
def load_template_images(model_number):
    template_img_path = f'../images/Template/{model_number:02d}.jpg'
    print(f"Template image path: {template_img_path}")

    if os.path.exists(template_img_path):
        template_img = cv.imread(template_img_path)
        print("Template image loaded successfully.")
        if model_number == 1:
            model_threshold = {
                'soldering_pads': (150, 255),
                'wire_tracks': (47, 150),
                'background': (0, 47),
            }

            defect_names = {
                'positive_defects': ['Missing_hole', 'Open_circuit',
                                     'Short', 'Spurious_copper'],
                'negative_defects': ['Missing_hole', 'Open_circuit',
                                     'Short', 'Spurious_copper'],
            }
        elif model_number == 4:
            model_threshold = {
                'soldering_pads': (120, 255),
                'wire_tracks': (40, 120),
                'background': (0, 40),
            }

            defect_names = {
                'positive_defects': ['Missing_hole', 'Open_circuit',
                                     'Short', 'Spurious_copper'],
                'negative_defects': ['Missing_hole', 'Open_circuit',
                                     'Short', 'Spurious_copper'],
            }
        else:
            print(f"Invalid model number. Exiting.")
            return None, None, None
        return template_img, model_threshold, defect_names
    else:
        print(f"Template image not found for model {model_number}. Exiting.")
        return None, None, None


# 图片处理
def process_pcb_image(template_img,
                      test_img,
                      model_threshold,
                      detected_defects):
    # 图片灰度化
    gray_test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    gray_template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)

    # ---- 图像预处理 ----
    # 用核为7 * 7大小模型去除椒盐噪声
    med_test = cv.medianBlur(gray_test_img, 7)
    med_template = cv.medianBlur(gray_template_img, 7)

    # 使用高斯滤波器(sigma = 1)来抑制图像内的高强度变化
    gaus_test = cv.GaussianBlur(med_test, ksize=(3, 3), sigmaX=1)
    gaus_template = cv.GaussianBlur(med_template, ksize=(3, 3), sigmaX=1)

    # ---- 图像分割 ----
    # 使用特定型号的阈值分割图像
    sold_test = cv.inRange(gaus_test,
                           model_threshold['soldering_pads'][0],
                           model_threshold['soldering_pads'][1])
    sold_template = cv.inRange(gaus_template,
                               model_threshold['soldering_pads'][0],
                               model_threshold['soldering_pads'][1])
    wire_test = cv.inRange(gaus_test,
                           model_threshold['wire_tracks'][0],
                           model_threshold['wire_tracks'][1])
    wire_template = cv.inRange(gaus_template,
                               model_threshold['wire_tracks'][0],
                               model_threshold['wire_tracks'][1])
    bg_test = cv.inRange(gaus_test,
                         model_threshold['background'][0],
                         model_threshold['background'][1])
    bg_template = cv.inRange(gaus_template,
                             model_threshold['background'][0],
                             model_threshold['background'][1])

    # 处理一下分割后的线迹
    kernel = np.ones((7, 7))
    open_wire_test = cv.morphologyEx(wire_test, cv.MORPH_OPEN, kernel)
    open_wire_template = cv.morphologyEx(wire_template, cv.MORPH_OPEN, kernel)

    # 处理焊点
    kernel = np.ones((13, 3))
    close_sold_test = cv.morphologyEx(sold_test, cv.MORPH_CLOSE, kernel)
    close_sold_template = cv.morphologyEx(sold_template, cv.MORPH_CLOSE, kernel)

    # 焊点反色
    sold_test_fill = close_sold_test.copy()
    sold_template_fill = close_sold_template.copy()

    h, w = sold_test_fill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(sold_test_fill, mask, (0, 0), 255);
    hole_test = cv.bitwise_not(sold_test_fill)

    h, w = sold_template_fill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(sold_template_fill, mask, (0, 0), 255);
    hole_template = cv.bitwise_not(sold_template_fill)

    # ---- 错误识别 ---- The segmented images (wiring tracks, soldering pads and holes) of test and template images
    # differ from each other due to defects in testing PCB image. So, the defects can be simply detected by image
    # subtraction. These defects are of two types:
    # (1) positive defects (PD) and
    # (2) negative defects (ND)
    # PDi = testingi - templatei
    # NDi = templatei - testingi

    # 焊盘识别
    pd_sold = sold_test - sold_template
    nd_sold = sold_template - sold_test

    kernel = np.ones((3, 3))
    open_pd_sold = cv.morphologyEx(pd_sold, cv.MORPH_OPEN, kernel)
    open_nd_sold = cv.morphologyEx(nd_sold, cv.MORPH_OPEN, kernel)

    # 线迹识别
    pd_wire = open_wire_test - open_wire_template
    nd_wire = open_wire_template - open_wire_test

    kernel = np.ones((3, 3))
    open_pd_wire = cv.morphologyEx(pd_wire, cv.MORPH_OPEN, kernel)
    open_nd_wire = cv.morphologyEx(nd_wire, cv.MORPH_OPEN, kernel)

    # 焊点识别
    pd_hole = hole_test - hole_template
    nd_hole = hole_template - hole_test

    kernel = np.ones((3, 3))
    open_nd_hole = cv.morphologyEx(nd_hole, cv.MORPH_OPEN, kernel)

    # ---- 结果展示 ----
    defects = test_img.copy()

    defects[open_pd_sold == 255] = [0, 0, 255]
    defects[open_pd_wire == 255] = [0, 0, 255]
    defects[pd_hole == 255] = [0, 0, 255]

    defects[open_nd_sold == 255] = [255, 0, 0]
    defects[open_nd_wire == 255] = [255, 0, 0]
    defects[open_nd_hole == 255] = [255, 0, 0]

    return defects, detected_defects


def detect_pcb_defects(test_img_path):
    model_number, detected_defects, _ = get_pcb_model(os.path.basename(test_img_path))

    if model_number:
        template_img, model_threshold, _ = load_template_images(model_number)

        if template_img is not None:
            result, detected_defects = process_pcb_image(
                template_img,
                # cv.imread(test_img_path),
                cv.imdecode(np.fromfile(test_img_path, dtype=np.uint8), -1),
                model_threshold,
                detected_defects
            )

            return result, detected_defects
    else:
        return None, None


if __name__ == '__main__':
    test_img_path = input("Enter the path of the test PCB image: ")
    result, detected_defects = detect_pcb_defects(test_img_path)

    if result is not None and detected_defects is not None:
        print(f"Detected defects: {' '.join(detected_defects)}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Defective PCB with {" ".join(detected_defects)} Defects')
        plt.show()
    else:
        print("Error detecting defects.")