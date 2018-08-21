#!/usr/bin/env python

import sys
import csv
import colorsys

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

data = {}
for i in range(20):
    with open('./data/itr_1710/path%02d.csv' % i, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        data[i] = []
        for n, row in enumerate(spamreader):
            if n != 0:
                 data[i].append([float(el) for el in row])
        data[i] = np.array(data[i])

class MyCanvas(QWidget):
    def __init__(self, parent):
        super(MyCanvas, self).__init__(parent)
        self.no = 0
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        for i in range(len(data[self.no])):
            if i == len(data[self.no]) - 1: break
            p11 = data[self.no][i][0] * 360 * 0.5
            p12 = data[self.no][i][2] * 360 * 0.5
            p21 = data[self.no][i+1][0] * 360 * 0.5
            p22 = data[self.no][i+1][2] * 360 * 0.5

            v = (data[self.no][i+1][4] ** 2 + data[self.no][i+1][5] ** 2) ** (1/2)
            rgb = hsv2rgb(0.5, 0.5, v)
            
            pen = QPen(QColor(rgb[0], rgb[1], rgb[2]), 3)
            painter.setPen(pen)
            painter.drawLine(p21, p22, p11, p12)

class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setWindowTitle("Hello World!")
        self.width, self.height = 380, 400
        self.resize(self.width, self.height)
        
        self.widget = MyCanvas(self)
        self.widget.setGeometry(10, 10, 360, 360)
        self.widget.setStyleSheet("background-color:grey;")

        self.button = QPushButton('Next', self)
        self.button.setGeometry(10, 375, 360, 20)
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        self.widget.no = (self.widget.no + 1) % 20
        self.widget.repaint()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
