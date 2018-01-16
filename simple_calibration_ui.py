from PyQt5.QtWidgets import (QApplication, QGraphicsView,
        QGraphicsPixmapItem, QGraphicsScene, QDesktopWidget)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import (QObject, QPointF,
        QPropertyAnimation, pyqtProperty)
import sys
import cv2
import collections
import math
import pyautogui
import os
import time

class Ball(QObject):
    def __init__(self):
        super().__init__()

        self.pixmap_item = QGraphicsPixmapItem(QPixmap("red_ball.png"))

    def _set_pos(self, pos):
        self.pixmap_item.setPos(pos)

    pos = pyqtProperty(QPointF, fset=_set_pos)


class Example(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.initView()

    def initView(self):
        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()
        x = ag.width()
        y = ag.height()

        self.ball = Ball()

        self.anim = QPropertyAnimation(self.ball, b'pos')
        self.anim.setDuration(10000)
        self.anim.setStartValue(QPointF(0, 0))

        self.anim.setKeyValueAt(0.25, QPointF(x, 0))
        self.anim.setKeyValueAt(0.5, QPointF(x, y))
        self.anim.setKeyValueAt(0.75, QPointF(0, y))

        self.anim.setEndValue(QPointF(0, 0))

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(128, 128, x-128, y-128)
        self.scene.addItem(self.ball.pixmap_item)
        self.setScene(self.scene)

        self.setWindowTitle("Calibration")
        self.setRenderHint(QPainter.Antialiasing)
        #self.resize(x, y)
        #self.setGeometry(0, 0, x, y)

        self.anim.start()
        self.showFullScreen()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

