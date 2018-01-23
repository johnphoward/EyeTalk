from PyQt5.QtWidgets import (QApplication, QGraphicsView,
        QGraphicsPixmapItem, QGraphicsScene, QDesktopWidget, QTextEdit)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import (QObject, QPointF, QTimer,
        QPropertyAnimation, pyqtProperty, Qt)
import sys
from GazeDetector import GazeDetector
from multiprocessing import Process, Queue


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
        self.data = []

    def initView(self):
        self.gaze = GazeDetector()
        self.showFullScreen()

        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()
        self.screen_width = ag.width()
        self.screen_height = ag.height()

        self.setWindowTitle("Calibration")
        self.setRenderHint(QPainter.Antialiasing)

        self.initPreBallMessage()


    def initPreBallMessage(self):
        self.textbox = QTextEdit(self)
        self.textbox.setReadOnly(True)
        self.textbox.move(self.screen_width / 2 - self.textbox.width(), self.screen_height / 2 - self.textbox.height())
        text = """
            <h2>Calibration GUI</h2>
            <p>Please stare at the red ball while it moves</p>
            <p>Ball will begin in the top left corner</p>
        """

        self.textbox.setHtml(text)
        self.textbox.setAlignment(Qt.AlignCenter)
        self.textbox.setFrameStyle(0)
        self.textbox.show()

        timer = QTimer(self)
        timer.timeout.connect(self.endPreBallMessage)
        timer.start(10000)
        self.timer = timer

    def endPreBallMessage(self):
        self.textbox.deleteLater()
        self.textbox.hide()
        del self.textbox

        self.timer.stop()
        self.initBallAnimation()

    def initBallAnimation(self):
        self.ball = Ball()
        self.ball_width = self.ball.pixmap_item.boundingRect().size().width()
        self.ball_height = self.ball.pixmap_item.boundingRect().size().height()

        anim = QPropertyAnimation(self.ball, b'pos')
        anim.setDuration(10000)
        anim.setStartValue(QPointF(0, 0))

        # TODO: do better at this - flush out where we want the ball to go

        anim.setKeyValueAt(0.25, QPointF(self.screen_width - self.ball_width, 0))
        anim.setKeyValueAt(0.5, QPointF(self.screen_width - self.ball_width, self.screen_height - self.ball_height))
        anim.setKeyValueAt(0.75, QPointF(0, self.screen_height - self.ball_height))

        anim.setEndValue(QPointF(0, 0))

        self.anim = anim

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, self.screen_width, self.screen_height)
        self.scene.addItem(self.ball.pixmap_item)
        self.setScene(self.scene)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample_start)
        self.timer.start(500)

        self.t2 = QTimer(self)
        self.t2.timeout.connect(self.pull_data)
        self.t2.start(50)

        self.t3 = QTimer(self)
        self.t3.timeout.connect(self.endBallAnimation)
        self.t3.start(10000)

        self.data_queue = Queue()

        self.anim.start()

    def endBallAnimation(self):
        self.scene.removeItem(self.ball.pixmap_item)
        self.timer.stop()
        self.timer = None
        self.t2.stop()
        del self.t2
        self.t3.stop()
        del self.t3
        self.ball.deleteLater()
        del self.ball

        self.initPostBallMessage()

    def initPostBallMessage(self):
        self.textbox = QTextEdit(self)
        self.textbox.setReadOnly(True)
        self.textbox.setFrameStyle(0)

        self.textbox.move(self.screen_width / 2 - self.textbox.width(), self.screen_height / 2 - self.textbox.height())

        text = """
            <h2>Data gathering complete!</h2>
            <p>The data will now be sent to our database</p>
        """

        self.textbox.setAlignment(Qt.AlignCenter)
        self.textbox.setHtml(text)
        self.textbox.show()

        self.dataSent = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.endPostBallMessage)
        self.timer.start(1000)

        self.sendData()

    def endPostBallMessage(self):
        if self.dataSent:
            app.quit()

    def sendData(self):
        self.dataSent = True

        # TODO: Implement data send to database

    def sample_start(self):
        current_pos = self.ball.pixmap_item.scenePos()
        p = Process(target=self.sample_features, args=(self.gaze, self.data_queue, current_pos))
        p.start()

    def sample_features(self, detector, queue, pos):
        x, y = int(pos.x()) + self.ball_width / 2, int(pos.y()) + self.ball_height / 2

        x_max, y_max = self.screen_width, self.screen_height
        features = detector.sample_features_mock()
        queue.put((features, [x, y, x_max, y_max]))

    def pull_data(self):
        try:
            data = self.data_queue.get(timeout=0.01)
            self.data.append(data)
        except:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

