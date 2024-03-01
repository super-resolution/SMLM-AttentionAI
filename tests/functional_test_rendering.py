import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from matplotlib import cm

from utility.emitters import Emitter
from visualization.gui.user_interface import Ui_MainWindow
from visualization.visualization_open_gl import Points
from visualization.widget import CustomnWidget


class FunctionalTestRendering():
    def __init__(self):
        #todo: use emitter_set
        self.emitters = Emitter.load("tmp.npy")

    def plot_emitters(self):
        app = pg.mkQApp()
        widget = CustomnWidget()
        widget.show()
        #todo: add drag events
        #todo: disable fov turning overwrite event
        cmap = cm.get_cmap('hot')
        cmap = np.array([cmap(i/1000) for i in range(0,1000)])
        #widget.opts["center"] = QtGui.QVector3D(self.emitters.xyz[1].max() / 2, self.emitters.xyz[0].max() / 2, 0)
        p = Points(self.emitters.xyz, self.emitters.sigxsigy, self.emitters.frames, cmap)
        p.background_render()
        widget.addItem(p)
        #
        # data = np.array(p.image)[:,:,0].astype(np.uint8)
        # plt.imshow(data)
        # plt.show()


        # thread = AlphaThread(image, roi=ROI)
        # size = (-1, ROI[1] - ROI[0], -1, ROI[3] - ROI[2])
        # x, bars = WidgetFactory.gridget(size)
        # x.show()
        # x.resize(1000, 700)

        # item = render.PlotBars()
        #thread.sig.connect(lambda state, state2: (bars.setData(state, state2), x.update()))
        # x.addItem(item)
        # item.show()
        # cmap = cm.get_cmap('jet')
        # cmap = cm.get_cmap('jet')
        # rgba = [cmap((k - image.min()) / (image.max() - image.min())) for k in image.flatten()]

        # item.set_data(image.shape[0], image.flatten(), rgba)
        #thread.start()

        app.exec()

    def plot_with_ui(self):
        qtApp = QApplication(sys.argv)
        main_window = QMainWindow()
        widget = Ui_MainWindow()
        widget.setupUi(main_window)
        openGLWidget = CustomnWidget(main_window)
        #add widget to main window
        widget.openGLWidget = openGLWidget
        widget.openGLWidget.setObjectName("openGLWidget")
        widget.verticalLayout.addWidget(widget.openGLWidget)

        widget.verticalLayoutWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        #set slider values
        widget.horizontalSlider.setMaximum(int(self.emitters.sigxsigy.max()*100))
        widget.horizontalSlider.setValue(int(self.emitters.sigxsigy.max()*100))

        widget.horizontalSlider_2.setMaximum(int(self.emitters.sigxsigy.max()*100))

        widget.horizontalSlider_3.setMaximum(self.emitters.frames.max())
        widget.horizontalSlider_4.setMaximum(self.emitters.frames.max())
        widget.horizontalSlider_4.setValue(self.emitters.frames.max())

        cmap = cm.get_cmap('hot')
        cmap = np.array([cmap(i/1000) for i in range(0,1000)])

        main_window.show()
        p = Points(self.emitters.xyz, self.emitters.sigxsigy, self.emitters.frames, self.emitters.p, cmap)

        #write update function and connect signals
        widget.horizontalSlider.valueChanged.connect(lambda v: p.update_uniform("precision_filter",v,1))
        widget.horizontalSlider.valueChanged.connect(lambda v: widget.l_prec_max.setText(f"Precision filter max: {v}" ))

        widget.horizontalSlider_2.valueChanged.connect(lambda v: p.update_uniform("precision_filter",v,0))
        widget.horizontalSlider_2.valueChanged.connect(lambda v: widget.l_prec_min.setText(f"Precision filter min: {v}" ))


        widget.horizontalSlider_3.valueChanged.connect(lambda v: p.update_uniform("frame_filter",v,0))
        widget.horizontalSlider_3.valueChanged.connect(lambda v: widget.l_frame_min.setText(f"Frame filter min: {v}" ))

        widget.horizontalSlider_4.valueChanged.connect(lambda v: p.update_uniform("frame_filter",v,1))
        widget.horizontalSlider_4.valueChanged.connect(lambda v: widget.l_frame_max.setText(f"Frame filter max: {v}" ))
        #widget.opts["center"] = QtGui.QVector3D(self.emitters.xyz[1].max() / 2, self.emitters.xyz[0].max() / 2, 0)

        widget.horizontalSlider_5.valueChanged.connect(lambda v: p.update_uniform("probability_filter",v/100,0))
        widget.horizontalSlider_5.valueChanged.connect(lambda v: widget.l_prob_min.setText(f"Probability filter min: {v}" ))

        widget.horizontalSlider_6.valueChanged.connect(lambda v: p.update_uniform("probability_filter",v/100,1))
        widget.horizontalSlider_6.valueChanged.connect(lambda v: widget.l_prob_max.setText(f"Probabilty filter max: {v}" ))

        p.background_render()
        #p.set_texture()
        widget.openGLWidget.addItem(p)
        sys.exit(qtApp.exec())


if __name__ == '__main__':
    t = FunctionalTestRendering()
    t.plot_with_ui()