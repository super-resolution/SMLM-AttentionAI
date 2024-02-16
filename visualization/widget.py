import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore,QtGui,QtOpenGL

class CustomnWidget(gl.GLViewWidget):
    """
    Extend pygraph GLViewWidget by some customn events
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.opts["elevation"] = 90
        self.opts["azimuth"] = 0
        #self.opts["distance"] = 4000

    def mouseMoveEvent(self, ev):
        """Drag event converts mouse position to world space. Z axis is set to zero."""
        diff = self.cliptoworld(self.viewMatrix(), self.projectionMatrix(),ev.pos(), self.size().width(), self.size().height()) - self.cliptoworld(self.viewMatrix(), self.projectionMatrix(),self.mousePos, self.size().width(), self.size().height())
        self.mousePos = ev.pos()
        if ev.buttons() == QtCore.Qt.RightButton:
            # self.viewMatrix.translate(diff.x(),diff.y(),0)
            self.opts["center"] -= diff.toVector3D(  )  # pg.QtGui.QVector3D(diff.x(),diff.y(),0)
            self.update()
    @staticmethod
    def cliptoworld(view, projection, position, width, height):
        """Creates a QVector4D in world position with incoming QPoint.
        Read http://trac.bookofhook.com/bookofhook/trac.cgi/wiki/MousePicking for detailed explanation"""
        #Get camera Position in real space.
        camera_position_clip = QtGui.QVector4D(0,0,0,1)
        #Screen coordinates on orthogonal view in pixel (0,0)= top left.
        #Z and W value for point out of camera plane.
        z = 1.0
        w = 1.0
        #Normalize screen coordinates. (0,0) is the center. Right upper corner is (1 ,1).
        mouse_pos = QtGui.QVector2D(position)
        scale = QtGui.QVector2D(2.0/width, -2.0/height)
        transform = QtGui.QVector2D(-1,1)
        #Wrap point in QVector4D 4th component is either 0 for vector or 1 for point.
        p_clip = QtGui.QVector4D((mouse_pos*scale+transform), z, w)


        #Invert matrices.
        iprojection = projection.inverted()
        iview = view.inverted()

        #Check invertible and transform point from camera to world space.
        if iprojection[1] and iview[1] == True:
            point_view = iprojection[0]*p_clip
            camera_pos_view = iprojection[0]*camera_position_clip
            #Normalize.
            point_view_normalized = point_view/point_view.w()
            camera_pos_norm = camera_pos_view/camera_pos_view.w()

            p_world = iview[0]*point_view_normalized
            camera_pos = iview[0]*camera_pos_norm
        else:
            raise ValueError("Error in matrix inversion")
        #Get intersection with x axis.
        point_world = camera_pos-camera_pos.z()/(
            camera_pos.z()-p_world.z())*(camera_pos-p_world)

        return point_world