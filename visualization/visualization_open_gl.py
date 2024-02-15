import os
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from visualization.shader import Shader
from PyQt5 import QtCore,QtGui
import math
from visualization.buffers import Renderbuffer,Texturebuffer,Framebuffer
from visualization.objects import Texture,Surface
from visualization.textures import Create1DTexture,Create2DTexture
from PIL import Image

MAX_VBO_SIZE = 50000
class Points(GLGraphicsItem):
    def __init__(self, positions, sig, frames, probability, cmap):
        super().__init__()
        basepath = os.path.dirname(os.path.realpath(__file__)) +r"\shader_files"
        self.filename = basepath + r"\STORM2"
        self.modelview = []
        self.projection = []
        self.point_filters = {"precision_filter":np.array([.0,200.], dtype=np.float),
                              "frame_filter": np.array([0,10**9], dtype=np.int32),
                              "probability_filter":np.array([0.,2.], dtype=np.float)}

        self._shader = Shader(self.filename)
        self.image_shader = Shader(basepath + r"\Image")
        self.enums = [GL_POINT_SPRITE, GL_PROGRAM_POINT_SIZE, GL_BLEND]
        self.position = positions
        #get precision in nm
        sig *= 100
        #self.cluster = np.array((0.0,0.0))
        #todo: this should be buffered
        self.size = 20.0#
        #self.ratio = 1
        #self.fg_color = [0.0,0.0,0.0,0.0]
        self.color = [1.0,1.0,1.0,1.0]
        self.maxEmission = 0
        self.updateData = True
        #todo: enums for filtering
        self.args = ["position", "size", "color", "maxEmission", "cluster"]
        #todo: add cmap array as enum?
        #todo: add sigx and sigy
        #uniform texture
        self.cmapTexture = Create1DTexture()
        self.cmapTexture.set_texture(cmap, GL_NEAREST)
        #todo: chunk stuff into buffers with ~80 000 points
        #todo: multiple draw calls
        self.sizes = [MAX_VBO_SIZE]*(self.position.shape[0]//MAX_VBO_SIZE)+[self.position.shape[0]%MAX_VBO_SIZE]
        self.n_buffers = (self.position.shape[0]//MAX_VBO_SIZE)+1
        self.sig_buffer = [vbo.VBO(sig[MAX_VBO_SIZE*i:MAX_VBO_SIZE*(i+1)].astype("f"), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER') for i in range(self.n_buffers)]
        self.xyz_buffer = [vbo.VBO(positions[MAX_VBO_SIZE*i:MAX_VBO_SIZE*(i+1)].astype("f"), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER') for i in range(self.n_buffers)]
        self.frames_buffer = [vbo.VBO(frames[MAX_VBO_SIZE*i:MAX_VBO_SIZE*(i+1)].astype("f"), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER') for i in range(self.n_buffers)]
        self.prob_buffer = [vbo.VBO(probability[MAX_VBO_SIZE*i:MAX_VBO_SIZE*(i+1)].astype("f"), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER') for i in range(self.n_buffers)]

        self.imageTexture = Create2DTexture()
        self.Quad = Texture()
        self.Surface = Surface()
        # self._m_clusterarray_buffer = vbo.VBO(self.cluster.astype("f"), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER')
        #self.set_data(**kwds)

    def update_uniform(self,k, v, l_h):
        self.point_filters[k][l_h] = v
        self.background_render()
        self.update()

    def background_render(self, precision=999.):
        self.width = int(self.position[:,0].max()//10)#int(896/322*1449)
        self.height = int(self.position[:,1].max()//10)
        #self.roi = rect
        X = self.position[:,1].max()
        Y = self.position[:,0].max()
        #compute X center
        Xt = X/2#X/2
        Yt = Y/2#Y/2-5000

        glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." )

        dist = math.tan(math.radians(60))*Y/2# Field of view in Y direction
        aspect = float(self.width)/float(self.height)#aspect ratio of display
        center = QtGui.QVector3D(Xt, Yt, 0)
        eye = QtGui.QVector3D(Xt, Yt, dist)#Point of eye in space
        up = QtGui.QVector3D(0, 1, 0)

        modelview = QtGui.QMatrix4x4()
        modelview.lookAt(eye,center,up)

        projection = QtGui.QMatrix4x4()
        projection.perspective(60.0, aspect, dist*0.0001, dist*10000.0)

        renderbuffer = Renderbuffer()
        renderbuffer.build(self.width, self.height)
        self.texturebuffer = Texturebuffer()
        self.texturebuffer.build(self.width, self.height)
        self.framebuffer = Framebuffer()
        self.framebuffer.build(renderbuffer.handle, self.texturebuffer.handle)
        try:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer.handle)
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._shader.__setitem__("u_modelview", modelview)
            self._shader.__setitem__("u_projection", projection)
            for k,v in self.point_filters.items():
                #v is already a np array in the right datatype
                self._shader.__setitem__(k, v)
            self.setupGLState()
            # Enable several enumerators
            for enum in self.enums:
                glEnable(enum)
            # Additative blending for render
            glBlendFunc(GL_ONE, GL_ONE)
            self.draw_points()
        finally:
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            buf = glReadPixels(0,0,self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
            self.image = Image.frombytes(mode="RGBA", size=(self.width, self.height), data=buf)
            #self.image = image.transpose(Image.FLIP_LEFT_RIGHT)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            renderbuffer.delete()
            #self.texturebuffer.delete()
            self.framebuffer.delete()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_texture(self):
        self.imageTexture.set_texture(np.array(self.image)[:,:,1], GL_LINEAR)

    def draw_points(self):
        with self._shader:
            #todo: draw all buffers
            # Bind buffer objects
            for i in range(self.n_buffers):
                glEnableVertexAttribArray(1)
                self.xyz_buffer[i].bind()
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
                #bind size buffer with sigma
                glEnableVertexAttribArray(2)
                self.sig_buffer[i].bind()
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
                #bin frames
                glEnableVertexAttribArray(3)
                self.frames_buffer[i].bind()
                glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, None)
                #bind prob buffer
                glEnableVertexAttribArray(4)
                self.prob_buffer[i].bind()
                glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, None)
                try:
                    # Draw everything
                    #todo: put in correct sizes
                    glDrawArrays(GL_POINTS, 0, self.sizes[i])
                finally:
                    # Clean up
                    glDisableVertexAttribArray(1)
                    self.xyz_buffer[i].unbind()
                    glDisableVertexAttribArray(2)
                    self.sig_buffer[i].unbind()
                    glDisableVertexAttribArray(3)
                    self.frames_buffer[i].unbind()
                    glDisableVertexAttribArray(4)
                    self.prob_buffer[i].unbind()
            # Disable enumerators
            for enum in self.enums:
                glDisable(enum)
                #unbind cmap


    def paint(self):
        #self.background_render()
        self.modelview = self.view().viewMatrix()*self.viewTransform()
        self.projection = self.view().projectionMatrix()
        self.image_shader.__setitem__("u_modelview", self.modelview)
        self.image_shader.__setitem__("u_projection", self.projection)
        self.setupGLState()
        with self.image_shader:
            #glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer.handle)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texturebuffer.handle)
            glActiveTexture(GL_TEXTURE0+1)
            glBindTexture(GL_TEXTURE_1D, self.cmapTexture.textureHandle)
            self.Surface.vertex_vbo.bind()
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            self.Quad.vertex_vbo.bind()
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
            try:
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            finally:
                # Clean up
                glDisableVertexAttribArray(1)
                glDisableVertexAttribArray(2)
                self.Surface.vertex_vbo.unbind()
                self.Quad.vertex_vbo.unbind()
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_1D, 0)
                glActiveTexture(GL_TEXTURE0+1)
                glBindTexture(GL_TEXTURE_1D, 0)


    def paint_points(self):
        self.modelview = self.view().viewMatrix()*self.viewTransform()
        self.projection = self.view().projectionMatrix()
        self._shader.__setitem__("u_modelview", self.modelview)
        self._shader.__setitem__("u_projection", self.projection)
        self.setupGLState()
        # Enable several enumerators
        for enum in self.enums:
            glEnable(enum)
        # Additative blending for render
        glBlendFunc(GL_ONE, GL_ONE)
        self.draw_points()




