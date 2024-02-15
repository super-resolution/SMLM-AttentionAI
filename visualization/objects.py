from OpenGL.GL import *
from OpenGL.arrays import vbo
import numpy as np

class Texture:
    def __init__(self):
        positions = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            ], dtype="f")
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)


class Surface:
    def __init__(self):
        positions = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            ],dtype="f")
        #indices = np.array([
        #    0, 1, 2,
        #    2, 1, 3
        #], dtype=np.int32)
        #Create the VBO for positions:
        self.vertex_vbo = vbo.VBO(data=positions, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)
        #Create the VBO for indices:
        #self.index_vbo = vbo.VBO(data=indices, usage=GL_STATIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)