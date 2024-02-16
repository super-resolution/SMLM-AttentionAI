import numpy as np
from OpenGL.GL import *


class Create1DTexture:
    """
    Create 1d texture on GPU for OpenGL rendering
    """
    def __init__(self):
        """
        Initialize the basics like texture handle and an indicator whether data was successfully set
        """
        self.textureHandle = glGenTextures(1)
        self.dataSet = False

    def set_texture(self, data:np.ndarray[np.float32], interpolation:int):
        """
        Put data into the texture buffer
        :param data: n data points
        :param interpolation: Numerator for interpolation mode either linear or nearest
        """
        glEnable(GL_TEXTURE_1D)
        glTexImage1D(GL_PROXY_TEXTURE_1D, 0, GL_RGBA, data.shape[0], 0, GL_RGBA, GL_FLOAT, None)
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_1D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception("OpenGL failed to create 1D texture (%dx%d); too large for this hardware." % data.shape[:2])

        glBindTexture(GL_TEXTURE_1D, self.textureHandle)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, interpolation)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, interpolation)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, data.shape[0], 0, GL_RGBA, GL_FLOAT, data.astype(np.float))
        glBindTexture(GL_TEXTURE_1D,0)
        self.dataSet = True


class Create2DTexture:
    """
    Create 2D texture on GPU for OpenGL rendering
    """
    def __init__(self):
        """
        Initialize the basics like texture handle and an indicator whether data was successfully set
        """
        self.textureHandle = glGenTextures(1)
        self.dataSet = False

    def set_texture(self, data:np.ndarray[float], interpolation:int):
        """
        Put data into the texture buffer
        :param data: nxm data points
        :param interpolation: Numerator for interpolation mode either linear or nearest
        """
        glEnable(GL_TEXTURE_2D)
        glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, data.shape[1], data.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, None)
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % data.shape[:2])

        glBindTexture(GL_TEXTURE_2D, self.textureHandle)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, interpolation)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, interpolation)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.shape[1],
                     data.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, data.astype(np.uint8))
        glBindTexture(GL_TEXTURE_2D,0)
        self.dataSet = True