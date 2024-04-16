import os
import numpy as np
from PIL import Image

import unittest

from simulation.src.sauer_lab import SauerLab


class TestCreateLabLogo(unittest.TestCase):
    def test_create_binary_image(self):
        # Load a sample image for testing
        test_image_path = os.path.join("resources", "test_image.png")
        test_image = np.random.randint(0, 256, size=(100, 100, 3)).astype(np.uint8)
        Image.fromarray(test_image).save(test_image_path)

        # Create a binary image using the test image
        binary_image = SauerLab.create_binary_image(test_image)

        # Check if the output is a binary image
        assert np.array_equal(binary_image, binary_image.astype(bool)), "Output is not a binary image"

    def test_generate_sauerlab_pointcloud_all(self):
        sauer_lab = SauerLab()
        point_cloud = sauer_lab.generate_sauerlab_pointcloud_all()

        # Check if the output is a numpy array
        assert isinstance(point_cloud, np.ndarray), "Output is not a numpy array"


# Run the test cases
if __name__ == '__main__':
    unittest.main()