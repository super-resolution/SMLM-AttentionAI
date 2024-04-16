import unittest

import numpy as np

from simulation.src.background_structure import create_images


class TestCreateImages(unittest.TestCase):

    def test_create_images_positive(self):
        images = create_images(n_images=10, seed_val=1)
        self.assertEqual(images.shape, (10, 60, 60))
        self.assertTrue(np.all(images >= 0))
        self.assertTrue(np.all(images <= 255))

    def test_create_images_negative(self):
        with self.assertRaises(ValueError):
            create_images(n_images=-10, seed_val=1)

if __name__ == '__main__':
    unittest.main()