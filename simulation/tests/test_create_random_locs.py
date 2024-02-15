import numpy as np

import unittest

from ..random_locs import create_locs

# Define test case for create_locs function
class TestCreateImages(unittest.TestCase):
    def test_create_locs(self):
        im_size = np.array([800, 600])  # Test image size
        n = 100000  # Test number of points
        locs = create_locs(im_size, n)

        # Check if the generated points are within the image size
        assert (locs >= 0).all() and (locs[:, 0] <= im_size[0]).all() and (locs[:, 1] <= im_size[1]).all(), \
            "Generated points are outside the image size"

        # Check if the number of points generated matches the expected number
        assert locs.shape[0] == n, f"Expected {n} points, but got {locs.shape[0]} points"

        print("Test passed!")


# Run the test case
if __name__ == '__main__':
    unittest.main()
