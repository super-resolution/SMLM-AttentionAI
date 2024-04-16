import unittest
from simulation.src.data_augmentation import DropoutBox
from copy import deepcopy
import torch
class TestDropoutBox(unittest.TestCase):
    def setUp(self) -> None:
        # Instantiate the DropoutBox
        self.dropout_box = DropoutBox(1, device="cpu")
        
    def test_forward(self):
        # Generate random points
        num_points = 1000
        points = torch.rand(num_points, 2)

        # Perform forward pass
        indices = self.dropout_box.forward(points)
        filtered_points = points[indices]

        # Check if the filtered points are within the box
        assert all((filtered_points[:, 0] < self.dropout_box.box["x_min"]) |
                   (filtered_points[:, 0] > self.dropout_box.box["x_max"]) |
                   (filtered_points[:, 1] > self.dropout_box.box["y_max"]) |
                   (filtered_points[:, 1] < self.dropout_box.box["y_min"])), "Some points are inside the box"

    def test_no_points_back(self):
        num_points = 1
        points = torch.rand(num_points, 2)

        # Perform forward pass
        self.dropout_box.box["x_min"] =0
        self.dropout_box.box["x_max"] =1
        self.dropout_box.box["y_min"] =0
        self.dropout_box.box["y_max"] =1

        indices = self.dropout_box.forward(points)
        assert len(indices[0])==0, f"indices has len{len(indices)} should be 0"
        filtered_points = points[indices]

    def test_box(self):
        assert 0 <= self.dropout_box.box["x_min"] <= 1, "x_min is not within [0, 1]"
        assert 0 <= self.dropout_box.box["x_max"] <= 1, "x_max is not within [0, 1]"
        assert self.dropout_box.box["x_min"] <= self.dropout_box.box["x_max"], "x_min is greater than x_max"
        assert 0 <= self.dropout_box.box["y_min"] <= 1, "y_min is not within [0, 1]"
        assert 0 <= self.dropout_box.box["y_max"] <= 1, "y_max is not within [0, 1]"
        assert self.dropout_box.box["y_min"] <= self.dropout_box.box["y_max"], "y_min is greater than y_max"

    def test_update_box(self):
        # Instantiate the DropoutBox
        prev = deepcopy(self.dropout_box.box)
        # Update the box
        self.dropout_box.update_box()

        # Check if the box is updated properly
        assert all([a!=b for a,b in zip(prev.values(), self.dropout_box.box.values())]), "did not update"
        #test if box is still valid
        self.test_box()


if __name__ == '__main__':
    unittest.main()
