import numpy as np
from PIL import Image
from skimage import feature
from skimage.filters import threshold_otsu
from skimage.morphology import dilation

class SauerLab:
    @staticmethod
    def create_binary_image(im: np.ndarray) -> np.ndarray:
        """
        Create a binary image from a given input image.

        Parameters:
            im (np.ndarray): Input image array.

        Returns:
            np.ndarray: Binary image array.
        """
        new = np.sum(im, axis=-1)
        thresh = threshold_otsu(new)
        bin_image = new < thresh
        bin_image = dilation(bin_image ^ feature.canny(new, 2))

        return bin_image

    def generate_sauerlab_pointcloud_all(self, n_approx: int = 50000) -> np.ndarray:
        """
        Generate a point cloud for SauerLab using a predefined image grid.

        Parameters:
            n_approx (int): Approximate number of points. Defaults to 50000.

        Returns:
            np.ndarray: Array containing the generated point cloud.
        """
        px_size = 100

        # Load the logo image
        im_logo = Image.open(r"resources/Logo_weiß.png")
        im_logo = np.array(im_logo)

        # Create a binary image from the logo image
        bin_logo = self.create_binary_image(im_logo)
        indices = np.array(np.where(bin_logo == 1))

        # Generate subset of points based on binary image and approximate number
        coeff = n_approx / len(indices[0])
        subset = indices.T[np.where(np.random.binomial(1, coeff, indices[0].shape[0]))]

        # Convert indices to location points and adjust for pixel size
        loc = subset.astype(np.float32) / px_size
        return np.array(loc)


if __name__ == '__main__':
    s = SauerLab()
    position_data = s.generate_sauerlab_pointcloud_all()
    np.save("../data/lab_logo2.npy", position_data)
