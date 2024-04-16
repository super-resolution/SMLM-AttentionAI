import os

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from tifffile.tifffile import imwrite, imread

from simulation.src.background_structure import create_images
from simulation.src.random_locs import create_locs
from simulation.src.simulator import Simulator
from simulation.src.sauer_lab import SauerLab
from simulation.src.structs import ApproachingLines



@hydra.main(config_name="base.yaml", config_path="../cfg/")
def my_app(cfg):
    # A batch should contain 250 images
    # Should end up in ~100 000 frames
    # Variate density and number of localisations
    # Define off time per batch
    # Keep microscope static
    #todo: restrucure
    #todo: cls from photon traces
    path = get_original_cwd() + "\\" + cfg.dataset.path + "\\" + cfg.dataset.name
    # check if bg images and coords are already defined in the target folder
    if os.path.exists(path):
        bg_images = imread(path + "/bg_images.tif")
        o_arr = np.load(path + f"/coords.npy", allow_pickle=True)
    else:
        os.mkdir(path)
        # Limited amount is fine since these undergo a distribution
        bg_images = create_images(cfg.dataset.n_batches*cfg.dataset.batch_size//4,seed_val=cfg.dataset.seed)
        imwrite(path + "/bg_images.tif", bg_images)
        #c = ApproachingLines()
        #o_arr = c.approaching_lines()
        #c = SauerLab()
        o_arr = create_locs(cfg.dataset.n_pix)
        #o_arr = c.generate_sauerlab_pointcloud_all()
        np.save(path + "/coords.npy", o_arr)

    s = Simulator(cfg.dataset.batch_size, cfg.dataset.n_pix, path, cfg.microscope, cfg.emitter.emitter_density,
                  cfg.emitter.off_time, o_arr, cfg.dataset.mode)
    s(bg_images, cfg.dataset.n_batches)


if __name__ == '__main__':
    my_app()

