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



@hydra.main(config_name="simulation_images.yaml", config_path="cfg/")
def my_app(cfg):
    # A batch should contain 250 images
    # Should end up in ~100 000 frames
    # Variate density_old and number of localisations
    # Define off time per batch
    # Keep microscope static

    #get original working directory from hydra
    cwd = get_original_cwd()
    path_list = [cwd, cfg.dataset.path, cfg.dataset.name]
    data_folder = os.path.join(*path_list[:-1])
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    path = os.path.join(*path_list)
    bg_images_path = os.path.join(path, "bg_images.tif")
    coords_path = os.path.join(path, "coords.npy")
    photon_trace_path = os.path.join(*path_list[:-1],"emitter_traces", cfg.emitter.photon_trace_file)
    # check if bg images and coords are already defined in the target folder
    if os.path.exists(path):
        bg_images = imread(bg_images_path)
        o_arr = np.load(coords_path, allow_pickle=True)
    else:
        #Create dataset if it does not exist
        os.mkdir(path)
        # Limited amount is fine since these undergo a distribution
        bg_images = create_images(cfg.dataset.n_batches*cfg.dataset.batch_size//4,seed_val=cfg.dataset.seed)
        imwrite(bg_images_path, bg_images)
        #Lines
        #c = ApproachingLines()
        #o_arr = c.approaching_lines()
        #Logo
        #c = SauerLab()
        #Points
        o_arr = create_locs(cfg.dataset.n_pix)
        #o_arr = c.generate_sauerlab_pointcloud_all()
        np.save(coords_path, o_arr)

    s = Simulator(cfg.dataset.batch_size, cfg.dataset.n_pix, path, cfg.microscope, cfg.emitter.emitter_density,
                  cfg.emitter.off_time, o_arr, cfg.dataset.mode, photon_trace_path)
    s(bg_images, cfg.dataset.n_batches)


if __name__ == '__main__':
    my_app()

