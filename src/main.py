import cv2
import yaml
import os
import argparse
from crf import *
import subprocess
import open3d as o3d
from glob import glob
from utilities import * 
import matplotlib.pyplot as plt
from object_flow import ObjectFlow


class BundleAdjuster:
    
    def __init__(self, config):

        self.bundle_file = config["params"]["BUNDLE_FILE"]
        self.nonmonotonic_steps = config["CERES_PARAMS"]['nonmonotonic_steps']
        self.output_ply = config["params"]["FINAL_POINT_CLOUD"]
        self.inner_iterations = config["CERES_PARAMS"]['inner_iterations']
        self.max_iterations = config["CERES_PARAMS"]['maxIterations']
        self.input_ply = config["params"]["INITIAL_POINT_CLOUD"]
        self.solver = config["CERES_PARAMS"]['solver']

    def bundle_adjust(self):
        input, output = self.input_ply, self.output_ply
        maxitr, inneritr = self.max_iterations, self.inner_iterations
        subprocess.call([
            self.solver,
            '--num_iterations={}'.format(maxitr),
            '--input={}'.format(self.bundle_file),
            '--inner_iterations={}'.format(inneritr),
            '--initial_ply={}'.format(input),
            '--nonmonotonic_steps={}'.format(self.nonmonotonic_steps),
            '--final_ply={}'.format(output),
        ])


if __name__ == '__main__':
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    image_dir = config["params"]["IMAGE_DIR"]
    output_folder = config["params"]["OUTPUT_FOLDER"]
    if not (os.path.isdir(output_folder)):
        os.mkdir(output_folder)


    image_files = sorted(glob(image_dir+'/*'))

    images = []
    for image in image_files:
        images.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))

    klt_tracker = ObjectFlow(images, config)
    features_klt = klt_tracker.get_matches()
    features_klt = klt_tracker.homography_filter()
    klt_tracker.vis_feature_points()

    inital_point_cloud_path = config["params"]["INITIAL_POINT_CLOUD"]
    finall_point_cloud_path = config["params"]["FINAL_POINT_CLOUD"]
    klt_tracker.init_feature_points()
    klt_tracker.BA_inputs('../output/bundle.out')
    
    bundle_adjuster = BundleAdjuster(config)
    bundle_adjuster.bundle_adjust()
    pcd = o3d.io.read_point_cloud(finall_point_cloud_path)
    depth_map = point_cloud_2_depth_map(pcd)
    dense_crf  = crf_model(config)
