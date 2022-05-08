import os
import csv
import cv2
import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class plane_sweep:

    def __init__(self,cfg):
        
        self.cfg            = cfg
        self.fx             = cfg["camera_params"]["fx"]
        self.fy             = cfg["camera_params"]["fy"]
        self.cx             = cfg["camera_params"]["cx"]
        self.cy             = cfg["camera_params"]["cy"]
        self.extrinsic_file = cfg["params"]["EXTRINSIC_FILE"]
        self.folder         = cfg["params"]["IMAGE_DIR"] 
        self.K = np.array([
            [self.fx,  0, self.cx],
            [ 0, self.fy, self.cy],
            [ 0, 0,     1],
        ])


    def create_homography(self,K, C1, R1, C2, R2, depth):
        k_inv = np.linalg.inv(K)
        rays_mat = np.matmul(R1.T,k_inv)
        w_2_cref = np.matmul(K,R2)
        final_mat = np.matmul(w_2_cref,rays_mat)
        H  = depth * final_mat
        H[:,2] += K @ R2 @ (C1 - C2)
        return H

    def avg_scores(self,scores, valid_ratio = 0.5):
        valid_score_cnt = scores.shape[0] * valid_ratio
        valid_score_cnt = int(valid_score_cnt)
        index_sc = np.argpartition(scores, valid_score_cnt, axis=0)
        sorted_score = np.take_along_axis(scores, index_sc[:valid_score_cnt,:], axis=0)
        total_score = np.sum(sorted_score, axis=0)
        total_score = total_score / valid_score_cnt
        return total_score


    def score_confidence_calculate(self,unary_cost_array):
        first_min = 0
        second_min = 0
        confidence = 0
        for i in range(unary_cost_array.shape[1]):
            for j in range(unary_cost_array.shape[2]):
                first_min, second_min = np.partition(unary_cost_array[:, i, j], 1)[0:2]
                score = unary_cost_array[:, i, j]
                confidence = (1 + second_min) / (1 + first_min)
                unary_cost_array[:, i, j] = confidence * score
        return unary_cost_array


    def create_patches(self,window_width):
        ref_img = self.ref_img
        ref_img_patches = as_strided(ref_img, shape=(ref_img.shape[0] - window_width*2,
                                        ref_img.shape[1] - window_width*2, 1 + window_width*2, 1 + window_width*2),
                                        strides=ref_img.strides + ref_img.strides, writeable=False)
        h, w, _, _      = ref_img_patches.shape
        window_size     = 1 + window_width*2
        ref_img_patches = ref_img_patches.reshape((ref_img_patches.shape[0]*ref_img_patches.shape[1], np.power(window_size, 2)))

        warp_patches    = np.zeros((len(self.warped_images), ref_img_patches.shape[0], ref_img_patches.shape[1]))
        warped_images = self.warped_images

        for i in range(len(warped_images)):
            int_warp_patch = as_strided(warped_images[i], shape=(warped_images[i].shape[0] - window_width*2,
                                warped_images[i].shape[1] - window_width*2, 1 + window_width*2, 1 + window_width*2),
                                strides=warped_images[i].strides + warped_images[i].strides, writeable=0)
            int_m = np.power(window_size, 2)
            int_n = int_warp_patch.shape[0]*int_warp_patch.shape[1]
            int_warp_patch = int_warp_patch.reshape((int_n, int_m))
            warp_patches[i,:,:] = int_warp_patch

        return(warp_patches, ref_img_patches,h,w)

    def depth_sweep_plane(self,folder, outfile, depth_samples, scale, window_width):

        C, R = [], []
        
        with open(self.extrinsic_file) as ext_file:
            csv_reader = csv.reader(ext_file, delimiter=',')

            for row in csv_reader:
                project_mat = []
                for r in row[:-1]:
                    project_mat.append(float(r))

                rot_mat, _ = cv2.Rodrigues(np.array(project_mat[:3]))
                centre_mat = np.matmul(-1*np.linalg.inv(rot_mat), np.array(project_mat[3:6]))

                C.append(centre_mat)
                R.append(rot_mat)

        files =  sorted(os.listdir(self.folder))[:len(R)]
        self.all_img = []

        for file in files :
            im = cv2.imread(os.path.join(self.folder, file))
            self.all_img.append(im)


        self.scaled_gray_images = []
        for img in self.all_img :
            gray_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
            for i in range(scale):
                gray_img = cv2.pyrDown(gray_img)
            self.scaled_gray_images.append(gray_img)

        self.ref_img = self.scaled_gray_images[0]
        height, width = self.ref_img.shape

        self.unary_cost_array = np.zeros((depth_samples.shape[0], height, width))
        num_images = len(self.all_img)
        unary_cost_array = np.zeros((depth_samples.shape[0], height, width))

        for idx, depth in enumerate(tqdm(depth_samples)):
            shape1 = (num_images, 3, 3)
            homographies = np.zeros(shape1)
            self.warped_images = []

            for index in range(num_images) :
                h = self.create_homography(self.K, C[0], R[0], C[index], R[index], depth)
                orig_scale = np.power(2, scale)
                h[:,:2] = h[:,:2] * orig_scale
                h[2,:] = h[2,:] * orig_scale
                homographies[index,:,:] = h

            for i in range(1, num_images):
                warp_criteria = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                warp = cv2.warpPerspective(self.scaled_gray_images[i], homographies[i], self.ref_img.shape[::-1], warp_criteria)
                self.warped_images.append(warp)

            (warp_patches,ref_img_patches,h_,w_) = self.create_patches(window_width)

            reprojc_loss = np.sum(np.abs(warp_patches - ref_img_patches), axis=2)
            score = self.avg_scores(reprojc_loss, valid_ratio = 0.5)

            hww = height-window_width
            www = width-window_width


            unary_cost_array[idx, window_width:hww, window_width:www] = score.reshape((h_, w_))

            unary_cost_array[idx, 0: window_width, :] = unary_cost_array[idx, window_width, :]
            unary_cost_array[idx, 1+hww:, :] = unary_cost_array[idx, hww, :]
            shape1 = unary_cost_array[idx, :, window_width].shape[0]
            unary_cost_array[idx, :, 0: window_width] = unary_cost_array[idx, :, window_width].reshape((shape1,1))
            unary_cost_array[idx, :, 1+www:] = unary_cost_array[idx, :, www].reshape((unary_cost_array[idx, :, www].shape[0], 1))

        unary_cost_array = self.score_confidence_calculate(unary_cost_array)

        return unary_cost_array.astype('float32')



class crf_model:

    def __init__(self,cfg):
    
        self.cfg = cfg
        self.folder        = cfg["params"]["IMAGE_DIR"]
        self.output_folder = cfg["params"]["OUTPUT_FOLDER"]
        self.num_samples   = cfg["ps_params"]["num_samples"]
        self.scale         = cfg["crf_params"]["scale"]
        self.max_depth     = cfg["ps_params"]["max_depth"]
        self.min_depth     = cfg["ps_params"]["min_depth"]
        self.patch_radius  = cfg["ps_params"]["patch_radius"]
        
        self.fx           = cfg["camera_params"]["fx"]
        self.fy           = cfg["camera_params"]["fy"]
        self.cx           = cfg["camera_params"]["cx"]
        self.cy           = cfg["camera_params"]["cy"]
        
        self.iters        = cfg["crf_params"]["iters"]
        self.weight       = cfg["crf_params"]["weight"]
        self.pos_std      = cfg["crf_params"]["pos_std"]
        self.rgb_std      = cfg["crf_params"]["rgb_std"]
        self.max_penalty  = cfg["crf_params"]["max_penalty"]
    
        self.plane_sweep_obj = plane_sweep(cfg)
        self.create_depth_samples()
        self.compute_unary_photo_loss()
        self.solve_crf()


    def create_depth_samples(self):
    
        self.depth_vals = np.zeros(self.num_samples)
        step_size = 1.0 / (self.num_samples - 1.0)
        for iter_ in range(self.num_samples):
            self.depth_vals[iter_] = (self.fx/(self.min_depth +  (iter_ * step_size)))
        
        ref_file_name = sorted(os.listdir(self.folder))[0]
        ref_img = cv2.imread(self.folder+"/"+ref_file_name)
        
        for _ in range(self.scale):
            ref_img = cv2.pyrDown(ref_img)
            
        ref_img = cv2.pyrMeanShiftFiltering(ref_img, 20, 20, 1)
        self.ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2Lab)
        
        
    def compute_unary_photo_loss(self):
    
        unary_score = self.plane_sweep_obj.depth_sweep_plane(self.folder,self.output_folder,self.depth_vals,self.scale,self.patch_radius)

        max_depth = np.max(self.depth_vals)
        min_depth = np.min(self.depth_vals)
        unary_depth = np.zeros((unary_score.shape[1], unary_score.shape[2]))
        depth_index = np.argmin(unary_score, axis=0)
        unary_depth = ((self.depth_vals[depth_index] - min_depth) * 255.0)/(max_depth - min_depth)
        file_name = self.output_folder + "/wta.png"
        cv2.imwrite(file_name, unary_depth)
        self.unary_score  = np.where(np.sum(unary_score ,axis=0)<=1e-9, 0 , unary_score /(np.sum(unary_score ,axis=0)))
        
        
        
    def solve_crf(self):
        number_labels = self.unary_score.shape[0]
        self.unary_score = unary_from_softmax(self.unary_score)
        graph = dcrf.DenseCRF2D(self.ref_img.shape[1], self.ref_img.shape[0],number_labels )
        
        graph.setUnaryEnergy(self.unary_score)
        pos_std_ = tuple(float(x) for x in self.pos_std.split(','))
        rgb_std_ = tuple(float(x) for x in self.rgb_std.split(','))
        weight_ = float(self.weight)
        max_penalty_ = float(self.max_penalty)


        graph.addPairwiseBilateral(sxy=pos_std_, srgb=rgb_std_, rgbim=self.ref_img, compat=np.array([weight_, number_labels*max_penalty_]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        depth_labels = graph.inference(self.iters)
        
        depth_ = np.argmax(depth_labels, axis=0).reshape((self.ref_img.shape[:2]))
        depth_map = self.depth_vals[depth_]
        # import pdb
        # pdb.set_trace()
        min_depth_val = np.min(depth_map)
        max_depth_val = np.max(depth_map)
        depth_map = ((depth_map - min_depth_val)/(max_depth_val - min_depth_val)) * 255.0

        fine_name = self.output_folder + "/depth_map.png"
        cv2.imwrite(fine_name, depth_map)
