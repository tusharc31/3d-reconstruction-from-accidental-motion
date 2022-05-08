import cv2
import numpy as np
from utilities import *
from itertools import compress
import matplotlib.pyplot as plt

class ObjectFlow:
    
    def __init__(self, images, config):
        camera_params = config["camera_params"]
        self.images = images[1:]
        self.reference_image = images[0]
        self.reference_features_world_points = None
        self.reference_features_textures = None
        self.feature_params = config["feature_params"]
        self.lk_params = config["lk_params"]
        self.K = np.array([
            [camera_params['fx'],  camera_params['s'], camera_params['cx']],
            [                  0, camera_params['fy'], camera_params['cy']],
            [                  0,                   0,                  1],
        ])
        self.features_klt = []
        self.reference_features = cv2.goodFeaturesToTrack(gray(self.reference_image), mask = None, **self.feature_params)
        for reference_feture in self.reference_features:
            self.features_klt.append([(reference_feture.ravel()[0], reference_feture.ravel()[1])])
        

    def homography_filter(self, threshold = 0.95):
        image_pts = np.zeros((len(self.features_klt[0]), len(self.features_klt), 2))
        mask = np.zeros((len(self.features_klt), 1))

        total_images = len(self.features_klt[0])
        total_feat_points = len(self.features_klt)

        for i in range(total_feat_points):
            for j in range(total_images):
                for fm in [0, 1]:
                    image_pts[j, i, fm] = self.features_klt[i][j][fm]

        reference_image_pts = image_pts[0, :, :]    
        
        for j in range(total_images - 1):
                _, inliers = cv2.findHomography(image_pts[j + 1, :, :], reference_image_pts, cv2.RANSAC, 5.0)
                mask += inliers
        
        mask = (mask >= threshold * total_images)
        reference_image_pts = reference_image_pts[mask[:, 0], :]
        self.features_klt = compress(self.features_klt, mask)
        self.features_klt = list(self.features_klt)
        self.reference_features = np.reshape(reference_image_pts, (reference_image_pts.shape[0], 1, 2))
        
        return self.features_klt


    def get_matches(self):

        features_klt = self.features_klt
        reference_features = self.reference_features
        
        for view in self.images:
            new_feat = []
            gray_reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            gray_view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            current_features, valid, _ = cv2.calcOpticalFlowPyrLK(gray_reference_image, gray_view, reference_features, None, **self.lk_params)
            mask = valid == 1
            current_features = current_features[mask]
            reference_features = reference_features[mask]
            shape1 = reference_features.shape[0]

            for i, j in enumerate(valid):
                 if j == 1:
                     new_feat.append(features_klt[i])
            features_klt = new_feat

            for i, feature in enumerate(current_features):
                u = feature.ravel()[0]
                v = feature.ravel()[1]
                features_klt[i].append((u, v))


            reference_features = reference_features.reshape((shape1, 1, 2))

        self.features_klt = features_klt
        self.reference_features = reference_features

        return features_klt
        
    


    def vis_feature_points(self):
        feat_shape = (-1,1,2)
        line_color = (255, 0, 0)
        mask, image = np.zeros_like(self.reference_image), self.reference_image.copy()
        for feature in self.features_klt:
            feature = np.array(feature, np.int32)
            feature = feature.reshape(feat_shape)
            cv2.polylines(mask, [feature], 0, line_color)
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        plt.imsave(config["params"]["OPTICAL_FLOW_PLOT"], cv2.add(image, mask))


    def init_feature_points(self):
        shape1 = self.reference_features.shape[0]
        reference_features = self.reference_features.reshape(self.reference_features.shape[0], 2).astype('uint32')
        ref_n, ref_m = reference_features[:,1], reference_features[:,0]
        self.reference_features_textures = (self.reference_image[ref_n, ref_m, :] ).astype('float64')
        M = np.ones((shape1, 1))
        X = np.hstack((reference_features, M))
        X = X.T
        X = X / X[2, :]
        invdepthVector = 1/np.random.uniform(2, 4, (X.shape[1]))
        
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        cam_params = config["camera_params"]
        h, w = 2*cam_params['cy'], 2*cam_params['cx']
        X[0, :], X[1, :] = -X[0,:]+w/2, -X[1,:] + h/2 
        X[2,:] = X[2,:] * config["camera_params"]['fx'] * invdepthVector

        self.reference_features_world_points = X.T


    def BA_inputs(self, bundle_file_path, bundle_weights=None):
        f = open(bundle_file_path, 'w')
        params = '%d %d\n' % (len(self.features_klt[0]), len(self.features_klt))
        f.write(params)

        total_images, total_feat_points = len(self.features_klt[0]), len(self.features_klt)
        
        params = print_camera_params()
        file_params = ''
        for _ in range(total_images): file_params += params
        f.write(file_params)

        file_params = ''

        with open("config.yaml", 'r') as f_new:
            config = yaml.safe_load(f_new)

        for pt in range(total_feat_points):
            
            color, point = self.reference_features_textures[pt, :], self.reference_features_world_points[pt, :]
            x, y, z = point[0], point[1], point[2]
            r, g, b = color[0], color[1], color[2]
            params = '%f %f %f\n %d %d %d\n' % (x, y, z, r, g, b)
            
            for img in range(total_images):
                c1 = config["camera_params"]['cx'] - self.features_klt[pt][img][0]
                c2 = config["camera_params"]['cy'] - self.features_klt[pt][img][1]
                paramsLine = '%d %d %d %d ' % (img, pt*total_images + img, c1, c2)
                params += paramsLine

            params += '\n'
            file_params += params

        f.write(file_params)
        f.close()  
    
