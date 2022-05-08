import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

def back_project_points(K, imagePts):
    imageHomogeneousPts = np.hstack((imagePts, np.ones((imagePts.shape[0], 1))))
    points3D = np.linalg.inv(K) @ imageHomogeneousPts.T
    points3D = points3D.T
    return points3D

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def print_camera_params():
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    rotation = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    content = '%d %d %d\n' % (config["camera_params"]['fx'], config["camera_params"]['k1'], config["camera_params"]['k2'])
    for i in range(3):
        r1, r2, r3 = rotation[i, 0], rotation[i, 1], rotation[i, 2]
        rot = '%d %d %d\n' % (r1, r2, r3)
        content += rot
    content += '0 0 0\n'
    return content


def read_extrinsics_params(file):
    return np.delete(np.genfromtxt(file, delimiter=','), -1, 1)


def params_to_transfomation_mtx(params):
    transformations = []
    for i in range(len(params)):
        rodrigous_rot = params[i][0:3]
        translation = params[i][3:6]

        rotation_matrix, _ = cv2.Rodrigues(rodrigous_rot)
        transformation_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        transformation_matrix[0:3, 3] = translation
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformations.append(transformation_matrix)

    return np.array(transformations)


def get_transformations(file):
    return params_to_transfomation_mtx(read_extrinsics_params(file))


def point_cloud_2_depth_map(pcd):
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    camera_params = config["camera_params"]

    points_3D = np.asarray(pcd.points)
    points_3D = points_3D[ points_3D[:,2] > 0, :].T

    transformations = get_transformations(config["params"]["EXTRINSIC_FILE"])
    min_depth, max_depth = np.min(points_3D[2, :]), np.max(points_3D[2, :])


    K = np.array([
        [config["camera_params"]['fx'], config["camera_params"]['s'],  config["camera_params"]['cx']],
        [                  0,           config["camera_params"]['fy'], config["camera_params"]['cy']],
        [                  0,                                       0,                             1],
    ])

    points_3D = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))
    image_coordinates = np.matmul(K, np.matmul(transformations[0][:3,:], points_3D))
    image_coordinates = image_coordinates / image_coordinates[2, :]

    image_coordinates = np.int0(image_coordinates)
    pixel_depth_val = 255 - ((points_3D[2, :] - min_depth) * 255 / (max_depth - min_depth))
    depth_image = np.zeros((2*camera_params['cy'], 2*camera_params['cx']))

    height_image, width_image = 2*camera_params['cy'], 2*camera_params['cx']
    height_image, width_image = int(height_image), int(width_image)

    point_in_view = 0
    for i in range(image_coordinates.shape[1]):
        if image_coordinates[0, i] >= 0 and image_coordinates[1, i] >= 0:
            if depth_image.shape[0] > image_coordinates[1, i] and depth_image.shape[1] > image_coordinates[0, i]:
                point_in_view = point_in_view + 1
                hii = height_image - image_coordinates[1, i]
                wii = width_image - image_coordinates[0, i]
                depth_image[hii, wii] = pixel_depth_val[i]

    plt.imsave(config["params"]["SPARSE_DEPTH_MAP"], depth_image, cmap='gray')
    return depth_image


if __name__=='__main__':
    pcd = o3d.io.read_point_cloud("../output/final_point_cloud.ply")
    point_cloud_2_depth_map(pcd)
