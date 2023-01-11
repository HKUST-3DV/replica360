from sklearn.preprocessing import normalize
from genericpath import isdir
import os
import os.path as osp
import argparse
from tkinter.messagebox import NO


import numpy as np
import open3d as o3d
# import o3d.geometry
# import o3d.visualization
import json
from plyfile import PlyData
from functools import reduce
from scipy.spatial.transform import Rotation
import glob
import pyransac3d as pyrsc
import cv2

from PIL import Image, ImageOps
from data_config import replicapano_colorbox, ReplicaXRDatasetConfig

BASE_DIR = osp.dirname(osp.abspath(__file__))

IMG_WIDTH = 1024
IMG_HEIGHT = 512


def vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath):
    def get_unit_spherical_map():
        h = 512
        w = 1024
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        # do not flip horizontal
        Phi = -np.repeat(Phi, h, axis=0)

        X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
        Y = np.expand_dims(np.cos(Theta), 2)
        Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

        return unit_map

    assert osp.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert osp.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = Image.open(depth_img_filepath)
    if len(raw_depth_img.split()) == 3:
        raw_depth_img = ImageOps.grayscale(raw_depth_img)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    raw_rgb_img = Image.open(rgb_img_filepath)
    rgb_img = np.asarray(raw_rgb_img)
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0

    depth_img = np.expand_dims((depth_img/4000.0), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(
        pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud


def load_panorama_cam_pose(filepath):
    T_w_c = np.eye(4)

    quat = np.zeros(4)
    t_w_c = np.zeros(3)
    with open(filepath, 'r') as ifs:
        line = ifs.readline().strip()
        t_w_c[0] = line.split()[0]
        t_w_c[1] = line.split()[1]
        t_w_c[2] = line.split()[2]
        quat[0] = line.split()[3]
        quat[1] = line.split()[4]
        quat[2] = line.split()[5]
        quat[3] = line.split()[6]

    # print('t_w_c: ', t_w_c)
    # print('quat: ', quat)
    T_w_c[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T_w_c[:3, 3] = t_w_c
    return T_w_c


# transform bbox info from world frame into camera frame
def transform_and_save_bbox_info(
        w_obj_bbox_info_filepath, c_obj_bbox_info_filepath, T_w_c=np.eye(4), pointcloud_in_c=None):
    with open(w_obj_bbox_info_filepath, 'r') as fd:
        obj_data_in_w = json.load(fd)

    obj_data_in_c = obj_data_in_w.copy()
    obj_data_in_c['objects'] = []
    v_obj_data = obj_data_in_w['objects']
    print('len obj_data: ', len(v_obj_data))

    for obj_data in v_obj_data:
        cls_name = obj_data['name'][0:obj_data['name'].rfind('_')]
        if not (cls_name in ReplicaXRDatasetConfig().type2class):
            continue

        angle_x = float(obj_data['rotations']['x'])
        angle_y = float(obj_data['rotations']['y'])
        angle_z = float(obj_data['rotations']['z'])

        obj_data['rotations']['z'] = -angle_z
        rotation = Rotation.from_euler(
            'zyx', [angle_z, angle_y, angle_x], degrees=True).as_matrix()

        size_x = float(obj_data['dimensions']['length'])
        size_y = float(obj_data['dimensions']['width'])
        size_z = float(obj_data['dimensions']['height'])
        sizes = np.array([size_x, size_y, size_z])

        center_x = float(obj_data['centroid']['x'])
        center_y = float(obj_data['centroid']['y'])
        center_z = float(obj_data['centroid']['z'])

        new_center = np.array([center_x, center_y, center_z] - T_w_c[:3, 3])
        # flip X of bbox center
        new_center[0] *= -1
        obj_data['centroid']['x'] = float(new_center[0])
        obj_data['centroid']['y'] = float(new_center[1])
        obj_data['centroid']['z'] = float(new_center[2])

        if pointcloud_in_c is not None:
            # bbox_min_bound = new_center - sizes/2
            # bbox_max_bound = new_center + sizes/2
            # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min_bound, max_bound=bbox_max_bound)
            # bbox.color = (1, 0, 0)
            # crop_pcl = pointcloud_in_c.crop(bbox)
            # crop_bbox = crop_pcl.get_axis_aligned_bounding_box()
            # crop_bbox.color = (0, 1, 0)
            bbox = o3d.geometry.OrientedBoundingBox(
                center=new_center, R=rotation, extent=sizes)
            bbox.color = (1, 0, 0)
            crop_pcl = pointcloud_in_c.crop(bbox)
            crop_bbox = crop_pcl.get_axis_aligned_bounding_box()
            crop_bbox.color = (0, 1, 0)
            crop_pcl_sizes = (crop_bbox.max_bound - crop_bbox.min_bound)
            # print('bbox sizes: {}, crop_bbox_size: {}'.format(sizes, crop_pcl_sizes))
            # if len(crop_pcl.points) > 50 and np.all(crop_pcl_sizes/sizes >= 0.5):
            if len(crop_pcl.points) > 100:
                # o3d.visualization.draw_geometries([pointcloud_in_c, bbox, crop_bbox])
                obj_data_in_c['objects'].append(obj_data)
            else:
                continue

    obj_data_in_c['filename'] = 'bbox.ply'
    obj_data_in_c['path'] = '/media/rars13/Bill-data/Replica/pointclouds/bbox.ply'
    print('len obj_data: ', len(obj_data_in_c['objects']))

    with open(c_obj_bbox_info_filepath, 'w') as fd:
        json.dump(obj_data_in_c, fd)

    return obj_data_in_c['objects']


def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def cam3d2rad(cam3d):
    """
    Transform 3D points in camera coordinate to longitude and latitude.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    n x 2 numpy array of longitude and latitude in radiation
    first rotate left-right, then rotate up-down
    longitude: (left) -pi -- 0 --> +pi (right)
    latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
    """
    backend, atan2 = (np, np.arctan2)
    lon = atan2(cam3d[..., 0], cam3d[..., 1])
    # lat = backend.arcsin(cam3d[..., 1] / backend.linalg.norm(cam3d, axis=-1))
    lat = backend.arccos(
        cam3d[..., 2] / backend.linalg.norm(cam3d, axis=-1)) - np.pi/2
    return backend.stack([lon, lat], -1)


def camrad2pix(camrad):
    """
    Transform longitude and latitude of a point to panorama pixel coordinate.

    Parameters
    ----------
    camrad: n x 2 numpy array

    Returns
    -------
    n x 2 numpy array of xy coordinate in pixel
    x: (left) 0 --> (width - 1) (right)
    y: (up) 0 --> (height - 1) (down)
    """
    # if 'K' in self.camera:
    #     raise NotImplementedError
    # if isinstance(camrad, torch.Tensor):
    #     campix = torch.empty_like(camrad, dtype=torch.float32)
    # else:
    campix = np.empty_like(camrad, dtype=np.float32)
    width, height = IMG_WIDTH, IMG_HEIGHT
    # if isinstance(camrad, torch.Tensor):
    #     width, height = [x.view([-1] + [1] * (camrad.dim() - 2))
    #                      for x in (width, height)]
    campix[..., 0] = camrad[..., 0] * width / (2. * np.pi) + width / 2. + 0.5
    campix[..., 1] = camrad[..., 1] * height / np.pi + height / 2. + 0.5
    return campix


def cam3d2pix(cam3d):
    """
    Transform 3D points from camera coordinate to pixel coordinate.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    for 3D points: n x 2 numpy array of xy in pixel.
    x: (left) 0 --> width - 1 (right)
    y: (up) 0 --> height - 1 (down)
    """
    # if isinstance(cam3d, dict):
    #     campix = self.world2campix(self.cam3d2world(cam3d))
    # else:
    #     if 'K' in self.camera:
    #         campix = self.transform(self.camera['K'], cam3d)
    #     else:
    campix = camrad2pix(cam3d2rad(cam3d))
    return campix


def obj2frame(point, bdb3d):
    """
    Transform 3D points or Trimesh from normalized object coordinate frame to coordinate frame bdb3d is in.
    object: x-left, y-back, z-up (defined by iGibson)
    world: right-hand coordinate of iGibson (z-up)

    Parameters
    ----------
    point: n x 3 numpy array or Trimesh
    bdb3d: dict, self['objs'][id]['bdb3d']

    Returns
    -------
    n x 3 numpy array or Trimesh
    """
    # if isinstance(obj, trimesh.Trimesh):
    #     obj = obj.copy()
    #     normalized_vertices = normalize_to_unit_square(obj.vertices, keep_ratio=False)[0]
    #     obj_vertices = normalized_vertices / 2
    #     obj.vertices = IGTransform.obj2frame(obj_vertices, bdb3d)
    #     return obj
    # if isinstance(obj, torch.Tensor):
    #     size = bdb3d['size'].unsqueeze(-2)
    #     centroid = bdb3d['centroid'].unsqueeze(-2)
    #     return (bdb3d['basis'] @ (obj * size).transpose(-1, -2)).transpose(-1, -2) + centroid
    # else:
    rotation = Rotation.from_euler(
        'zyx', [bdb3d['rotations']['z'], bdb3d['rotations']['y'], bdb3d['rotations']['x']], degrees=True).as_matrix()
    centroid = np.array(
        [-bdb3d['centroid']['x'], bdb3d['centroid']['y'], bdb3d['centroid']['z']])
    sizes = np.array([bdb3d['dimensions']['length'],
                      bdb3d['dimensions']['width'], bdb3d['dimensions']['height']])
    return (rotation @ (point * sizes).T).T + centroid


def bdb3d_corners(bdb3d: (dict, np.ndarray)):
    """
    Get ordered corners of given 3D bounding box dict or disordered corners

    Parameters
    ----------
    bdb3d: 3D bounding box dict

    Returns
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    if isinstance(bdb3d, np.ndarray):
        centroid = np.mean(bdb3d, axis=0)
        z = bdb3d[:, -1]
        surfaces = []
        for surface in (bdb3d[z < centroid[-1]], bdb3d[z >= centroid[-1]]):
            surface_2d = surface[:, :2]
            center_2d = centroid[:2]
            vecters = surface_2d - center_2d
            angles = np.arctan2(vecters[:, 0], vecters[:, 1])
            orders = np.argsort(-angles)
            surfaces.append(surface[orders][(0, 1, 3, 2), :])
        corners = np.concatenate(surfaces)
    else:
        corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
                                axis=1, bitorder='little', count=-5).astype(np.float32)
        corners = corners - 0.5
        # if isinstance(bdb3d['size'], torch.Tensor):
        #     corners = torch.from_numpy(corners).to(bdb3d['size'].device)
        corners = obj2frame(corners, bdb3d)
    return corners


def wrapped_line(image, p1, p2, colour, thickness, lineType=cv2.LINE_AA):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0]+image.shape[1], p1[1]])
    p2b = np.array([p2[0]-image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness, lineType=lineType)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness, lineType=lineType)
        cv2.line(image, tuple(p1b), p2, colour, thickness, lineType=lineType)


# visualize 3dbbox on panorama
def vis_objs3d(image, v_bbox3d, b_show_axes=False, b_show_centroid=False, b_show_bbox3d=True, b_show_info=False, thickness=2):

    def draw_line3d(image, p1, p2, color, thickness, quality=30, frame='world'):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        if frame == 'world':
            print('input points must be in camera frame')
        elif frame != 'cam3d':
            raise NotImplementedError
        points = interpolate_line(p1, p2, quality)
        normal_points = normalize(points)
        pix = np.round(cam3d2pix(normal_points)).astype(np.int32)
        for t in range(quality - 1):
            p1, p2 = pix[t], pix[t + 1]
            wrapped_line(image, tuple(p1), tuple(p2), color,
                         thickness, lineType=cv2.LINE_AA)

    def draw_objaxes(image, centroid, sizes, rotation, thickness=2):

        for axis in np.eye(3, dtype=np.float32):
            endpoint = rotation @ ((axis / 2) * sizes) + centroid
            color = axis * 255
            draw_line3d(image, centroid, endpoint,
                        color, thickness, frame='cam3d')

    def draw_centroid(image, centroid, color, thickness=2):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        normal_centroid = centroid/np.linalg.norm(centroid)
        center = cam3d2pix(normal_centroid)
        cv2.circle(image, tuple(center.astype(np.int32).tolist()),
                   5, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_bdb3d(image, bdb3d, color, thickness=2):
        corners = bdb3d_corners(bdb3d)
        corners_box = corners.reshape(2, 2, 2, 3)
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    draw_line3d(
                        image, corners_box[idx1], corners_box[idx2], color, thickness=thickness, frame='cam3d')
        for idx1, idx2 in [(0, 5), (1, 4)]:
            draw_line3d(image, corners[idx1], corners[idx2],
                        color, thickness=thickness, frame='cam3d')

    def draw_objinfo(image, bdb3d_centeroid, obj_cls_name, color):
        color = [255 - c for c in color]
        normal_centroid = bdb3d_centeroid/np.linalg.norm(bdb3d_centeroid)
        bdb3d_pix = cam3d2pix(normal_centroid)
        bottom_left = bdb3d_pix.astype(np.int32)
        bottom_left[1] -= 16
        cv2.putText(image, obj_cls_name, tuple(bottom_left.tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    image = image.copy()
    dis = [np.linalg.norm([-o['centroid']['x'], o['centroid']
                           ['y'], o['centroid']['z']]) for o in v_bbox3d]
    i_objs = sorted(range(len(dis)), key=lambda k: dis[k])
    for i_obj in reversed(i_objs):
        bdb3d = v_bbox3d[i_obj]
        obj_label = bdb3d['name'][0:bdb3d['name'].rfind('_')]

        obj_cls_id = ReplicaXRDatasetConfig().type2class[obj_label]
        color = (replicapano_colorbox[obj_cls_id]
                 * 255).astype(np.uint8).tolist()
        centroid = np.array(
            [-bdb3d['centroid']['x'], bdb3d['centroid']['y'], bdb3d['centroid']['z']])
        sizes = np.array([bdb3d['dimensions']['length'],
                          bdb3d['dimensions']['width'], bdb3d['dimensions']['height']])
        bdb3d['rotations']['z'] *= -1
        rotation = Rotation.from_euler(
            'zyx', [bdb3d['rotations']['z'], bdb3d['rotations']['y'], bdb3d['rotations']['x']], degrees=True).as_matrix()

        if b_show_axes:
            draw_objaxes(image, centroid, sizes, rotation, thickness=thickness)
        if b_show_centroid:
            draw_centroid(image, centroid,
                          color, thickness=thickness)
        if b_show_bbox3d:
            draw_bdb3d(image, bdb3d, color, thickness=thickness)
        if b_show_info:
            draw_objinfo(image, centroid, obj_label, color)
    return image


def main(raw_dataset_path, render_dataset_path):

    house_folders = [f for f in os.listdir(
        render_dataset_path) if osp.isdir(osp.join(render_dataset_path, f))]

    for folder in house_folders:
        # if 'frl_apartment_0_000' != folder:
        #     continue

        print(
            f' ------------------------ preprocessing house {folder} ------------------------ ')

        scene_name = folder[0:folder.rfind('_')]
        house_name = folder
        # raw 3d object bbox file
        obj_bbox_filepath = osp.join(
            raw_dataset_path, scene_name, scene_name+'_aligned.json')
        print('raw_3dbbox_file: ', obj_bbox_filepath)

        v_spot_folders = [spot for spot in os.listdir(osp.join(
            render_dataset_path, house_name)) if osp.isdir(osp.join(render_dataset_path, house_name, spot))]

        assert len(v_spot_folders), 'Empty house folder: {}'.format(house_name)

        for spot_name in v_spot_folders:
            print(f' ------ preprocessing spot {spot_name} ------ ')

            spot_rgb_img_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'rgb.png')
            spot_depth_img_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'depth.png')
            spot_pose_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'pose.txt')
            saved_obj_bbox_info_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'bbox.json')
            saved_color_pcl_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'bbox.ply')
            saved_bbox_vis_img_filepath = osp.join(
                render_dataset_path, house_name, spot_name, 'bbox_vis.png')

            T_w_c = load_panorama_cam_pose(spot_pose_filepath)

            # save color pointcloud in camera frame
            pcl_in_c = vis_color_pointcloud(
                spot_rgb_img_filepath, spot_depth_img_filepath, saved_color_pcl_filepath)
            # transform 3dbbox of world frame into camera frame
            bbox_in_c = transform_and_save_bbox_info(
                obj_bbox_filepath, saved_obj_bbox_info_filepath, T_w_c, pcl_in_c)

            # visualize bbox on panorama
            rgb_img = cv2.imread(spot_rgb_img_filepath, -1)
            vis_img = vis_objs3d(rgb_img, bbox_in_c,
                                 b_show_axes=False, b_show_centroid=True, b_show_bbox3d=True, b_show_info=True)
            cv2.imwrite(saved_bbox_vis_img_filepath, vis_img)


if __name__ == '__main__':
    # post process replica-pano dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_folderpath", type=str,
                        default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1")
    parser.add_argument("--rendered_dataset_folderpath", type=str,
                        default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext/debug_20230105")

    args = parser.parse_args()
    raw_dataset_path = args.raw_dataset_folderpath
    render_dataset_path = args.rendered_dataset_folderpath

    main(raw_dataset_path, render_dataset_path)
