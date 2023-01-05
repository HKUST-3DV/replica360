import os
import os.path as osp
import argparse


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
import math

BASE_DIR = osp.dirname(osp.abspath(__file__))


class ReplicaXRDatasetConfig(object):
    def __init__(self):
        self.num_class = 9

        self.type2class = {'basket': 0, 'bed': 1, 'cabinet': 2, 'chair': 3, 'sofa': 4, 'table': 5,
                           'door': 6, 'window': 7, 'bookshelf': 8, 'picture': 9,
                           'counter': 10, 'blinds': 11, 'desk': 12,
                           'shelves': 13, 'curtain': 14, 'dresser': 15, 'pillow': 16, 'mirror': 17,
                           'floor_mat': 18, 'clothes': 19, 'books': 20, 'fridge': 21, 'tv': 22,
                           'paper': 23, 'towel': 24, 'shower_curtain': 25, 'box': 26,
                           'whiteboard': 27, 'person': 28, 'nightstand': 29, 'toilet': 30,
                           'sink': 31, 'lamp': 32, 'bathtub': 33, 'bag': 34, 'stool': 35, 'rug': 36}


def load_scene_gravity(scene_semantic_file):
    if not osp.exists(scene_semantic_file):
        print(f'File {scene_semantic_file} doesnt exist!!!')
        exit(-1)

    gravity_center = np.zeros((3,), dtype=np.float32)
    gravity_direction = np.zeros((3,), dtype=np.float32)
    with open(scene_semantic_file, 'r') as sd:
        sem_data = json.load(sd)
        grav_center = sem_data['gravityCenter']
        grav_dir = sem_data['gravityDirection']

    gravity_center[0] = grav_center['x']
    gravity_center[1] = grav_center['y']
    gravity_center[2] = grav_center['z']
    print(f'gravityCenter: {gravity_center}')

    # rotate gravity direction to upward
    gravity_direction[0] = grav_dir['x']
    gravity_direction[1] = grav_dir['y']
    gravity_direction[2] = grav_dir['z']
    print(f'gravityDirection: {gravity_direction}')

    return gravity_center, gravity_direction


def translate_mesh(vertices, trans):
    for vertice in vertices:
        vertice[0] += -trans[0]
        vertice[1] += -trans[1]
        vertice[2] += -trans[2]


def rotate_mesh(vertices, rot_matrix):
    for vertice in vertices:
        v = np.array([vertice[0], vertice[1], vertice[2]])
        rot_v = rot_matrix @ v
        vertice[0] = rot_v[0]
        vertice[1] = rot_v[1]
        vertice[2] = rot_v[2]

def get_axis_aligned_mesh(mesh_vertices, saved_ply_filepath=None, z_rotation_res=0.1, b_vis=False):

    def vis_and_save_pointcloud(pcd, b_vis=True, save_path=None):
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        rbb = pcd.get_oriented_bounding_box()
        rbb.color = (0,1,0)
        if b_vis:
            o3d.visualization.draw_geometries([pcd, aabb, rbb],
                                        zoom=0.7,
                                        front=[0.5439, -0.2333, -0.8060],
                                        lookat=[2.4615, 2.1331, 1.338],
                                        up=[-0.1781, -0.9708, 0.1608])
        if save_path is not None:
            o3d.io.write_point_cloud(save_path,pcd)

    pcd = o3d.geometry.PointCloud()
    origin_points = np.asarray([[vert[0], vert[1], vert[2]] for vert in mesh_vertices])
    pcd.points = o3d.utility.Vector3dVector(origin_points)
    plane1 = pyrsc.Plane()
    points_floor = origin_points[np.where(origin_points[:, 2] < 0.5)]
    # vis_and_save_pointcloud(points_floor, b_vis)

    best_eq, best_inliers = plane1.fit(points_floor, 0.01)
    inject_normal = np.array(best_eq[:3])
    obj_normal = np.array([0,0,-1])
    theta = math.acos(np.dot(inject_normal, obj_normal))
    if theta > np.pi/2:
        theta = np.pi - theta
        inject_normal = -inject_normal
    print('floorplane_deta: {}'.format(theta))
    plane_normal = np.cross(inject_normal, obj_normal)
    plane_normal = plane_normal/np.linalg.norm(plane_normal)
    plane_normal = theta * plane_normal
    R_floorplane = Rotation.from_rotvec(plane_normal).as_matrix()
    points = np.transpose(np.dot(R_floorplane, np.transpose(pcd.points)))
    pcd_update = o3d.geometry.PointCloud()
    pcd_update.points = o3d.utility.Vector3dVector(points)
    min_sum = 10000
    min_idx = 0

    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    resolution = z_rotation_res
    steps_num = int(180/resolution)
    for i in range(steps_num):
        pcd_update.transform(rotz(resolution/180*np.pi))
        aabb = pcd_update.get_axis_aligned_bounding_box()
        # print('aabb: ', aabb)
        current_sum = math.sqrt((aabb.max_bound[0]-aabb.min_bound[0])**2 + (aabb.max_bound[1]-aabb.min_bound[1])**2)
        if(min_sum>(current_sum)):
            min_idx = i
            min_sum = current_sum
        # print('min_sum: ', min_sum)
    
    R_right = np.eye(4)
    R_right[:3,:3] = R_floorplane
    
    phi = resolution / 180 * np.pi * min_idx
    if phi > np.pi/2:
        phi = phi - np.pi
    print('xyplane_delta: ', phi)
    final_matrix = np.dot(rotz(phi), R_right)
    pcd.transform(final_matrix)
    vis_and_save_pointcloud(pcd, b_vis=b_vis, save_path=saved_ply_filepath)

    return final_matrix

def transform_and_save_mesh(input_filepath, save_filepath, trans_vector, rot_vec=None, b_axis_aligned_mesh=True):
    mesh = PlyData.read(input_filepath)
    v_vertices = mesh.elements[0]
    print(f'vertices: {v_vertices}')
    v_faces = mesh.elements[1]
    print(f'faces: {v_faces}')

    translate_mesh(v_vertices, trans_vector)
    if rot_vec is not None:
        R_grav_pcl = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_vec)
        rotate_mesh(v_vertices, R_grav_pcl)
    if b_axis_aligned_mesh:
        axis_aligned_mesh_filepath = osp.join(osp.dirname(input_filepath), 'axis_aligned_'+osp.basename(input_filepath))
        T = get_axis_aligned_mesh(v_vertices, saved_ply_filepath=axis_aligned_mesh_filepath)
        R_axis_align = T[:3,:3]
        rotate_mesh(v_vertices, R_axis_align)

    PlyData([v_vertices, v_faces], text=False).write(save_filepath)

    if b_axis_aligned_mesh:
        return R_axis_align
    else:
        None


def transform_and_save_camposition(cam_position_filepath, saved_cam_position_filepath, gravity_center):
    v_cam_position = []
    with open(cam_position_filepath, 'r') as ifs:
        lines_str = ifs.read().splitlines()
        for line in lines_str:
            if len(line) == 0:
                continue
            cam_pos_data = [float(d) for d in line.split()]
            cam_pos_data[0] -= float(gravity_center[0])
            cam_pos_data[1] -= float(gravity_center[1])
            cam_pos_data[2] -= float(gravity_center[2])
            v_cam_position.append(cam_pos_data)

    assert len(v_cam_position)
    with open(saved_cam_position_filepath, 'w') as ofs:
        for cam_pos in v_cam_position:
            pos_str = reduce(lambda a, b: str(a) + " " + str(b), cam_pos)
            ofs.write(pos_str)
            ofs.write('\n')

    return True


def transform_and_save_cam_trajectory(cam_trajectory_filepath, saved_cam_trajectory_filepath, gravity_center, rot_vec=None):
    v_cam_position = []
    rot_x_90 = Rotation.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
    rot_x_90n = np.transpose(rot_x_90)

    traj_file_name = osp.basename(cam_trajectory_filepath)
    large_apart_0_traj_files = ['large_apartment_0_trajectory0.json', 'large_apartment_0_trajectory1.json',
                                'large_apartment_0_trajectory2.json',
                                'large_apartment_0_trajectory3.json']
    with open(cam_trajectory_filepath, 'r') as ifs:
        full_data = json.load(ifs)
        keyframes = full_data['keyframes']

        for kfm_data in keyframes:
            if len(kfm_data) == 0 or (not 'userTransforms' in kfm_data):
                continue
            # print(kfm_data['userTransforms'])
            kfm_transforms = kfm_data['userTransforms']
            assert(len(kfm_transforms) == 2)

            if kfm_transforms[0]['name'] == 'sensor':
                sensor_kfm_t = kfm_transforms[0]
                agent_kfm_t = kfm_transforms[1]
            elif kfm_transforms[0]['name'] == 'agent':
                sensor_kfm_t = kfm_transforms[1]
                agent_kfm_t = kfm_transforms[0]

            # habitat-frame
            #     /|\ Y
            #      |
            #      |
            #      |______________\ X
            #     /               /
            #    /
            #   /
            # |/_ Z
            # camera position in habitat frame
            pos = sensor_kfm_t['transform']['translation']
            # rotate gravity_vector into habitat frame
            rot_g = rot_x_90n @ gravity_center
            # caemra pos in translated habitat frame
            pos_translated = pos - rot_g
            if rot_vec is not None:
                print('R_aa:\n', rot_vec)
                pos_translated = np.linalg.inv(rot_vec) @ pos_translated

            # world-frame
            #     /|\ Z
            #      |    _ Y
            #      |    /|
            #      |   /
            #      |  /
            #      | /
            #      |/___________\ X
            #                   /
            # rotate to world frame
            pos_w = rot_x_90n @ pos_translated

            # camera-frame
            #     /|\ Z
            #      |    _ Y
            #      |    /|
            #      |   /
            #      |  /
            #      | /
            #      |/___________\ X
            #                   /
            # rotate to world frame
            # transform to camera frame
            pos_c = -rot_x_90n @ pos_w
            # fix pos_y == 1.6m
            if traj_file_name in large_apart_0_traj_files:
                pos_c[1]=1.6 + 2.75
            else:
                pos_c[1] = 1.6
            v_cam_position.append([pos_c[0], pos_c[1], pos_c[2]])

    spots_num = len(v_cam_position)
    assert spots_num > 100

    spots_interval = spots_num // 100
    v_cam_position_sampled = v_cam_position[0:spots_num:spots_interval]
    with open(saved_cam_trajectory_filepath, 'w') as ofs:
        for cam_pos in v_cam_position_sampled:
            pos_str = reduce(lambda a, b: str(a) + " " + str(b), cam_pos)
            ofs.write(pos_str)
            ofs.write('\n')

    return True


def transform_and_save_infosemanticjson(object_semantic_filepath, saved_object_semantic_filepath, gravity_center):
    total_sem_data = None
    v_obj_bbox = None
    with open(object_semantic_filepath, 'r') as ifs:
        total_sem_data = json.load(ifs)
        v_obj_bbox = total_sem_data['objects']
        print(f' contain object num: {len(v_obj_bbox)}')

    for obj_bbox in v_obj_bbox:
        obj_cls_name = obj_bbox['class_name']
        obj_id = obj_bbox['id']
        # if obj_cls_name in ReplicaXRDatasetConfig().type2class:
        obb_center = obj_bbox['oriented_bbox']['abb']['center']
        obb_R = Rotation.from_quat(obj_bbox['oriented_bbox']
                            ['orientation']['rotation']).as_matrix()
        obb_t = np.array(obj_bbox['oriented_bbox']
                         ['orientation']['translation'])
        obb_T = np.eye(4)
        obb_T[:3, :3] = obb_R
        obb_T[:3, 3] = obb_t
        obb_T_inv = np.linalg.inv(obb_T)
        tmp = obb_center.copy()
        tmp = tmp - (obb_T_inv[:3, :3] @ gravity_center + obb_T_inv[:3, 3])
        obb_center[0] = float(tmp[0])
        obb_center[1] = float(tmp[1])
        obb_center[2] = float(tmp[2])
        # print('object {}_{} center: {}'.format(obj_cls_name, obj_id, obj_bbox['oriented_bbox']['abb']['center']))

    with open(saved_object_semantic_filepath, 'w') as ofs:
        json.dump(total_sem_data, ofs)

    return True


def transform_and_save_bbox_prior(object_semantic_filepath, saved_3dbbox_prior_filepath, gravity_center, scene_name=None):
    if scene_name is not None:
        pointcloud_filename = scene_name+'.ply'
    else:
        pointcloud_filename = 'room_0.ply'

    v_ret_dict = []

    total_sem_data = None
    v_obj_bbox = None
    with open(object_semantic_filepath, 'r') as ifs:
        total_sem_data = json.load(ifs)
        v_obj_bbox = total_sem_data['objects']

    for obj_bbox in v_obj_bbox:
        obj_cls_name = obj_bbox['class_name']
        obj_id = obj_bbox['id']

        # if not (obj_cls_name in ReplicaXRDatasetConfig().type2class):
        #     continue

        obb_center = obj_bbox['oriented_bbox']['abb']['center']
        obb_size = obj_bbox['oriented_bbox']['abb']['sizes']
        obb_R = Rotation.from_quat(obj_bbox['oriented_bbox']
                            ['orientation']['rotation']).as_matrix()
        obb_t = np.array(obj_bbox['oriented_bbox']
                         ['orientation']['translation'])
        obb_T = np.eye(4)
        obb_T[:3, :3] = obb_R
        obb_T[:3, 3] = obb_t
        angles = Rotation.from_quat(obj_bbox['oriented_bbox']['orientation']['rotation']).as_euler(
            'zyx', degrees=True)

        tmp_bbox_center = obb_center.copy()
        # world frame
        tmp = (obb_T[:3, :3] @ tmp_bbox_center + obb_T[:3, 3])
        # translate bbox center in world
        tmp = tmp - gravity_center

        ret_dict = {}
        ret_dict['name'] = obj_cls_name + "_" + str(obj_id)
        ret_dict['centroid'] = {}
        ret_dict['centroid']['x'] = float(tmp[0])
        ret_dict['centroid']['y'] = float(tmp[1])
        ret_dict['centroid']['z'] = float(tmp[2])
        ret_dict['dimensions'] = {}
        ret_dict['dimensions']['length'] = float(obb_size[0])
        ret_dict['dimensions']['width'] = float(obb_size[1])
        ret_dict['dimensions']['height'] = float(obb_size[2])
        ret_dict['rotations'] = {}
        ret_dict['rotations']['x'] = float(angles[2])
        ret_dict['rotations']['y'] = float(angles[1])
        ret_dict['rotations']['z'] = float(angles[0])
        v_ret_dict.append(ret_dict)

    root_node = {}
    root_node['folder'] = 'pointclouds'
    root_node['filename'] = pointcloud_filename
    root_node['path'] = 'pointclouds\\'+pointcloud_filename
    root_node['objects'] = v_ret_dict

    bbox_annotation_filepath = saved_3dbbox_prior_filepath
    with open(bbox_annotation_filepath, 'w') as fd:
        json.dump(root_node, fd)


def main(dataset_path):

    scene_folders = [f for f in os.listdir(
        dataset_path) if osp.isdir(osp.join(dataset_path, f))]

    b_skip_mesh = True
    b_skip_semantic_mesh = True
    b_skip_info_semantic_json = True
    b_skip_3dbbox_prior = True
    b_skip_cam_6dof_file = False
    for folder in scene_folders:
        if folder != 'hotel_0':
            continue

        print(
            f' ------------------------ preprocessing scene {folder} ------------------------ ')

        scene_name = folder
        scene_mesh_filepath = osp.join(dataset_path, folder, 'mesh.ply')
        semantic_mesh_filepath = osp.join(
            dataset_path, folder, 'habitat/mesh_semantic.ply')
        scene_semantic_filepath = osp.join(
            dataset_path, folder, 'semantic.json')
        object_semantic_filepath = osp.join(
            dataset_path, folder, 'habitat/info_semantic.json')
        saved_mesh_filepath = osp.join(dataset_path, folder, scene_name+'_aligned.ply')
        saved_semantic_mesh_filepath = osp.join(
            dataset_path, folder, 'habitat/rotated_mesh_semantic.ply')
        cam_position_filepath = glob.glob(osp.join(dataset_path, folder, scene_name + '_trajectory*.json'))
        saved_cam_position_filepath = [traj_fp[:-5]+'.txt' for traj_fp in cam_position_filepath]
        saved_object_semantic_filepath = osp.join(
            dataset_path, folder, 'habitat/rotated_info_semantic.json')
        saved_obj_bbox_prior_filepath = osp.join(
            dataset_path, folder, scene_name+'.json')
        saved_axis_align_mesh_T_filepath = osp.join(dataset_path, folder, 'axis_aligned_transform.npy')

        assert osp.exists(
            scene_mesh_filepath), f"{scene_mesh_filepath} doesnt exist..."
        assert osp.exists(
            semantic_mesh_filepath), f"{semantic_mesh_filepath} doesnt exist..."
        assert osp.exists(
            scene_semantic_filepath), f"{scene_semantic_filepath} doesnt exist..."
        assert len(cam_position_filepath)
        assert osp.exists(
            object_semantic_filepath), f"{object_semantic_filepath} doesnt exist..."

        # read gravity center and direction
        gravity_center, gravity_direction = load_scene_gravity(
            scene_semantic_filepath)

        if osp.exists(saved_axis_align_mesh_T_filepath):
            R_aa = np.load(saved_axis_align_mesh_T_filepath)
        else:
            R_aa = None

        if scene_name == 'large_apartment_0':
            gravity_center += np.array([0, 0, -1.07])

        # translate the origin of mesh
        if not b_skip_mesh:
            R_axis_align = transform_and_save_mesh(
                    scene_mesh_filepath, saved_mesh_filepath, gravity_center, rot_vec=None)
            if R_axis_align is not None:
                np.save(saved_axis_align_mesh_T_filepath, R_axis_align)

        if not b_skip_semantic_mesh:
            transform_and_save_mesh(
                semantic_mesh_filepath, saved_semantic_mesh_filepath, gravity_center)

        # translate the camera trajectory
        if not b_skip_cam_6dof_file:
            # transform_and_save_camposition(
            #     cam_position_filepath, saved_cam_position_filepath, gravity_center)
            for idx in range(len(cam_position_filepath)):

                transform_and_save_cam_trajectory(
                    cam_position_filepath[idx], saved_cam_position_filepath[idx], gravity_center, rot_vec=R_aa)

        # translate object
        if not b_skip_info_semantic_json:
            transform_and_save_infosemanticjson(
                object_semantic_filepath, saved_object_semantic_filepath, gravity_center)

        # parse and save objects' 3dbbox
        if not b_skip_3dbbox_prior:
            transform_and_save_bbox_prior(
                object_semantic_filepath, saved_obj_bbox_prior_filepath, gravity_center, scene_name)


if __name__ == '__main__':
    # translate the raw mesh to the gravity center and align with gravity
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folderpath", type=str,
                        default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_mesh", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_semantic_mesh", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_info_semantic_json", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_cam_position", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")

    args = parser.parse_args()
    dataset_path = args.dataset_folderpath

    main(dataset_path)
