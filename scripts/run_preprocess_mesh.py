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
from scipy.spatial.transform import Rotation as R

BASE_DIR = osp.dirname(osp.abspath(__file__))


class ReplicaXRDatasetConfig(object):
    def __init__(self):
        self.num_class = 9

        self.type2class={'bed':1, 'cabinet':2, 'chair':3, 'sofa':4, 'table':5,
                        'door':6, 'window':7, 'bookshelf':8, 'picture':9,
                        'counter':10, 'blinds':11, 'desk':12,
                        'shelves':13, 'curtain':14, 'dresser':15, 'pillow':16, 'mirror':17,
                        'floor_mat':18, 'clothes':19, 'books':20, 'fridge':21, 'tv':22,
                        'paper':23, 'towel':24, 'shower_curtain':25, 'box':26,
                        'whiteboard':27, 'person':28, 'nightstand':29, 'toilet':30,
                        'sink':31, 'lamp':32, 'bathtub':33, 'bag':34}

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
        v=np.array([vertice[0], vertice[1], vertice[2]])
        rot_v = rot_matrix @ v
        vertice[0] = rot_v[0]
        vertice[1] = rot_v[1]
        vertice[2] = rot_v[2]

def transform_and_save_mesh(input_filepath, save_filepath, trans_vector, R_matrix=None):
    mesh = PlyData.read(input_filepath)
    v_vertices = mesh.elements[0]
    print(f'vertices: {v_vertices}')
    v_faces = mesh.elements[1]
    print(f'faces: {v_faces}')

    translate_mesh(v_vertices, trans_vector)
    # R_grav_pcl = o3d.geometry.get_rotation_matrix_from_axis_angle(gravity_direction)
    # rotate_mesh(v_vertices, R_grav_pcl)
    PlyData([v_vertices, v_faces], text=False).write(save_filepath)

    return True

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
            pos_str = reduce(lambda a, b : str(a)+ " " +str(b), cam_pos)
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
        obb_R = R.from_quat(obj_bbox['oriented_bbox']['orientation']['rotation']).as_matrix()
        obb_t = np.array(obj_bbox['oriented_bbox']['orientation']['translation'])
        obb_T = np.eye(4)
        obb_T[:3, :3] = obb_R
        obb_T[:3, 3] = obb_t
        obb_T_inv = np.linalg.inv(obb_T)
        if obj_id == 22:
            print(f'blinds_22 Transform: \n {obb_T}')
        tmp = obb_center.copy()
        tmp = tmp - (obb_T_inv[:3,:3] @ gravity_center + obb_T_inv[:3,3])
        obb_center[0] = float(tmp[0])
        obb_center[1] = float(tmp[1])
        obb_center[2] = float(tmp[2])
        # print('object {}_{} center: {}'.format(obj_cls_name, obj_id, obj_bbox['oriented_bbox']['abb']['center']))

    with open(saved_object_semantic_filepath, 'w') as ofs:
        json.dump(total_sem_data, ofs)

    return True


def transform_and_save_bbox_prior(object_semantic_filepath, saved_3dbbox_prior_filepath, gravity_center):
    v_ret_dict = []

    total_sem_data = None
    v_obj_bbox = None
    with open(object_semantic_filepath, 'r') as ifs:
        total_sem_data = json.load(ifs)
        v_obj_bbox = total_sem_data['objects']

    for obj_bbox in v_obj_bbox:
        obj_cls_name = obj_bbox['class_name']
        obj_id = obj_bbox['id']

        if not (obj_cls_name in ReplicaXRDatasetConfig().type2class):
            continue

        obb_center = obj_bbox['oriented_bbox']['abb']['center']
        obb_size = obj_bbox['oriented_bbox']['abb']['sizes']
        obb_R = R.from_quat(obj_bbox['oriented_bbox']['orientation']['rotation']).as_matrix()
        obb_t = np.array(obj_bbox['oriented_bbox']['orientation']['translation'])
        obb_T = np.eye(4)
        obb_T[:3, :3] = obb_R
        obb_T[:3, 3] = obb_t
        angles = R.from_quat(obj_bbox['oriented_bbox']['orientation']['rotation']).as_euler('zyx', degrees=True)

        tmp_bbox_center = obb_center.copy()
        # world frame
        tmp = (obb_T[:3,:3] @ tmp_bbox_center + obb_T[:3,3])
        # translate bbox center in world
        tmp = tmp - gravity_center

        ret_dict = {}
        ret_dict['name'] = obj_cls_name +"_"+ str(obj_id)
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
    root_node['filename'] = 'room_0.ply'
    root_node['path'] = "/media/rars13/Bill-data/Replica/pointclouds/room_0.ply"
    root_node['objects'] = v_ret_dict

    bbox_annotation_filepath = saved_3dbbox_prior_filepath
    with open(bbox_annotation_filepath, 'w') as fd:
        json.dump(root_node, fd)

def main(dataset_path):

    scene_folders = [f for f in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, f))]

    b_skip_mesh = True
    b_skip_semantic_mesh = True
    b_skip_info_semantic_json = False
    b_skip_cam_6dof_file = True
    for folder in scene_folders:
        if folder != 'room_0':
            continue

        print(f' ------------------------ preprocessing scene {folder} ------------------------ ')

        scene_name = folder
        scene_mesh_filepath = osp.join(dataset_path, folder, 'mesh.ply')
        semantic_mesh_filepath = osp.join(dataset_path, folder, 'habitat/mesh_semantic.ply')
        scene_semantic_filepath = osp.join(dataset_path, folder, 'semantic.json')
        object_semantic_filepath = osp.join(dataset_path, folder, 'habitat/info_semantic.json')
        saved_mesh_filepath = osp.join(dataset_path, folder, 'rotated_mesh.ply')
        saved_semantic_mesh_filepath = osp.join(dataset_path, folder, 'habitat/rotated_mesh_semantic.ply')
        cam_position_filepath = osp.join(dataset_path, folder, scene_name + '_6dof.txt')
        saved_cam_position_filepath = cam_position_filepath
        saved_object_semantic_filepath = osp.join(dataset_path, folder, 'habitat/rotated_info_semantic.json')
        saved_obj_bbox_prior_filepath = osp.join(dataset_path, folder, 'habitat/3dbbox_prior.json')


        assert osp.exists(scene_mesh_filepath), f"{scene_mesh_filepath} doesnt exist..."
        assert osp.exists(semantic_mesh_filepath), f"{semantic_mesh_filepath} doesnt exist..."
        assert osp.exists(scene_semantic_filepath), f"{scene_semantic_filepath} doesnt exist..."
        assert osp.exists(cam_position_filepath), f"{cam_position_filepath} doesnt exist..."
        assert osp.exists(object_semantic_filepath), f"{object_semantic_filepath} doesnt exist..."

        # read gravity center and direction
        gravity_center, gravity_direction = load_scene_gravity(scene_semantic_filepath)

        # translate the origin of mesh
        if not b_skip_mesh:
            transform_and_save_mesh(scene_mesh_filepath, saved_mesh_filepath, gravity_center)
        if not b_skip_semantic_mesh:
            transform_and_save_mesh(semantic_mesh_filepath, saved_semantic_mesh_filepath, gravity_center)

        # translate the camera trajectory
        if not b_skip_cam_6dof_file:
            transform_and_save_camposition(cam_position_filepath, saved_cam_position_filepath, gravity_center)

        # translate object
        if not b_skip_info_semantic_json:
            transform_and_save_infosemanticjson(object_semantic_filepath, saved_object_semantic_filepath, gravity_center)
            transform_and_save_bbox_prior(object_semantic_filepath, saved_obj_bbox_prior_filepath, gravity_center)



if __name__ == '__main__':
    # translate the raw mesh to the gravity center and align with gravity
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folderpath", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_mesh", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_semantic_mesh", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_info_semantic_json", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    # parser.add_argument("--skip_cam_position", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")


    args = parser.parse_args()
    dataset_path = args.dataset_folderpath

    main(dataset_path)
