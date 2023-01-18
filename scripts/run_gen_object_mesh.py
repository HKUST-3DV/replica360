# from apriltags_eth import make_default_detector
import os
import os.path as osp
import json
import re
from pandas import array
import argparse

import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
import open3d.visualization.gui as gui

from scipy.spatial.transform import Rotation as R

from data_config import d3_40_colors_rgb, ReplicaXRDatasetConfig


def load_scene_semantic_info(semantic_info_filepath:str = None):

    if not os.path.exists(semantic_info_filepath):
        print(f'File {semantic_info_filepath} doesnt exist!!!')
        exit(-1)

    with open(semantic_info_filepath) as fd:
        data = json.load(fd)
        objects_data = data['objects']
        print(f'instance object mesh num: {len(objects_data)}')
    return objects_data

def save_as_trimesh(v_vertices, v_vertice_normals, v_vertice_colors, v_faces, output_filepath):
    origin_mesh = trimesh.Trimesh(vertices=v_vertices, faces=v_faces, vertex_normals=v_vertice_normals, vertex_colors=v_vertice_colors, process=False)
    bbox = origin_mesh.bounding_box.bounds
    # Compute location and scale
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0])
    # normalized mesh
    origin_mesh.apply_translation(-loc)
    origin_mesh.apply_scale(1.0/scale)
    origin_mesh.export(output_filepath)


def save_as_pointcloud(v_vertices, v_vertice_normals, v_vertice_colors, output_filepath):
    o3d_pcl = o3d.geometry.PointCloud()
    if v_vertices.shape[1] != 3:
        v_vertices.reshape(-1, 3)
    o3d_pcl.points = o3d.utility.Vector3dVector(v_vertices)
    if v_vertice_normals.shape[1] != 3:
        v_vertice_normals.reshape(-1, 3)
    o3d_pcl.normals = o3d.utility.Vector3dVector(v_vertice_normals)
    if v_vertice_colors.shape[1] != 3:
        v_vertice_colors.reshape(-1, 3)
    o3d_pcl.colors = o3d.utility.Vector3dVector(v_vertice_colors)

    o3d.io.write_point_cloud(output_filepath, o3d_pcl)


def gen_object_mesh_per_scene(mesh_filepath, scene_semantic_json_data, output_objects_folderpath):
    mesh = PlyData.read(mesh_filepath)
    v_vertices = mesh.elements[0]
    # print(f'vertices: {v_vertices}')
    v_faces = mesh.elements[1]
    # print(f'faces: {v_faces}')

    v_object_ids = {}
    for data in v_faces:
        face = data[0]
        obj_id = data[1]
        if not (obj_id in v_object_ids):
            v_object_ids[obj_id] = []
        v_object_ids[obj_id].append((face,))

    assert len(scene_semantic_json_data) > 0
    # generate objects' mesh
    for object_sem_data in scene_semantic_json_data:
        object_name = object_sem_data['class_name']
        object_id = object_sem_data['id']

        if not (object_name in ReplicaXRDatasetConfig().type2class):
            continue

        print(f"Saving object mesh for {object_name} ")
        out_folder = osp.join(output_objects_folderpath, object_name)
        if not osp.exists(out_folder):
            os.makedirs(out_folder)
        out_path = osp.join(out_folder, str(object_id)+'.ply')

        v_vertex_ids = []
        v_obj_vertices = []
        v_obj_faces = []
        for face in v_object_ids[object_id]:
            # process single face
            input_face_data = face[0]
            # print(f'input_face_data: {input_face_data}')
            num_vertex = input_face_data.shape[0]
            output_face_data = np.array([],dtype=np.uint32)
            # print(f'per face num_vertices: {num_vertex}')
            for i in range(num_vertex):
                input_vertex_id = input_face_data[i]
                if not( input_vertex_id in v_vertex_ids):
                    v_vertex_ids.append(input_vertex_id)
                    v_obj_vertices.append(v_vertices[input_vertex_id])
                    output_face_data = np.append(output_face_data, len(v_obj_vertices)-1)
                else:
                    index = v_vertex_ids.index(input_vertex_id)
                    output_face_data = np.append(output_face_data, index)
            # print(f'out_face_data: {output_face_data}')
            v_obj_faces.append((output_face_data,))

        faces_out = PlyElement.describe(np.array(v_obj_faces, dtype=[('vertex_indices', 'O')]), 'face')
        vertices_out = PlyElement.describe(np.array(v_obj_vertices, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        # PlyData([vertices_out, faces_out], text=True).write(out_path)

        # cvt to mesh
        vertices = []
        vertex_normals = []
        vertex_colors = []
        faces = []
        for v in vertices_out.data:
            vertices.append([v[0], v[1], v[2]])
            vertex_normals.append([v[3], v[4], v[5]])
            vertex_colors.append([v[6], v[7], v[8]])
        for f in faces_out.data:
            faces.append(f[0])

        vertices = np.asarray(vertices)
        vertex_normals = np.asarray(vertex_normals)
        vertex_colors = np.asarray(vertex_colors)
        faces = np.asarray(faces)
        
        # save_as_trimesh(vertices, vertex_normals, vertex_colors, faces, out_path)
        save_as_pointcloud(vertices, vertex_normals, vertex_colors/255.0, out_path)

def gen_colored_semantic_mesh(mesh_filepath, scene_semantic_json_data, output_scene_mesh_filepath):
    mesh = PlyData.read(mesh_filepath)
    v_vertices = mesh.elements[0]
    print(f'vertices: {v_vertices}')
    v_faces = mesh.elements[1]
    print(f'faces: {v_faces}')

    v_object_ids = {}
    for data in v_faces:
        face = data[0]
        obj_id = data[1]
        if not (obj_id in v_object_ids):
            v_object_ids[obj_id] = []
        v_object_ids[obj_id].append((face,))

    assert len(scene_semantic_json_data) > 0
    # generate objects' mesh
    for object_sem_data in scene_semantic_json_data:
        object_name = object_sem_data['class_name']
        object_id = object_sem_data['id']

        print(f'process object {object_name}')

        # object_name_filter = '(wall|floor|ceiling|Unknown|kitchen|stair|handrail|rack|undefined)'
        # if re.search(object_name_filter, object_name):
        #     continue
        # if not (object_name in ReplicaXRDatasetConfig().type2class):
        #     continue

        v_vertex_ids = []
        v_obj_vertices = []
        v_obj_faces = []
        for face in v_object_ids[object_id]:
            # process single face
            input_face_data = face[0]
            # print(f'input_face_data: {input_face_data}')
            num_vertex = input_face_data.shape[0]
            # print(f'per face num_vertices: {num_vertex}')
            for i in range(num_vertex):
                input_vertex_id = input_face_data[i]
                if not( input_vertex_id in v_vertex_ids):
                    v_vertex_ids.append(input_vertex_id)
                    # v_obj_vertices.append()
                    v_vertices[input_vertex_id][6] = d3_40_colors_rgb[object_id%40][0]
                    v_vertices[input_vertex_id][7] = d3_40_colors_rgb[object_id%40][1]
                    v_vertices[input_vertex_id][8] = d3_40_colors_rgb[object_id%40][2]

                else:
                    index = v_vertex_ids.index(input_vertex_id)

            # print(f'out_face_data: {output_face_data}')
        #     v_obj_faces.append((output_face_data,))

        # faces_out = PlyElement.describe(np.array(v_obj_faces, dtype=[('vertex_indices', 'O')]), 'face')
        # vertices_out = PlyElement.describe(np.array(v_obj_vertices, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        # PlyData([vertices_out, faces_out], text=True).write(out_path)

    # cvt to mesh
    vertices = []
    vertex_normals = []
    vertex_colors = []
    faces = []
    for v in v_vertices.data:
        vertices.append([v[0], v[1], v[2]])
        vertex_normals.append([v[3], v[4], v[5]])
        vertex_colors.append([v[6], v[7], v[8]])
    for f in v_faces.data:
        faces.append(f[0])

    vertices = np.asarray(vertices)
    vertex_normals = np.asarray(vertex_normals)
    vertex_colors = np.asarray(vertex_colors)
    faces = np.asarray(faces)
    origin_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals, vertex_colors=vertex_colors, process=False)
    # origin_mesh.compute_vertex_normals()
    target_position = np.array([3.5, -2.6653447, 0.0])
    origin_mesh.apply_translation(-target_position)
    origin_mesh.export(output_scene_mesh_filepath)

def main(input_folder, output_folder):

    scene_folders = [f for f in os.listdir(input_folder) if osp.isdir(osp.join(input_folder, f))]
    for scene_name in scene_folders:
        if 'frl_apartment_0' != scene_name:
            continue

        print(f'Processing scene_name ----------------------- {scene_name} ------------------------')

        scene_path = osp.join(input_folder, scene_name)
        mesh_filepath = osp.join(scene_path,'habitat/mesh_semantic.ply')
        semantic_info_filepath = osp.join(scene_path, 'habitat/info_semantic.json')
        output_folderpath = osp.join(output_folder, scene_name)
        if not osp.exists(output_folderpath):
            os.makedirs(output_folderpath)
        saved_color_sem_mesh_filepath = osp.join(scene_path,'habitat/mesh_semantic_colored.ply')

        instance_semantic_data = load_scene_semantic_info(semantic_info_filepath)

        gen_object_mesh_per_scene(mesh_filepath, instance_semantic_data, output_folderpath)
        # gen_colored_semantic_mesh(mesh_filepath, instance_semantic_data, saved_color_sem_mesh_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1")
    parser.add_argument("--output_folder", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext/replica_obj")

    args = parser.parse_args()
    dataset_folderpath = args.dataset_folder
    output_folder = args.output_folder
    if not osp.exists(dataset_folderpath):
        print(f'Path {dataset_folderpath} doesnt exist!!!')
        exit(-1)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    main(dataset_folderpath, output_folder)
