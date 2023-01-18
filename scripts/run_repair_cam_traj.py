from curses.ascii import isdigit
from genericpath import isdir
import os
import os.path as osp
from random import sample, shuffle
import numpy as np
import json
import argparse
import glob
from functools import reduce

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

# load sample camera pose file of Open3D
def load_cam_pose(cam_pose_file):
    with open(cam_pose_file, 'r') as ifs:
        trajectory = json.load(ifs)
        assert( trajectory['class_name'] == "PinholeCameraParameters")

        T_w_c = np.eye(4)
        intrinsic_matrix = np.eye(3)

        extrinsics = trajectory['extrinsic']
        T_w_c[:,0] = extrinsics[:4]
        T_w_c[:,1] = extrinsics[4:8]
        T_w_c[:,2] = extrinsics[8:12]
        T_w_c[:,3] = extrinsics[12:]
        # print(T_w_c)

        intrinsics = trajectory['intrinsic']
        img_width = intrinsics['width']
        img_height = intrinsics['height']
        intrin_mat = intrinsics['intrinsic_matrix']
        intrinsic_matrix[:,0] = intrin_mat[:3]
        intrinsic_matrix[:,1] = intrin_mat[3:6]
        intrinsic_matrix[:,2] = intrin_mat[6:]
        # print(intrinsic_matrix)
        return T_w_c, intrinsic_matrix

def read_spots_need_repaired(spot_need_repair_filepath):
    v_spots = []
    with open(spot_need_repair_filepath) as ifs:
        lines = ifs.readlines()
        # print(lines)
        for line in lines:
            line = line.rstrip('\n')
            if len(line) == 0:
                continue
            v_spots.append(line.split()[-1])
    return v_spots
        
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

def load_axis_aligned_mesh_transfomation(filepath):
    T = np.eye(4)
    with open(filepath, 'r') as ifs:
        lines_data = ifs.readlines()
        for idx in range(len(lines_data)):
            data = lines_data[idx].strip().split()
            T[idx, 0] = float(data[0])
            T[idx, 1] = float(data[1])
            T[idx, 2] = float(data[2])
            T[idx, 3] = float(data[3])
    return T


def gen_repair_trajectory(sample_traj_folderpath, saved_cam_trajectory_filepath, repair_spot_num=0):
    v_cam_pose_files = [ osp.join(sample_traj_folderpath, f)  for f in os.listdir(sample_traj_folderpath) if f.endswith('.json')]
    v_cam_pose_files.sort(key=lambda x:osp.basename(x))
    print(v_cam_pose_files)

    v_rotation_in = np.zeros([0, 4])
    v_pos_x_in = []
    v_pos_y_in = []
    v_pos_z_in = []
    for cam_pose_file in v_cam_pose_files:
        T_w_c, intrinsic = load_cam_pose(cam_pose_file)
        # T_c_w = np.linalg.inv(T_w_c)
        v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(T_w_c[:3,:3]).as_quat()], axis=0)
        v_pos_x_in.append(T_w_c[0,3])
        v_pos_y_in.append(T_w_c[1,3])
        v_pos_z_in.append(T_w_c[2,3])

    in_times = np.arange(0, len(v_rotation_in)).tolist()
    out_times = np.linspace(0, len(v_rotation_in)-1, len(v_rotation_in)*20).tolist()
    print(f'in_times: {(in_times)}')
    print(f'out_times: {(out_times)}')
    v_rotation_in = Rotation.from_quat(v_rotation_in)
    slerp = Slerp(in_times, v_rotation_in)
    v_interp_rotation = slerp(out_times)

    fx = interp1d(in_times, np.array(v_pos_x_in), kind='quadratic')
    fy = interp1d(in_times, np.array(v_pos_y_in), kind='quadratic')
    fz = interp1d(in_times, np.array(v_pos_z_in), kind='quadratic')
    v_interp_xs = fx(out_times)
    v_interp_ys = fy(out_times)
    v_interp_zs = fz(out_times)

    v_cam_position = []
    for idx in range(len(out_times)):
        trans = [v_interp_xs[idx], 4.35, v_interp_zs[idx]]
        v_cam_position.append(trans)
    
    spots_num = len(v_cam_position)
    spot_interval = spots_num // 100
    v_cam_position_sampled = v_cam_position[0:spots_num:spot_interval]
    with open(saved_cam_trajectory_filepath, 'w') as ofs:
        for cam_pos in v_cam_position_sampled:
            pos_str = reduce(lambda a, b: str(a) + " " + str(b), cam_pos)
            ofs.write(pos_str)
            ofs.write('\n')

    return True

# convert camera trajectory of habitat-sim into trajectory of OpenGL
def transform_and_save_cam_trajectory(cam_trajectory_filepath, saved_cam_trajectory_filepath, gravity_center, rot_vec=None, expect_spot_num=None):
    v_cam_position = []
    rot_x_90 = Rotation.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
    rot_x_90n = np.transpose(rot_x_90)

    traj_file_name = osp.basename(cam_trajectory_filepath)
    large_apart_0_traj_files = ['large_apartment_0_trajectory0.json', 'large_apartment_0_trajectory1.json',
                                'large_apartment_0_trajectory2.json',
                                'large_apartment_0_trajectory3.json',
                                'large_apartment_0_trajectory0_repair.json',
                                'large_apartment_0_trajectory7.json']
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
            if rot_vec is not None:
                pos_w = np.linalg.inv(rot_vec) @ pos_w

            # camera-frame
            #        __  Z
            #         /|
            #        /
            #       /___________\ X
            #      |            /
            #      |
            #      |
            #      |
            #      |
            #     \|/ Y
            # rotate to world frame
            # transform to camera frame
            pos_c = -rot_x_90n @ pos_w
            # fix pos_y == 1.6m
            if traj_file_name in large_apart_0_traj_files:
                pos_c[1] = 1.6 + 2.75
            else:
                pos_c[1] = 1.6
            v_cam_position.append([pos_c[0], pos_c[1], pos_c[2]])

    spots_num = len(v_cam_position)

    if expect_spot_num is not None:
        assert spots_num > expect_spot_num
        spots_interval = spots_num // expect_spot_num
    else:
        assert spots_num > 100
        spots_interval = spots_num // 100
    v_cam_position_sampled = v_cam_position[0:spots_num:spots_interval]
    with open(saved_cam_trajectory_filepath, 'w') as ofs:
        for cam_pos in v_cam_position_sampled:
            pos_str = reduce(lambda a, b: str(a) + " " + str(b), cam_pos)
            ofs.write(pos_str)
            ofs.write('\n')

    return True

def render_spots(scene_dir_path, output_dir_path, repair_house_name, cam_traj_filepath):

    assert osp.exists(cam_traj_filepath), "camera trajectory file doesnt exist!!!"
    scene = osp.basename(scene_dir_path)
    exe_path = './build/ReplicaSDK/ReplicaRendererDataset'

    img_width = '1024'
    img_height = '512'

    scene_ply_filepath = osp.join(scene_dir_path, scene+'.ply')
    texture_folderpath = osp.join(scene_dir_path, 'textures')
    # cam_traj_filepath = glob.glob(osp.join(scene_dir_path, scene+'_trajectory*_repair.txt'))
    # print(cam_traj_filepath)
    mesh_transform_filepath = osp.join(scene_dir_path, 'axis_aligned_transform.txt')

    output_path = osp.join(output_dir_path, repair_house_name + '_repair')


    cmd = [exe_path, scene_ply_filepath, texture_folderpath, cam_traj_filepath, output_path, img_width, img_height, mesh_transform_filepath]
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)

    return repair_house_name + '_repair'

def mv_repair_spots(raw_render_out_folderpath, need_repair_houses, repair_out_folderpath, repair_result_houses, v_need_repair_spots):
    for house_name in need_repair_houses:
        raw_house_folderpath = osp.join(raw_render_out_folderpath, house_name)
        assert (house_name+'_repair') in repair_result_houses, 'Unconsistent raw house folder and new house folder'
        repair_house_folderpath = osp.join(repair_out_folderpath, house_name+'_repair')

        repair_spots_folder = [f for f in os.listdir(repair_house_folderpath) if osp.isdir(osp.join(repair_house_folderpath, f)) and f.isdigit()]
        shuffle(repair_spots_folder)
        
        spot_idx = 0
        for need_repair_spot in v_need_repair_spots:
            raw_spot_folder = osp.join(raw_house_folderpath, need_repair_spot)
            new_spot_folder = osp.join(repair_house_folderpath, repair_spots_folder[spot_idx])
            cmd = f'cp  {new_spot_folder}/*  {raw_spot_folder} '
            print(cmd)
            spot_idx += 1
            os.system(cmd)

    
    

if __name__ == '__main__':
    # re-render rgb/depth/bbox in the specific scene
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_folderpath", type=str,
                        default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/frl_apartment_3")
    parser.add_argument("--raw_out_folderpath", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext/debug_20230105/")
    # parser.add_argument("--repair_houses", nargs='+', type=str, default=["frl_apartment_3_000"])
    parser.add_argument("--repair_house",  type=str, default="frl_apartment_3_000")
    parser.add_argument("--repair_out_folderpath", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext")

    args = parser.parse_args()
    scene_folderpath = args.scene_folderpath
    raw_render_out_folderpath = args.raw_out_folderpath
    need_repair_house_name = args.repair_house
    repair_out_folderpath = args.repair_out_folderpath
    
    scene_name = osp.basename(scene_folderpath)
    scene_semantic_filepath = osp.join(scene_folderpath, 'semantic.json')
    saved_axis_align_mesh_T_filepath = osp.join(scene_folderpath, 'axis_aligned_transform.txt')
    sample_traj_filepath = osp.join(scene_folderpath, scene_name+'_trajectory.json')
    saved_cam_trajectory_filepath = osp.join(scene_folderpath, scene_name+'_trajectory_repair.txt')
    spot_need_repair_filepath = osp.join(raw_render_out_folderpath, need_repair_house_name, 'unused_scans.txt')

    if not osp.exists(sample_traj_filepath):
        print(f'Folder {sample_traj_filepath} doesnt exist!')
        exit(-1)
    
    # load invalid scans data
    v_repair_spots = read_spots_need_repaired(spot_need_repair_filepath)
    print(v_repair_spots)

    # read gravity center and direction
    gravity_center, gravity_direction = load_scene_gravity(scene_semantic_filepath)

    if osp.exists(saved_axis_align_mesh_T_filepath):
        # T_aa = np.load(saved_axis_align_mesh_T_filepath)
        T_aa = load_axis_aligned_mesh_transfomation(saved_axis_align_mesh_T_filepath)
    else:
        T_aa = None
    # generate trajectory using trajectory samples
    # gen_repair_trajectory(sample_traj_folderpath, saved_cam_trajectory_filepath, len(v_repair_spots))
    transform_and_save_cam_trajectory(sample_traj_filepath, saved_cam_trajectory_filepath, gravity_center, rot_vec=T_aa[:3, :3], expect_spot_num=len(v_repair_spots))

    repair_result_house_name = render_spots(scene_folderpath, repair_out_folderpath, need_repair_house_name, saved_cam_trajectory_filepath)
    print(repair_result_house_name)
    # repair_result_house_name = 'large_apartment_0_003_repair'
    mv_repair_spots(raw_render_out_folderpath, [need_repair_house_name], repair_out_folderpath, [repair_result_house_name], v_repair_spots)

