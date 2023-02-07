import os
import os.path as osp
import shutil
import open3d as o3d

# root_dir = '/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext/debug_20230105'
# replica_folders = [f for f in os.listdir(root_dir)
#                    if osp.isdir(osp.join(root_dir, f))]

# for f in replica_folders:
#     scene_folder = osp.join(root_dir, f)
#     spot_folders = [f for f in os.listdir(
#         scene_folder) if osp.isdir(osp.join(scene_folder, f))]
#     for sf in spot_folders:
#         sf_path = osp.join(scene_folder, sf)
#         print(f'------- clean {sf_path} -----------')

#         tmp_ply_filepath = osp.join(sf_path, 'rgb.ply')
#         tmp_json_filepath = osp.join(sf_path, 'rgb.json')
#         if osp.exists(tmp_ply_filepath):
#             os.remove(tmp_ply_filepath)
#         if osp.exists(tmp_json_filepath):
#             os.remove(tmp_json_filepath)

mesh_filepath = '/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/large_apartment_0/large_apartment_0_aligned.ply'
pcl = o3d.io.read_point_cloud(mesh_filepath)

o3d.visualization.draw_geometries([pcl])