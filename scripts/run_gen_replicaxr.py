from genericpath import isdir
import os, sys, signal
import os.path as osp
import argparse
import glob

g_running = True

def main(dataset_dir_path, output_dir_path, exe_dir_path):

    scene_folders = [folder for folder in os.listdir(dataset_dir_path) if osp.isdir(osp.join(dataset_dir_path,folder))]
    assert len(scene_folders)

    exe_path = osp.join(exe_dir_path, 'build/ReplicaSDK/ReplicaRendererDataset')

    img_width = '1024'
    img_height = '512'
    for scene in scene_folders:
        if g_running:
            if scene == 'large_apartment_0':
                continue
            scene_ply_filepath = osp.join(dataset_dir_path, scene, scene+'_aligned.ply')
            texture_folderpath = osp.join(dataset_dir_path, scene, 'textures')
            cam_traj_filepath = glob.glob(osp.join(dataset_dir_path, scene, scene+'_trajectory*.txt'))
            # output_path = osp.join(output_dir_path, scene)

            scene_idx = 0
            for traj_file in cam_traj_filepath:
                output_path = osp.join(output_dir_path, '%03d'%(scene_idx))
                scene_idx += 1

                cmd = [exe_path, scene_ply_filepath, texture_folderpath, traj_file, output_path, img_width, img_height]
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folderpath", type=str,
                        default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_v1/")
    parser.add_argument("--output_folderpath", type=str, default="/media/ziqianbai/BACKPACK_DATA1/Replica_all/replica_for_panocontext/")
    parser.add_argument("--exe_ws", type=str, default="/home/ziqianbai/Projects/vlab/matryodshka-replica360/")

    args = parser.parse_args()
    dataset_dir_path = args.dataset_folderpath
    output_dir_path = args.output_folderpath
    exe_dir_path = args.exe_ws

    assert osp.exists(dataset_dir_path)
    assert osp.exists(exe_dir_path)
    if not osp.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        global g_running
        g_running = False
        print('g_running: ', g_running) 
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    try:
        main(dataset_dir_path, output_dir_path, exe_dir_path)
    finally:
        print("Goodbye")


