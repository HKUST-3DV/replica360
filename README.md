# Replica 360 Dataset Generator
This repo contains the modified version of [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset) to generate the training and testing dataset used in [MatryODShka](https://visual.cs.brown.edu/projects/matryodshka-webpage/) paper. This is a companion to the training code [repo](https://github.com/brownvc/matryodshka).

The original ReplicaRenderer and ReplicaViewer remain the same. See original repo for basic usage. See below for 360 dataset generation.

## Setup
* Git Clone this repo.
* Run `git submodule update --init --recursive` to install 3rd party packages.
* Run `./download.sh` to download the mesh files.
* Run `./build.sh` to build all executables.


### Basic Usage
To generate the panoramic rgb/depth images, run 

```
./build/ReplicaSDK/ReplicaRendererDataset [dataset/scene_name/mesh.ply] [dataset/scene_name/textures] [camera_trajectory_file]  [output_dir]  [img_width] [img_height]
```

The data generation takes in a text file specifying the camera position, ods baseline and target camera positions for each navigable position within the scene. A single line in the input text file (camera_parameters.txt) is formatted as:
```
camera_position_x camera_position_y camera_position_z ods_baseline 
target1_offset_x target1_offset_y target1_offset_z 
target2_offset_x target2_offset_y target2_offset_z 
target3_offset_x target3_offset_y target3_offset_z
```
The existing text files contain navigable positions within each scene, sampled with [Habitat-SIM](https://github.com/facebookresearch/habitat-sim). 
Find all the existing text files in glob/.




