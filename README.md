# Replica 360 Dataset
The original ReplicaRenderer and ReplicaViewer remain the same. See [original repo](https://github.com/facebookresearch/Replica-Dataset) for basic usage. See below for 360 dataset generation.

## Setup
* Run `./download.sh` to download the mesh files.
* Run `./build.sh` to build all executables.

## 360 Dataset generation
To generate the training and testing dataset used in [Matryodshka](https://github.com/brownvc/matryodshka) repo [[1]](#1), run 
```
./scripts/gen-3dof-360-*-data.sh OUTPUT_DIR WIDTH HEIGHT
```

### Basic Usage
```
./build/ReplicaSDK/ReplicaRendererDataset dataset/scene_name/mesh.ply 
dataset/scene_name/textures dataset/scene_name/glass.sur camera_parameters[file.txt / n] 
spherical[y/n] output/dir/ output_width output_height
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


## Video Rendering
This repo also supports video dataset rendering specifically. 
### Basic Usage 
```
./build/ReplicaSDK/ReplicaVideoRenderer path/to/scene/mesh.ply 
path/to/scene/textures path/to/scene/glass.sur camera_parameters.txt 
spherical[y/n] output/dir/ width height
```
The difference is the format of a single line in input text file for video rendering is:
```
camera_position_x camera_position_y camera_position_z 
lookat_x lookat_y lookat_z ods_baseline 
rotation_x rotation_y rotation_z target_x target_y target_z
```

To generate a video path, one can use `glob/gen_video_path.py`. See `glob/example_script` for an example of straight path generation.

## Depth-based Mesh rendering
This repo also support depth-based mesh rendering as in <em>Motion parallax for 360° RGBD video</em> [[2]](#2)
### Basic Usage:
```
./build/ReplicaSDK/DepthMeshRendererBatch images.txt camera_poses.txt 
OUT_DIR SPHERICAL<y|n> y y OUTPUT_WIDTH OUTPUT_HEIGHT
```

images.txt should contain lines of format:
```
<ods_left_color>.png <ods_left_depth>.png
```
camera_poses.txt should contain the desired camera offsets for rendering as below for each line:
```
<translate_x> <translate_y> <translate_z>
```
See more instructions in assets/EVALUATION_HOWTO.
