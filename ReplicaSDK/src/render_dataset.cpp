// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#define cimg_display 0

#include <DepthMeshLib.h>
#include <EGL.h>
#include <PTexLib.h>
#include <glog/logging.h>
#include <pangolin/image/image_convert.h>

#include <Eigen/Geometry>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include "file_system_tools.h"

const bool b_render_depth = true;
const bool b_render_equirect = true;
const float k_depth_scale = 4000.f;

// load camera trajectory file
// camera trajectory in camera frame: t_c_w
std::vector<std::vector<float>> readCameraTrajectory(
    const std::string& cam_position_file, const Eigen::Vector3f& bbox_min,
    const Eigen::Vector3f& bbox_max) {
  std::vector<std::vector<float>> v_cam_position;

  std::ifstream in(cam_position_file);
  if (!in.is_open()) {
    LOG(FATAL) << "Fail to open file " << cam_position_file;
  }
  std::string line;
  while (std::getline(in, line)) {
    float pos_x, pos_y, pos_z;
    std::stringstream ss(line);

    ss >> pos_x >> pos_y >> pos_z;

    std::vector<float> cam_position = {pos_x, pos_y, pos_z};

    // // fix the Y value to 1.6m
    // cam_position[1] = 1.6f;
    v_cam_position.push_back(cam_position);
  }
  // std::cout << "camera trajectory lens: " << v_cam_position.size() << "\n";
  return v_cam_position;
}

// save camera pose to file
bool saveCameraPose(const std::string &filepath, const Eigen::Matrix4d &T_w_c) {
  std::ofstream ofs(filepath);
  if (!ofs.is_open()){
    LOG(FATAL) << "Fail to open file : " << filepath;
    return false;
  }

  Eigen::Quaterniond q_w_c(T_w_c.block<3, 3>(0, 0));
  Eigen::Vector3d t_w_c = T_w_c.block<3, 1>(0, 3);
  // tx ty tz qx qy qz qw
  ofs << t_w_c[0] << " " << t_w_c[1] << " " << t_w_c[2] << " " << q_w_c.x() << " " << q_w_c.y() << " " << q_w_c.z() << " " << q_w_c.w() << "\n";
  ofs.close();

  return true;
}

bool loadMeshTransformationMatrix(const std::string &filepath, Eigen::Matrix4d &T_axis_align) {
  std::ifstream ifs(filepath);
  if (!ifs.is_open()){
    LOG(FATAL) << "Fail to open " << filepath;
    return false;
  }

  T_axis_align = Eigen::Matrix4d::Identity();

  int idx = 0;
  std::string line_str;
  std::stringstream ss;

  while(!ifs.eof()) {
    ss.clear();
    std::getline(ifs, line_str);
    if (line_str.empty()) continue;

    ss.str(line_str);
    ss >> T_axis_align(idx, 0) >> T_axis_align(idx, 1) >> T_axis_align(idx, 2) >> T_axis_align(idx, 3);
    idx ++;
  }

  ifs.close();
  std::cout << "load T_aa: \n " << T_axis_align << "\n";
  return true;
}

int main(int argc, char* argv[]) {
  auto model_start = std::chrono::high_resolution_clock::now();
  std::cout << argc;

  ASSERT(
      argc == 8,
      "Usage: ./Path/to/ReplicaRenderDataset [mesh.ply] [textures_folderpath]  "
      "[camera_pose_filepath] [output_directory] [img_width] [img_height] [mesh_transform_filepath]");

  const std::string mesh_filepath = argv[1];
  std::string texture_folderpath = argv[2];
  std::string cam_pose_filepath = argv[3];
  std::string output_folderpath = argv[4];
  int img_width = std::stoi(std::string(argv[5]));
  int img_height = std::stoi(std::string(argv[6]));
  std::string mesh_transform_filepath = argv[7];

  ASSERT(common::fileExists(mesh_filepath));
  ASSERT(common::pathExists(texture_folderpath));
  ASSERT(common::fileExists(cam_pose_filepath));
  ASSERT(common::fileExists(mesh_transform_filepath));
  if (!common::pathExists(output_folderpath)) {
    common::createPath(output_folderpath);
  }

  bool b_have_nav_cam = true;

  std::string scene_name;
  {  // get scene name
    const size_t last_slash_idx = mesh_filepath.rfind("/");
    const size_t second2last_slash_idx =
        mesh_filepath.substr(0, last_slash_idx).rfind("/");
    if (std::string::npos != last_slash_idx) {
      scene_name =
          mesh_filepath.substr(second2last_slash_idx + 1,
                               last_slash_idx - second2last_slash_idx - 1);
      std::cout << "Generating from scene_name " << scene_name << std::endl;
    }
  }

  // Setup EGL
  EGLCtx egl;
  egl.PrintInformation();

  // Don't draw backfaces
  GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  // rgb texture
  pangolin::GlTexture render(img_width, img_height);
  pangolin::GlRenderBuffer renderBuffer(img_width, img_height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);
  // depth texture
  pangolin::GlTexture depthTexture(img_width, img_height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // For cubemap dataset: rotation matrix of 90 degree for each face of the
  // cubemap t -> t -> t -> u -> d
  Eigen::Transform<double, 3, Eigen::Affine> t(Eigen::AngleAxis<double>(0.5 * M_PI, Eigen::Vector3d::UnitY()));
  Eigen::Transform<double, 3, Eigen::Affine> u(Eigen::AngleAxis<double>(0.5 * M_PI, Eigen::Vector3d::UnitX()));
  Eigen::Transform<double, 3, Eigen::Affine> d(Eigen::AngleAxis<double>(M_PI, Eigen::Vector3d::UnitX()));
  Eigen::Matrix4d R_side = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d R_up = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d R_down = Eigen::Matrix4d::Identity();
  R_side = t.matrix();
  R_up = u.matrix();
  R_down = d.matrix();

  // load mesh and textures
  PTexMesh ptexMesh(mesh_filepath, texture_folderpath, b_render_equirect);
  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(img_width, img_height);
  pangolin::ManagedImage<float> depthImage(img_width, img_height);
  pangolin::ManagedImage<uint16_t> depthImageInt(img_width, img_height);

  // get original mesh bbox
  Eigen::Vector3f mesh_bbox_min = ptexMesh.GetOriginMeshBBox().min();
  Eigen::Vector3f mesh_bbox_max = ptexMesh.GetOriginMeshBBox().max();
  std::cout << "origin mesh bbox: min: " << mesh_bbox_min.transpose()
            << " , max: " << mesh_bbox_max.transpose() << "\n";
  // load camera trajectory
  std::vector<std::vector<float>> v_cam_position;
  if (b_have_nav_cam) {
    v_cam_position =
        readCameraTrajectory(cam_pose_filepath, mesh_bbox_min, mesh_bbox_max);
  }

  size_t numSpots = 100;
  if (b_have_nav_cam) {
    numSpots = v_cam_position.size();
  }
  std::cout << "Spots number: " << numSpots << "\n";
  srand(2019);  // random seed

  // Setup a camera in GL coordinate system
  Eigen::Vector3f eye(0.0, 1.6, 0.0);
  if (b_have_nav_cam) {
    eye = Eigen::Vector3f(v_cam_position[0][0], v_cam_position[0][1],
                          v_cam_position[0][2]);
    std::cout << "First camera position:" << eye[0] << " " << eye[1] << " "
              << eye[2] << "\n";
  }
  // look_at: forward direction
  Eigen::Vector3f center = eye + Eigen::Vector3f(0, 1, 0);
  // up: upward direction
  Eigen::Vector3f up(0, 0, 1);
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          img_width, img_height, img_width / 2.0f, img_width / 2.0f,
          (img_width - 1.0f) / 2.0f, (img_height - 1.0f) / 2.0f, 0.1f, 100.0f),
      pangolin::ModelViewLookAtRDF(eye[0], eye[1], eye[2], center[0], center[1],
                                   center[2], up[0], up[1], up[2]));

  // Start at some origin
  Eigen::Matrix4d T_cam_world = s_cam.GetModelViewMatrix();
  // mesh transformation , to be used in GLSL code
  Eigen::Matrix4d T_axis_align = Eigen::Matrix4d::Identity();
  loadMeshTransformationMatrix(mesh_transform_filepath, T_axis_align);

  // rendering the dataset
  for (size_t j = 0; j < numSpots; j++) {
    if (!b_have_nav_cam) {
      // Render
      frameBuffer.Bind();
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, img_width, img_height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glEnable(GL_CULL_FACE);

      // set parameters
      ptexMesh.SetExposure(0.01);
      if (b_render_equirect) {
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2, T_axis_align);
      } else {
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 0, T_axis_align);
      }
      glDisable(GL_CULL_FACE);
      glPopAttrib();  // GL_VIEWPORT_BIT
      frameBuffer.Unbind();

      // Download and save
      render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);
      char img_file_folder[1000];
      snprintf(img_file_folder, 1000, "%s/%05ld", output_folderpath.c_str(), j);
      if (!common::pathExists(img_file_folder)) {
        common::createPath(img_file_folder);
      }
      std::string img_file_path = std::string(img_file_folder) + "/rgb.png";
      pangolin::SaveImage(image.UnsafeReinterpret<uint8_t>(),
                          pangolin::PixelFormatFromString("RGB24"),
                          img_file_path, 100.0);

    } else {
      // model_view matrix is T_c_w
      Eigen::Vector3d curr_cam_pos(v_cam_position[j][0], v_cam_position[j][1],
                                   v_cam_position[j][2]);
      T_cam_world.block<3, 1>(0, 3) = curr_cam_pos;
      // if (j + 1 < numSpots) {
      //     T_cam_world = R_side * T_cam_world;
      // }
      std::cout <<"T_c_w:\n" << T_cam_world << "\n";
      s_cam.GetModelViewMatrix() = T_cam_world;

      // Render
      frameBuffer.Bind();
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, img_width, img_height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glEnable(GL_CULL_FACE);

      // set parameters
      ptexMesh.SetExposure(0.01);
      if (b_render_equirect) {
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2, T_axis_align);
      } else {
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 0, T_axis_align);
      }

      glDisable(GL_CULL_FACE);
      glPopAttrib();
      frameBuffer.Unbind();

      // Download and save
      render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

      char img_file_folder[1000];
      snprintf(img_file_folder, 1000, "%s/%05ld", output_folderpath.c_str(), j);
      if (!common::pathExists(img_file_folder)) {
        common::createPath(img_file_folder);
      }
      std::string img_file_path = std::string(img_file_folder) + "/rgb.png";
      // char img_file_path[1000];
      // snprintf(img_file_path, 1000, "%s/%s_%05ld.png",
      //          output_folderpath.c_str(), scene_name.c_str(), j);
      pangolin::SaveImage(image.UnsafeReinterpret<uint8_t>(),
                          pangolin::PixelFormatFromString("RGB24"),
                          std::string(img_file_path), 100.0);

      std::string pos_filepath = std::string(img_file_folder) + "/pose.txt";
      saveCameraPose(pos_filepath, T_cam_world.inverse());

      if (b_render_depth) {
        // render depth image for the equirect image
        depthFrameBuffer.Bind();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, img_width, img_height);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        ptexMesh.RenderDepth(s_cam, k_depth_scale,
                             Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2, T_axis_align);
        glDisable(GL_CULL_FACE);
        glPopAttrib();

        depthFrameBuffer.Unbind();

        char img_file_folder[1000];
        snprintf(img_file_folder, 1000, "%s/%05ld", output_folderpath.c_str(), j);
        if (!common::pathExists(img_file_folder)) {
          common::createPath(img_file_folder);
        }
        std::string img_file_path = std::string(img_file_folder) + "/depth.png";
        // std::cout << "render depth image to " << img_file_path << "\n";
        depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

        // convert to 16-bit int
        for (size_t i = 0; i < depthImage.Area(); i++)
          depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

        pangolin::SaveImage(depthImageInt.UnsafeReinterpret<uint8_t>(),
                            pangolin::PixelFormatFromString("GRAY16LE"),
                            std::string(img_file_path), 100.0);
      }

      std::cout << "\r Spot " << j + 1 << "/" << numSpots << std::endl;
    }
  }

  auto model_stop = std::chrono::high_resolution_clock::now();
  auto model_duration = std::chrono::duration_cast<std::chrono::microseconds>(model_stop - model_start);
  std::cout << "Time taken rendering the scene " << scene_name << ": "
            << model_duration.count() << " microseconds" << std::endl;

  return 0;
}
