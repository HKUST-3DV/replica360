// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#define cimg_display 0

#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>

#include <Eigen/Geometry>
#include <string>
// #include "MirrorRenderer.h"
#include <DepthMeshLib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>


const bool b_render_depth = true;
const float k_depth_scale = 1000.f;

// load camera trajectory file
// FORMAT:
//  cam_position_x, cam_position_y, cam_position_z, ods, baseline,
//(interpolate) translate_x,       translate_y,       translate_z,
//(Rextrapolate)translate_x,       translate_y,       translate_z,
//(Lextrapolate)translate_x,       translate_y,       translate_z,
std::vector<std::vector<float>> readCameraTrajectory(
    const std::string& cam_position_file, const Eigen::Vector3f& bbox_min,
    const Eigen::Vector3f& bbox_max) {
  const float padding = 0.5f;
  std::vector<std::vector<float>> v_cam_position;
  // Eigen::Vector3f gravity_center(2.4068353176116943, 1.1334284543991089,
  //                                -1.4962862730026245);
  std::fstream in(cam_position_file);
  std::string line;
  while (std::getline(in, line)) {
    float value;
    std::stringstream ss(line);
    std::vector<float> cam_position;

    while (ss >> value) {
      cam_position.push_back(value);
    }
    // cam_position[0] -= gravity_center[0];
    // cam_position[1] -= gravity_center[1];
    // cam_position[2] -= gravity_center[2];
    // fix the Y value to 1.6m
    cam_position[1] = 1.6f;

    if ((cam_position[0] > bbox_min[0] + padding) &&
        (cam_position[0] < bbox_max[0] - padding) &&
        (cam_position[1] > bbox_min[2] + padding) &&
        (cam_position[1] < bbox_max[2] - padding) &&
        (cam_position[2] > bbox_min[1] + padding) &&
        (cam_position[2] < bbox_max[1] - padding)) {
      v_cam_position.push_back(cam_position);
    }
  }

  return v_cam_position;
}

int main(int argc, char* argv[]) {
  auto model_start = std::chrono::high_resolution_clock::now();
  std::cout << argc;

  ASSERT(
      argc == 7,
      "Usage: ./Path/to/ReplicaRenderDataset [mesh.ply] [textures_folderpath]  "
      "[camera_pose_filepath] [output_directory] [img_width] [img_height] ");

  const std::string mesh_filepath = argv[1];
  std::string texture_folderpath = argv[2];
  std::string cam_pose_filepath = argv[3];
  std::string output_folderpath = argv[4];
  int img_width = std::stoi(std::string(argv[5]));
  int img_height = std::stoi(std::string(argv[6]));

  ASSERT(pangolin::FileExists(mesh_filepath));
  ASSERT(pangolin::FileExists(texture_folderpath));
  ASSERT(pangolin::FileExists(output_folderpath));
  ASSERT(pangolin::FileExists(cam_pose_filepath));

  bool b_have_nav_cam = false;
  if (pangolin::FileExists(cam_pose_filepath)) b_have_nav_cam = true;

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
  bool b_render_equirect = true;
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
    v_cam_position = readCameraTrajectory(cam_pose_filepath, mesh_bbox_min, mesh_bbox_max);
  }

  size_t numSpots = 100;
  if (b_have_nav_cam) {
    numSpots = v_cam_position.size();
  }
  srand(2019);  // random seed

  // Setup a camera in GL coordinate system
  Eigen::Vector3f eye(0.0, 1.6, 0.0);
  if (b_have_nav_cam) {
    eye = Eigen::Vector3f(v_cam_position[0][0], v_cam_position[0][1], v_cam_position[0][2]);
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
  Eigen::Matrix4d T_world_cam = s_cam.GetModelViewMatrix();

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
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2);
      } else {
        ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));
      }
      glDisable(GL_CULL_FACE);
      glPopAttrib();  // GL_VIEWPORT_BIT
      frameBuffer.Unbind();

      // Download and save
      render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);
      char equirectFilename[1000];
      snprintf(equirectFilename, 1000, "%s/%s_%ld.jpeg",
               output_folderpath.c_str(), scene_name.c_str(), j);
      pangolin::SaveImage(image.UnsafeReinterpret<uint8_t>(),
                          pangolin::PixelFormatFromString("RGB24"),
                          std::string(equirectFilename), 100.0);

    } else if (b_render_equirect) {
      //

      Eigen::Vector3d curr_cam_pos(v_cam_position[j][0], v_cam_position[j][1], v_cam_position[j][2]);
      T_world_cam.block<3, 1>(0, 3) = curr_cam_pos;
      s_cam.GetModelViewMatrix() = T_world_cam;

      // Render
      frameBuffer.Bind();
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, img_width, img_height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glEnable(GL_CULL_FACE);

      // set parameters
      ptexMesh.SetExposure(0.01);
      ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2);

      glDisable(GL_CULL_FACE);
      glPopAttrib();  
      frameBuffer.Unbind();

      // Download and save
      render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

      char filename[1000];
      snprintf(filename, 1000, "%s/%s_%04zu.jpeg",
               output_folderpath.c_str(), scene_name.c_str(), (long unsigned)j);
      pangolin::SaveImage(image.UnsafeReinterpret<uint8_t>(),
                          pangolin::PixelFormatFromString("RGB24"),
                          std::string(filename), 100.0);

      if (b_render_depth) {
        // render depth image for the equirect image
        depthFrameBuffer.Bind();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, img_width, img_height);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        ptexMesh.RenderDepth(s_cam, k_depth_scale,
                             Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f), 2);
        glDisable(GL_CULL_FACE);
        glPopAttrib();  

        depthFrameBuffer.Unbind();

        char filename[1000];
        snprintf(filename, 1000, "%s/%s_%04zu.png", output_folderpath.c_str(),
                 scene_name.c_str(), (long unsigned)j);
        std::cout << "render depth image to " << filename << "\n";
        depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

        // convert to 16-bit int
        for (size_t i = 0; i < depthImage.Area(); i++)
          depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

        pangolin::SaveImage(depthImageInt.UnsafeReinterpret<uint8_t>(),
                            pangolin::PixelFormatFromString("GRAY16LE"),
                            std::string(filename), 100.0);
      }

      if (b_have_nav_cam) {
        if (j + 1 < numSpots) {
          T_world_cam = R_side * T_world_cam;
          s_cam.GetModelViewMatrix() = T_world_cam;
        }
      } else {
        continue;
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
