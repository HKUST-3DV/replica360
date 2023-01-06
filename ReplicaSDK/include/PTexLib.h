// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <pangolin/display/opengl_render_state.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>

#include <memory>
#include <string>

#include "Assert.h"
#include "MeshData.h"

#define XSTR(x) #x
#define STR(x) XSTR(x)

class PTexMesh {
 public:
  PTexMesh(const std::string& meshFile, const std::string& atlasFolder,
           bool renderSpherical);

  virtual ~PTexMesh();

  void RenderSubMesh(size_t subMesh, const pangolin::OpenGlRenderState& cam,
                     const Eigen::Vector4f& clipPlane, int lrC = 0,
                     const Eigen::Matrix4d& T_axis_align = Eigen::Matrix4d::Identity());

  void RenderSubMeshDepth(size_t subMesh,
                          const pangolin::OpenGlRenderState& cam,
                          const float depthScale,
                          const Eigen::Vector4f& clipPlane, int lrC = 0,
                          const Eigen::Matrix4d& T_axis_align = Eigen::Matrix4d::Identity());

  void Render(const pangolin::OpenGlRenderState& cam,
              const Eigen::Vector4f& clipPlane = Eigen::Vector4f(0.0f, 0.0f,
                                                                 0.0f, 0.0f),
              int lrC = 0,
              const Eigen::Matrix4d& T_axis_align = Eigen::Matrix4d::Identity());

  void RenderWireframe(const pangolin::OpenGlRenderState& cam,
                       const Eigen::Vector4f& clipPlane =
                           Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));

  void RenderDepth(const pangolin::OpenGlRenderState& cam,
                   const float depthScale = 1.0f,
                   const Eigen::Vector4f& clipPlane =
                       Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f),
                   int lrC = 0,
                   const Eigen::Matrix4d& T_axis_align = Eigen::Matrix4d::Identity());

  float Exposure() const;
  void SetExposure(const float& val);

  float Gamma() const;
  void SetGamma(const float& val);

  float Saturation() const;
  void SetSaturation(const float& val);

  void SetBaseline(const float& val);

  size_t GetNumSubMeshes() { return meshes.size(); }

  const Eigen::AlignedBox3f& GetOriginMeshBBox() { return origin_mesh_bbox_; }

 private:
  struct Mesh {
    pangolin::GlTexture atlas;
    pangolin::GlBuffer vbo;
    pangolin::GlBuffer ibo;
    pangolin::GlBuffer abo;
  };

  std::vector<MeshData> SplitMesh(const MeshData& mesh, const float splitSize);
  static void CalculateAdjacency(const MeshData& mesh,
                                 std::vector<uint32_t>& adjFaces);

  void LoadMeshData(const std::string& meshFile);
  void LoadAtlasData(const std::string& atlasFolder);

  float splitSize = 0.0f;
  uint32_t tileSize = 0;

  bool renderSpherical = false;

  pangolin::GlSlProgram shader;
  pangolin::GlSlProgram depthShader;

  float exposure = 1.0f;
  float gamma = 1.0f;
  float saturation = 1.0f;
  float baseline = 0.064f;
  bool isHdr = false;

  Eigen::AlignedBox3f origin_mesh_bbox_;

  static constexpr int ROTATION_SHIFT = 30;
  static constexpr int FACE_MASK = 0x3FFFFFFF;

  std::vector<std::unique_ptr<Mesh>> meshes;
};
