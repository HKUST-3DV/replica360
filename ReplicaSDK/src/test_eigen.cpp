#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  Eigen::Vector3d origin_up = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d new_up = Eigen::Vector3d::UnitY();

  Eigen::Vector3d origin_front = Eigen::Vector3d::UnitY();
  Eigen::Vector3d new_front = -Eigen::Vector3d::UnitZ();

  Eigen::Quaterniond rot_up =
      Eigen::Quaterniond::FromTwoVectors(new_up, origin_up);

  Eigen::Matrix3d rot_up_matrix = rot_up.toRotationMatrix();
  std::cout << "rotation up: \n" << rot_up_matrix << "\n";
  std::cout << "rotation_up_inv: \n" << rot_up_matrix.transpose() << "\n";

  Eigen::AngleAxisd rot_x(M_PI / 2, Eigen::Vector3d::UnitX());
  std::cout << "rot_x pi/2: \n" << rot_x.matrix() << "\n";

  Eigen::Quaterniond rot_front = Eigen::Quaterniond::FromTwoVectors(
      rot_up_matrix * new_front, origin_front);
  Eigen::Matrix3d rot_front_matrix = rot_front.toRotationMatrix();
  std::cout << "rotation front: \n" << rot_front_matrix << "\n";
  return 0;
}