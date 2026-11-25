#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <memory>
#include <chrono>
#include <Eigen/Dense>

#include <discrete_transformation/discrete_transformation.hpp>

namespace cpu {
class VoxelMaps;

struct AngularInfo {
  Eigen::Vector3i num_division;
  Eigen::Vector3d rpy_res;
  Eigen::Vector3d min_rpy;
};

struct CandidatePose {
  DiscreteTransformation<double> trans;
  double score_ratio;  // score as fraction of total source points
};

class BBS3D {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BBS3D();
  ~BBS3D();

  void set_tar_points(const std::vector<Eigen::Vector3d>& points, double min_level_res, int max_level);

  void set_src_points(const std::vector<Eigen::Vector3d>& points);

  void set_trans_search_range(const std::vector<Eigen::Vector3d>& points);

  void set_trans_search_range(const Eigen::Vector3d& min_xyz, const Eigen::Vector3d& max_xyz);

  void set_angular_search_range(const Eigen::Vector3d& min_rpy, const Eigen::Vector3d& max_rpy) {
    min_rpy_ = min_rpy;
    max_rpy_ = max_rpy;
  }

  void set_voxel_expantion_rate(const double rate) {
    v_rate_ = rate;
    inv_v_rate_ = 1.0 / rate;
  }

  void set_num_threads(const int num_threads) { num_threads_ = num_threads; }

  /// 邻居体素命中分数（核心体素固定 2.0）：1.0 = 加权 BBS，2.0 = 原始 3D-BBS
  /// 需在 set_tar_points() 之前调用，因为权重在建体素图时就写进了 buckets
  void set_neighbor_score(const double neighbor_score) { neighbor_score_ = neighbor_score; }

  void set_score_threshold_percentage(double percentage) { score_threshold_percentage_ = percentage; }

  void enable_timeout() { use_timeout_ = true; }

  void disable_timeout() { use_timeout_ = false; }

  void set_timeout_duration_in_msec(const int msec);

  std::vector<Eigen::Vector3d> get_src_points() const { return src_points_; }

  bool set_voxelmaps_coords(const std::string& folder_path);

  std::pair<Eigen::Vector3d, Eigen::Vector3d> get_trans_search_range() const {
    return std::pair<Eigen::Vector3d, Eigen::Vector3d>{min_xyz_, max_xyz_};
  }

  std::vector<Eigen::Vector3d> get_angular_search_range() const { return std::vector<Eigen::Vector3d>{min_rpy_, max_rpy_}; }

  Eigen::Matrix4d get_global_pose() const { return global_pose_; }

  double get_best_score() const { return best_score_; }

  double get_elapsed_time() const { return elapsed_time_; }

  double get_best_score_percentage() {
    if (src_points_.size() == 0)
      return 0.0;
    else
      return static_cast<double>(best_score_) / (2.0 * src_points_.size());
  };

  bool has_timed_out() { return has_timed_out_; }

  bool has_localized() { return has_localized_; }

  void localize();

  // Confidence evaluation
  double get_confidence_score() const { return confidence_score_; }
  double get_peak_margin() const { return peak_margin_; }
  std::vector<DiscreteTransformation<double>> get_top_candidates() const { return top_candidates_; }
  void set_nms_thresholds(double dist_m, double angle_deg) {
    nms_dist_thresh_sq_ = dist_m * dist_m;
    nms_angle_thresh_rad_ = angle_deg * M_PI / 180.0;
  }
  AngularInfo get_leaf_angular_info() const { return leaf_ang_info_; }

  // pcd iof
  bool load_voxel_params(const std::string& voxelmaps_folder_path);

  bool set_multi_buckets(const std::string& folder_path);

  bool save_voxelmaps_pcd(const std::string& folder_path);

  bool save_voxel_params(const std::string& folder_path);

  /// 加权体素图二进制缓存：保留每个 bucket 的 .w() 权重（core=2/neighbor=1）
  /// 与空槽哨兵（w=0）。上游 save_voxelmaps_pcd 只存 xyz、读回把 w 一律设为 1，
  /// 会破坏加权 BBS 的分数语义，故另存一份二进制缓存。
  bool save_voxelmaps_cache(const std::string& folder_path,
                            const std::string& source_map_path);

  /// 载入缓存；若源 map.pcd 的 size/mtime_ns 与缓存不符、配置参数不符或文件损坏，
  /// 返回 false，调用方应回退到 set_tar_points() 重建。
  /// expected_min_res/expected_max_level/expected_neighbor_score 为当前配置值，
  /// 用于判定体素图参数是否变化（变了就必须重建）。
  bool load_voxelmaps_cache(const std::string& folder_path,
                            const std::string& source_map_path,
                            double expected_min_res,
                            int expected_max_level,
                            double expected_neighbor_score);

private:
  void calc_angular_info(std::vector<AngularInfo>& ang_info_vec);

  std::vector<DiscreteTransformation<double>> create_init_transset(const AngularInfo& init_ang_info);

  void calc_score(
    DiscreteTransformation<double>& trans,
    const double trans_res,
    const Eigen::Vector3d& rpy_res,
    const Eigen::Vector3d& min_rpy,
    const std::vector<Eigen::Vector4i>& buckets,
    const int max_bucket_scan_count,
    const std::vector<Eigen::Vector3d>& points);

  bool is_spatially_distinct(
    const DiscreteTransformation<double>& a,
    const DiscreteTransformation<double>& b,
    double trans_res,
    const Eigen::Vector3d& rpy_res) const;

private:
  Eigen::Matrix4d global_pose_;
  bool has_timed_out_, has_localized_;
  double elapsed_time_;

  std::vector<Eigen::Vector3d> src_points_;

  std::unique_ptr<VoxelMaps> voxelmaps_ptr_;
  std::string voxelmaps_folder_name_;

  double v_rate_;  // voxel expansion rate
  double inv_v_rate_;
  int num_threads_;
  double neighbor_score_ = 1.0;  // 邻居体素命中分数（核心固定 2.0）

  double best_score_;
  double score_threshold_percentage_;
  bool use_timeout_;
  std::chrono::milliseconds timeout_duration_;
  Eigen::Vector3d min_xyz_;
  Eigen::Vector3d max_xyz_;
  Eigen::Vector3d min_rpy_;
  Eigen::Vector3d max_rpy_;
  std::pair<int, int> init_tx_range_;
  std::pair<int, int> init_ty_range_;
  std::pair<int, int> init_tz_range_;

  // Confidence & multi-peak NMS
  double confidence_score_ = 0.0;
  double peak_margin_ = 0.0;
  std::vector<DiscreteTransformation<double>> top_candidates_;
  double nms_dist_thresh_sq_ = 4.0;      // 2m default
  double nms_angle_thresh_rad_ = 0.5236;  // ~30 degrees default
  AngularInfo leaf_ang_info_;             // stored during localize() for pose reconstruction
};

}  // namespace cpu