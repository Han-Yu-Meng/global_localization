#pragma once

#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/functional/hash/hash.hpp>

namespace cpu {
class VoxelMaps {
public:
  VoxelMaps();
  ~VoxelMaps();

  struct VectorHash {
    size_t operator()(const Eigen::Vector3i& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, x[0]);
      boost::hash_combine(seed, x[1]);
      boost::hash_combine(seed, x[2]);
      return seed;
    }
  };

  struct VctorEqual {
    bool operator()(const Eigen::Vector3i& v1, const Eigen::Vector3i& v2) const { return v1 == v2; }
  };

  using UnorderedVoxelMap = std::unordered_map<Eigen::Vector3i, int, VectorHash, VctorEqual>;
  using Buckets = std::vector<Eigen::Vector4i>;

  void set_min_res(double min_level_res) { min_level_res_ = min_level_res; }

  void set_max_level(int max_level) { max_level_ = max_level; }

  void set_max_bucket_scan_count(int max_bucket_scan_count) { max_bucket_scan_count_ = max_bucket_scan_count; }

  /// 邻居体素命中分数（核心体素固定 2.0）：1.0 = 加权 BBS，2.0 = 原始 3D-BBS
  /// 必须在 create_voxelmaps() 之前设置，权重在建图时写进 buckets 的 .w()
  void set_neighbor_score(double neighbor_score) { neighbor_score_ = neighbor_score; }

  double get_min_res() const { return min_level_res_; }

  int get_max_level() const { return max_level_; }

  int get_max_bucket_scan_count() const { return max_bucket_scan_count_; }

  double get_neighbor_score() const { return neighbor_score_; }

  void create_voxelmaps(const std::vector<Eigen::Vector3d>& points, const int v_rate);

private:
  std::vector<Eigen::Vector3i> create_neighbor_coords(const Eigen::Vector3i& vec);

  Buckets create_hash_buckets(const UnorderedVoxelMap& unordered_voxelmap);

  Buckets create_hash_buckets_weighted(const UnorderedVoxelMap& unordered_voxelmap);

  UnorderedVoxelMap max_pool_voxel_map(const UnorderedVoxelMap& fine_map, int v_rate);

public:
  std::vector<Buckets> multi_buckets_;

  std::vector<double> voxelmaps_res_;

private:
  double min_level_res_;
  int max_level_, max_bucket_scan_count_;
  double neighbor_score_ = 1.0;  // 邻居体素命中分数（核心固定 2.0）
};
}  // namespace cpu