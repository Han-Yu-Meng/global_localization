#include <cpu_bbs3d/voxelmaps.hpp>
#include <algorithm>
#include <cmath>

namespace cpu {

VoxelMaps::VoxelMaps() : min_level_res_(1.0), max_level_(6), max_bucket_scan_count_(10) {}

VoxelMaps::~VoxelMaps() {}

void VoxelMaps::create_voxelmaps(const std::vector<Eigen::Vector3d>& points, const int v_rate) {
  const int multi_buckets_size = max_level_ + 1;
  multi_buckets_.resize(multi_buckets_size);
  voxelmaps_res_.resize(multi_buckets_size);

  // --- Level 0: build weighted voxel map from points ---
  double resolution = min_level_res_;
  voxelmaps_res_[0] = resolution;

  UnorderedVoxelMap weighted_map;
  std::vector<Eigen::Vector3i> coords(points.size());
  std::transform(points.begin(), points.end(), coords.begin(), [&](const Eigen::Vector3d& point) {
    return (point.array() / resolution).floor().cast<int>();
  });

  // 1 = core voxel (score 2.0), -1 = dilated neighbor (score = neighbor_score_, 默认 1.0)
  for (const auto& coord : coords) {
    if (weighted_map.count(coord) == 0) {
      weighted_map[coord] = 1;  // core
    } else if (weighted_map[coord] == -1) {
      weighted_map[coord] = 1;  // neighbor → promoted to core
    } else if (weighted_map[coord] == 1) {
      continue;  // already core
    }

    const auto neighbor_coords = create_neighbor_coords(coord);
    for (const auto& neighbor_coord : neighbor_coords) {
      if (weighted_map.count(neighbor_coord) == 0) {
        weighted_map[neighbor_coord] = -1;  // neighbor
      }
    }
  }
  multi_buckets_[0] = create_hash_buckets_weighted(weighted_map);

  // --- Levels 1..max_level: Max-Pooling from finer level ---
  for (int i = 1; i < multi_buckets_size; i++) {
    resolution = resolution * v_rate;
    voxelmaps_res_[i] = resolution;
    weighted_map = max_pool_voxel_map(weighted_map, v_rate);
    multi_buckets_[i] = create_hash_buckets_weighted(weighted_map);
  }
}

VoxelMaps::UnorderedVoxelMap VoxelMaps::max_pool_voxel_map(
    const UnorderedVoxelMap& fine_map, int v_rate) {
  // Pool fine-level (x,y,z,weight) → coarse-level via max
  // parent_coord = floor(child_coord / v_rate)
  UnorderedVoxelMap coarse_map;
  for (const auto& [child_coord, child_weight] : fine_map) {
    // child_weight: 1 (core) or -1 (neighbor，具体分数由 neighbor_score_ 决定)
    // Max-pool: take max weight among all children mapping to same parent
    Eigen::Vector3i parent_coord(
      static_cast<int>(std::floor(static_cast<double>(child_coord.x()) / v_rate)),
      static_cast<int>(std::floor(static_cast<double>(child_coord.y()) / v_rate)),
      static_cast<int>(std::floor(static_cast<double>(child_coord.z()) / v_rate))
    );

    auto it = coarse_map.find(parent_coord);
    if (it == coarse_map.end()) {
      coarse_map[parent_coord] = child_weight;
    } else if (child_weight > it->second) {
      // Max-pooling: parent takes the maximum weight among children
      it->second = child_weight;
    }
  }
  return coarse_map;
}

VoxelMaps::Buckets VoxelMaps::create_hash_buckets_weighted(const UnorderedVoxelMap& unordered_voxelmap) {
  // Same as create_hash_buckets but stores quantized score in .w():
  //   core (1)      → .w() = 2                (core voxel always scores 2.0)
  //   neighbor (-1) → .w() = round(neighbor_score_)
  //                   neighbor_score_ = 1.0 → 加权 BBS（邻居半权）
  //                   neighbor_score_ = 2.0 → 原始 3D-BBS（命中即满分，无加权惩罚）
  //   empty/absent  → .w() = 0                (empty sentinel, unchanged)
  // 夹到 [1, 2]：.w() == 0 是空槽哨兵值，绝不能让邻居分数取到 0
  const int neighbor_w = std::clamp(static_cast<int>(std::lround(neighbor_score_)), 1, 2);
  std::vector<Eigen::Vector4i> buckets;
  for (int num_buckets = unordered_voxelmap.size(); num_buckets <= unordered_voxelmap.size() * 16; num_buckets *= 2) {
    buckets.resize(num_buckets);
    std::fill(buckets.begin(), buckets.end(), Eigen::Vector4i::Zero());

    int success_count = 0;
    for (const auto& voxel : unordered_voxelmap) {
      int weight = (voxel.second == 1) ? 2 : neighbor_w;  // core=2, neighbor=neighbor_score_
      Eigen::Vector4i coord;
      coord << voxel.first.x(), voxel.first.y(), voxel.first.z(), weight;
      const std::uint32_t hash = (coord[0] * 73856093) ^ (coord[1] * 19349669) ^ (coord[2] * 83492791);

      for (int j = 0; j < max_bucket_scan_count_; j++) {
        const std::uint32_t bucket_index = (hash + j) % num_buckets;
        if (buckets[bucket_index].w() == 0) {
          buckets[bucket_index] = coord;
          success_count++;
          break;
        }
      }
    }

    const double success_rate = static_cast<double>(success_count) / unordered_voxelmap.size();
    if (success_rate > 0.999) {
      break;
    }
  }

  return buckets;
}

std::vector<Eigen::Vector3i> VoxelMaps::create_neighbor_coords(const Eigen::Vector3i& vec) {
  std::vector<Eigen::Vector3i> neighbor_coord;
  neighbor_coord.reserve(7);
  neighbor_coord.emplace_back(vec.x() - 1, vec.y() - 1, vec.z());
  neighbor_coord.emplace_back(vec.x() - 1, vec.y(), vec.z());
  neighbor_coord.emplace_back(vec.x(), vec.y() - 1, vec.z());
  neighbor_coord.emplace_back(vec.x() - 1, vec.y() - 1, vec.z() - 1);
  neighbor_coord.emplace_back(vec.x() - 1, vec.y(), vec.z() - 1);
  neighbor_coord.emplace_back(vec.x(), vec.y() - 1, vec.z() - 1);
  neighbor_coord.emplace_back(vec.x(), vec.y(), vec.z() - 1);
  return neighbor_coord;
}

VoxelMaps::Buckets VoxelMaps::create_hash_buckets(const UnorderedVoxelMap& unordered_voxelmap) {
  // Loop until the success rate exceeds the threshold
  std::vector<Eigen::Vector4i> buckets;
  for (int num_buckets = unordered_voxelmap.size(); num_buckets <= unordered_voxelmap.size() * 16; num_buckets *= 2) {
    buckets.resize(num_buckets);
    std::fill(buckets.begin(), buckets.end(), Eigen::Vector4i::Zero());

    int success_count = 0;
    for (const auto& voxel : unordered_voxelmap) {
      Eigen::Vector4i coord;
      coord << voxel.first.x(), voxel.first.y(), voxel.first.z(), 1;
      const std::uint32_t hash = (coord[0] * 73856093) ^ (coord[1] * 19349669) ^ (coord[2] * 83492791);

      // Find empty bucket
      for (int i = 0; i < max_bucket_scan_count_; i++) {
        const std::uint32_t bucket_index = (hash + i) % num_buckets;

        if (buckets[bucket_index].w() == 0) {
          buckets[bucket_index] = coord;
          success_count++;
          break;
        }
      }
    }

    const double success_rate = static_cast<double>(success_count) / unordered_voxelmap.size();
    if (success_rate > 0.999) {
      break;
    }
  }

  return buckets;
}
}  // namespace cpu
