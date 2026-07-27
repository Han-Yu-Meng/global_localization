#include <cpu_bbs3d/voxelmaps.hpp>
#include <cpu_bbs3d/bbs3d.hpp>

namespace cpu {
BBS3D::BBS3D()
: v_rate_(2.0),
  num_threads_(4),
  score_threshold_percentage_(0.0),
  use_timeout_(false),
  timeout_duration_(10000),
  has_timed_out_(false),
  has_localized_(false),
  voxelmaps_folder_name_("voxelmaps_coords") {
  inv_v_rate_ = 1.0 / v_rate_;
  min_rpy_ << -0.02, -0.02, 0.0;
  max_rpy_ << 0.02, 0.02, 2 * M_PI;
}

BBS3D::~BBS3D() {}

void BBS3D::set_timeout_duration_in_msec(const int msec) {
  timeout_duration_ = std::chrono::milliseconds(msec);
}

void BBS3D::set_tar_points(const std::vector<Eigen::Vector3d>& points, double min_level_res, int max_level) {
  voxelmaps_ptr_.reset(new VoxelMaps);
  voxelmaps_ptr_->set_min_res(min_level_res);
  voxelmaps_ptr_->set_max_level(max_level);
  voxelmaps_ptr_->create_voxelmaps(points, v_rate_);
}

void BBS3D::set_src_points(const std::vector<Eigen::Vector3d>& points) {
  src_points_.clear();
  src_points_.shrink_to_fit();
  src_points_.resize(points.size());
  std::copy(points.begin(), points.end(), src_points_.begin());
}

void BBS3D::set_trans_search_range(const std::vector<Eigen::Vector3d>& points) {
  // The minimum and maximum x, y, z values are selected from the 3D coordinate vector.
  Eigen::Vector3d min_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector3d max_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());

  for (const auto& point : points) {
    min_xyz = min_xyz.cwiseMin(point);
    max_xyz = max_xyz.cwiseMax(point);
  }

  set_trans_search_range(min_xyz, max_xyz);
}

void BBS3D::set_trans_search_range(const Eigen::Vector3d& min_xyz, const Eigen::Vector3d& max_xyz) {
  min_xyz_ = min_xyz;
  max_xyz_ = max_xyz;

  const int max_level = voxelmaps_ptr_->get_max_level();
  const double top_res = voxelmaps_ptr_->voxelmaps_res_[max_level];
  init_tx_range_ = std::make_pair<int, int>(std::floor(min_xyz.x() / top_res), std::ceil(max_xyz.x() / top_res));
  init_ty_range_ = std::make_pair<int, int>(std::floor(min_xyz.y() / top_res), std::ceil(max_xyz.y() / top_res));
  init_tz_range_ = std::make_pair<int, int>(std::floor(min_xyz.z() / top_res), std::ceil(max_xyz.z() / top_res));
}

void BBS3D::calc_angular_info(std::vector<AngularInfo>& ang_info_vec) {
  double max_norm = src_points_[0].norm();
  for (const auto& point : src_points_) {
    double norm = point.norm();
    if (norm > max_norm) {
      max_norm = norm;
    }
  }

  const int max_level = voxelmaps_ptr_->get_max_level();
  for (int i = max_level; i >= 0; i--) {
    const double cosine = 1 - (std::pow(voxelmaps_ptr_->voxelmaps_res_[i], 2) / std::pow(max_norm, 2)) * 0.5;
    double ori_res = std::acos(std::max(cosine, -1.0));
    ori_res = std::floor(ori_res * 10000) / 10000;
    Eigen::Vector3d rpy_res_temp;
    rpy_res_temp.x() = ori_res <= std::abs(max_rpy_.x() - min_rpy_.x()) ? ori_res : 0.0;
    rpy_res_temp.y() = ori_res <= std::abs(max_rpy_.y() - min_rpy_.y()) ? ori_res : 0.0;
    rpy_res_temp.z() = ori_res <= std::abs(max_rpy_.z() - min_rpy_.z()) ? ori_res : 0.0;

    Eigen::Vector3d max_rpy_piece;
    if (i == max_level) {
      max_rpy_piece = max_rpy_ - min_rpy_;
    } else {
      max_rpy_piece.x() = ang_info_vec[i + 1].rpy_res.x() != 0.0 ? ang_info_vec[i + 1].rpy_res.x() : max_rpy_.x() - min_rpy_.x();
      max_rpy_piece.y() = ang_info_vec[i + 1].rpy_res.y() != 0.0 ? ang_info_vec[i + 1].rpy_res.y() : max_rpy_.y() - min_rpy_.y();
      max_rpy_piece.z() = ang_info_vec[i + 1].rpy_res.z() != 0.0 ? ang_info_vec[i + 1].rpy_res.z() : max_rpy_.z() - min_rpy_.z();
    }

    // Angle division number
    Eigen::Vector3i num_division;
    num_division.x() = rpy_res_temp.x() == 0.0 ? 1 : std::ceil(max_rpy_piece.x() / rpy_res_temp.x());
    num_division.y() = rpy_res_temp.y() == 0.0 ? 1 : std::ceil(max_rpy_piece.y() / rpy_res_temp.y());
    num_division.z() = rpy_res_temp.z() == 0.0 ? 1 : std::ceil(max_rpy_piece.z() / rpy_res_temp.z());
    ang_info_vec[i].num_division = num_division;

    // Bisect an angle
    ang_info_vec[i].rpy_res.x() = num_division.x() == 1 ? 0.0 : max_rpy_piece.x() / num_division.x();
    ang_info_vec[i].rpy_res.y() = num_division.y() == 1 ? 0.0 : max_rpy_piece.y() / num_division.y();
    ang_info_vec[i].rpy_res.z() = num_division.z() == 1 ? 0.0 : max_rpy_piece.z() / num_division.z();

    ang_info_vec[i].min_rpy.x() = ang_info_vec[i].rpy_res.x() == 0.0 ? 0.0 : min_rpy_.x();
    ang_info_vec[i].min_rpy.y() = ang_info_vec[i].rpy_res.y() == 0.0 ? 0.0 : min_rpy_.y();
    ang_info_vec[i].min_rpy.z() = ang_info_vec[i].rpy_res.z() == 0.0 ? 0.0 : min_rpy_.z();
  }
}

std::vector<DiscreteTransformation<double>> BBS3D::create_init_transset(const AngularInfo& init_ang_info) {
  const int init_transset_size = (init_tx_range_.second - init_tx_range_.first + 1) * (init_ty_range_.second - init_ty_range_.first + 1) *
                                 (init_tz_range_.second - init_tz_range_.first + 1) * (init_ang_info.num_division.x()) *
                                 (init_ang_info.num_division.y()) * (init_ang_info.num_division.z());

  const int max_level = voxelmaps_ptr_->get_max_level();

  std::vector<DiscreteTransformation<double>> transset;
  transset.reserve(init_transset_size);
  for (int tx = init_tx_range_.first; tx <= init_tx_range_.second; tx++) {
    for (int ty = init_ty_range_.first; ty <= init_ty_range_.second; ty++) {
      for (int tz = init_tz_range_.first; tz <= init_tz_range_.second; tz++) {
        for (int roll = 0; roll < init_ang_info.num_division.x(); roll++) {
          for (int pitch = 0; pitch < init_ang_info.num_division.y(); pitch++) {
            for (int yaw = 0; yaw < init_ang_info.num_division.z(); yaw++) {
              transset.emplace_back(DiscreteTransformation<double>(0, max_level, tx, ty, tz, roll, pitch, yaw));
            }
          }
        }
      }
    }
  }
  return transset;
}

void BBS3D::calc_score(
  DiscreteTransformation<double>& trans,
  const double trans_res,
  const Eigen::Vector3d& rpy_res,
  const Eigen::Vector3d& min_rpy,
  const std::vector<Eigen::Vector4i>& buckets,
  const int max_bucket_scan_count,
  const std::vector<Eigen::Vector3d>& points) {
  const int num_buckets = buckets.size();
  const double inv_res = 1.0 / trans_res;
  Eigen::Transform<double, 3, Eigen::Affine> transform;
  transform = trans.create_matrix(trans_res, rpy_res, min_rpy);

  for (int i = 0; i < points.size(); i++) {
    const Eigen::Vector3d transed_point = transform * points[i];
    const Eigen::Vector3i coord = (transed_point.array() * inv_res).floor().cast<int>();
    const std::uint32_t hash = (coord[0] * 73856093) ^ (coord[1] * 19349669) ^ (coord[2] * 83492791);

    for (int j = 0; j < max_bucket_scan_count; j++) {
      const std::uint32_t bucket_index = (hash + j) % num_buckets;
      const Eigen::Vector4i& bucket = buckets[bucket_index];

      if (bucket.w() == 0) {
        break;
      }
      
      if (bucket.x() != coord.x() || bucket.y() != coord.y() || bucket.z() != coord.z()) {
        continue;
      }

      trans.score += bucket.w();  // weighted: core=2, neighbor=1
      break;
    }
  }
}

bool BBS3D::is_spatially_distinct(
  const DiscreteTransformation<double>& a,
  const DiscreteTransformation<double>& b,
  double trans_res,
  const Eigen::Vector3d& rpy_res) const {
  // Translation distance squared
  double dx = (a.x - b.x) * trans_res;
  double dy = (a.y - b.y) * trans_res;
  double dz = (a.z - b.z) * trans_res;
  double dist_sq = dx * dx + dy * dy + dz * dz;
  if (dist_sq > nms_dist_thresh_sq_) return true;

  // Angular difference (primary check on yaw for planar robot)
  double droll = std::abs(a.roll - b.roll) * rpy_res.x();
  double dpitch = std::abs(a.pitch - b.pitch) * rpy_res.y();
  double dyaw = std::abs(a.yaw - b.yaw) * rpy_res.z();
  if (droll > nms_angle_thresh_rad_ || dpitch > nms_angle_thresh_rad_ || dyaw > nms_angle_thresh_rad_)
    return true;

  return false;
}

void BBS3D::localize() {
  // Calc localize time limit
  const auto start_time = std::chrono::system_clock::now();
  const auto time_limit = start_time + timeout_duration_;
  has_timed_out_ = false;

  best_score_ = 0;
  const int score_threshold = std::floor(src_points_.size() * score_threshold_percentage_);
  int best_score = score_threshold;
  DiscreteTransformation<double> best_trans(best_score);

  // Clear multi-peak candidates and confidence from previous run
  top_candidates_.clear();
  confidence_score_ = 0.0;
  peak_margin_ = 0.0;
  const int max_candidates = 3;

  // Preapre initial transset
  const int max_bucket_scan_count = voxelmaps_ptr_->get_max_bucket_scan_count();
  const int max_level = voxelmaps_ptr_->get_max_level();
  std::vector<AngularInfo> ang_info_vec(max_level + 1);
  calc_angular_info(ang_info_vec);
  leaf_ang_info_ = ang_info_vec[0];  // Store for external pose reconstruction
  auto init_transset = create_init_transset(ang_info_vec[max_level]);

  // Calc initial transset scores
  const auto& top_buckets = voxelmaps_ptr_->multi_buckets_[max_level];
  double init_trans_res = voxelmaps_ptr_->voxelmaps_res_[max_level];
  const Eigen::Vector3d& rpy_res = ang_info_vec[max_level].rpy_res;
  const Eigen::Vector3d& min_rpy = ang_info_vec[max_level].min_rpy;
#pragma omp parallel for num_threads(num_threads_)
  for (int i = 0; i < init_transset.size(); i++) {
    calc_score(init_transset[i], init_trans_res, rpy_res, min_rpy, top_buckets, max_bucket_scan_count, src_points_);
  }

  std::priority_queue<DiscreteTransformation<double>> trans_queue(init_transset.begin(), init_transset.end());

  while (!trans_queue.empty()) {
    if (use_timeout_ && std::chrono::system_clock::now() > time_limit) {
      has_timed_out_ = true;
      break;
    }

    auto trans = trans_queue.top();
    trans_queue.pop();

    // pruning
    if (trans.score < best_score) {
      continue;
    }

    if (trans.is_leaf()) {
      // Multi-peak NMS: collect spatially distinct leaf candidates
      if (top_candidates_.empty()) {
        // First candidate is the best (highest-scoring leaf processed first)
        best_trans = trans;
        best_score = trans.score;
        top_candidates_.push_back(trans);
      } else {
        // Check spatial distinctness against already-collected candidates
        double min_res = voxelmaps_ptr_->get_min_res();
        bool is_distinct = true;
        for (const auto& cand : top_candidates_) {
          if (!is_spatially_distinct(trans, cand, min_res, ang_info_vec[0].rpy_res)) {
            is_distinct = false;
            break;
          }
        }
        if (is_distinct) {
          top_candidates_.push_back(trans);
        }
      }

      // Early exit: collected enough spatially distinct candidates
      if (top_candidates_.size() >= max_candidates) {
        break;
      }
    } else {
      const int child_level = trans.level - 1;
      const Eigen::Vector3i& num_division = ang_info_vec[child_level].num_division;
      const Eigen::Vector3d& rpy_res = ang_info_vec[child_level].rpy_res;
      const Eigen::Vector3d& min_rpy = ang_info_vec[child_level].min_rpy;
      auto children = trans.branch(child_level, static_cast<int>(v_rate_), num_division);

      const auto& buckets = voxelmaps_ptr_->multi_buckets_[child_level];
      double trans_res = voxelmaps_ptr_->voxelmaps_res_[child_level];

#pragma omp parallel for num_threads(num_threads_)
      for (int i = 0; i < children.size(); i++) {
        calc_score(children[i], trans_res, rpy_res, min_rpy, buckets, max_bucket_scan_count, src_points_);
      }

      for (const auto& child : children) {
        // pruning
        if (child.score < best_score) {
          continue;
        }

        trans_queue.push(child);
      }
    }
  }

  // Calc localization elapsed time
  const auto end_time = std::chrono::system_clock::now();
  elapsed_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1e6;

  // Confidence evaluation
  if (!top_candidates_.empty()) {
    best_trans = top_candidates_[0];
    best_score = best_trans.score;

    double score_1st = static_cast<double>(best_score);
    double absolute_score_ratio = (src_points_.size() > 0) ? score_1st / (2.0 * src_points_.size()) : 0.0;

    if (top_candidates_.size() > 1) {
      double score_2nd = static_cast<double>(top_candidates_[1].score);
      // Margin: 1.0 = highly unimodal, 0.0 = equally strong competing peaks
      peak_margin_ = (score_1st - score_2nd) / score_1st;
    } else {
      peak_margin_ = 1.0;  // No competing peak found
    }

    // Confidence: blend of peak separation and absolute match quality
    confidence_score_ = 0.6 * peak_margin_ + 0.4 * absolute_score_ratio;
  }

  // Not localized
  if (best_score == score_threshold || has_timed_out_ || top_candidates_.empty()) {
    has_localized_ = false;
    return;
  }

  double min_res = voxelmaps_ptr_->get_min_res();
  global_pose_ = best_trans.create_matrix(min_res, ang_info_vec[0].rpy_res, ang_info_vec[0].min_rpy);
  best_score_ = best_score;
  has_timed_out_ = false;
  has_localized_ = true;
}

}  // namespace cpu
