/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
  #include <tf2_eigen/tf2_eigen.hpp>
#else
  #include <tf2_eigen/tf2_eigen.h>
#endif

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

// Small GICP
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

// 3dBBS
#include <cpu_bbs3d/bbs3d.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <thread>
#include <std_msgs/msg/string.hpp>
#include <map>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <array>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <utility>

namespace fs = std::filesystem;

enum class SystemState { IDLE, COARSE_LOCALIZATION, FINE_LOCALIZATION, UNKNOWN_AREA };

class GlobalLocalizationNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        bool initialization_only = false;
        bool auto_search_range = true;
        Eigen::Vector3d min_search_xyz{-100.0, -100.0, -0.10};
        Eigen::Vector3d max_search_xyz{100.0, 100.0, 0.10};
        Eigen::Vector3d min_search_rpy{0.0, 0.0, -M_PI};
        Eigen::Vector3d max_search_rpy{0.0, 0.0, M_PI};
        
        int num_threads = 8;
        double score_threshold = 0.3;
        double accumulation_time = 1.0;
        double tracking_interval = 1.0;

        double min_level_res = 1.5;
        int max_level = 5;
        double global_leaf_size = 0.8;
        double registered_leaf_size = 0.5;
        double bbs_leaf_size = 0.5;
        double viz_leaf_size = 0.7;
        
        double max_dist_sq_fine = 4.0;
        double max_dist_sq_bbs_refine = 25.0;
        double max_gicp_error = 60000.0;
        int max_gicp_failures = 6000;
        std::string voxel_dir = "";

        // Smoothing filter for T_map_odom to prevent localization_pose jumps
        bool smooth_enabled = false;
        double smooth_alpha = 0.3;  // 0.0 = frozen, 1.0 = no smoothing

        // Pose publish trigger: "tf" = every odom callback, "gicp" = only when GICP updates T_map_odom
        std::string pose_trigger = "tf";

        // Planar constraint: force roll, pitch, z = 0 for map->odom and localization_pose
        bool planar_constraint = false;

        // NMS and confidence evaluation thresholds
        double nms_dist_thresh = 2.0;       // meters
        double nms_angle_thresh_deg = 30.0;  // degrees

        // 邻居体素命中分数（核心体素固定 2.0）：1.0 = 加权 BBS，2.0 = 原始 3D-BBS
        double neighbor_score = 1.0;

        // 单轮 3dBBS 搜索时限（毫秒），<=0 不限。无匹配（陌生区域）时 BBS 最坏情况
        // 会把整个搜索空间翻完（数百个候选逐层扩展、每个都要对全源点云打分），可达数分钟；
        // 设上限后超时按「未匹配」计入失败次数，配合 max_init_attempts 快速判负。
        int bbs_timeout_msec = 30000;

        // ---- 接受门控：BBS+GICP 结果必须真的落在地图上（initialization_only）----
        // res.error 是 covariance 加权残差和、不是米制量：错位候选只靠少量远距对应点
        // 就能把它拉低（实测错位 accept 的 error 比正确 accept 低一个数量级），
        // 因此接受前必须独立复核「变换后的源点在地图上的最近邻命中率」。
        int    min_init_inliers = 100;        // GICP 有效对应点数下限
        double min_init_inlier_ratio = 0.25;  // 有效对应点 / 源点数 下限
        double init_verify_max_dist = 0.5;    // 落图复核的最近邻距离门限 (m)
        double min_init_overlap_ratio = 0.30; // 落图复核命中率下限

        // ---- 初始化尝试上限与陌生区域处理（initialization_only）----
        int         max_init_attempts = 3;         // 连续失败多少次判定陌生区域
        std::string unknown_area_action = "stop";  // "stop" = 停止匹配驻留；"exit" = 退出进程
        double      unknown_area_retry_sec = 0.0;  // >0 陌生区域隔多久重试一次；0 = 不再尝试
    };

    struct JobData {
        bool has_new_job = false;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_odom; 
        fins::AcqTime ts;
        Eigen::Isometry3d T_odom_base;
    };

    void define() override {
        set_name("GlobalLocalization");
        set_category("Navigation>Localization");
        register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &GlobalLocalizationNode::on_cloud_callback);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{odom}^{baselink}$", &GlobalLocalizationNode::on_odom_tf_callback);
        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("global_map_viz");
        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("aligned_cloud");
        register_output<geometry_msgs::msg::TransformStamped>("$T_{map}^{odom}$");
        register_output<geometry_msgs::msg::PoseWithCovarianceStamped>("current_pose");
        register_output<geometry_msgs::msg::PoseWithCovarianceStamped>("initial_pose");
        register_input<std_msgs::msg::String>("tracking_status", &GlobalLocalizationNode::on_tracking_status);
        register_output<geometry_msgs::msg::TransformStamped>("initial_alignment");
        register_output<visualization_msgs::msg::MarkerArray>("bbs_candidates");
        register_output<std_msgs::msg::String>("initializer_status");
    }

    void initialize() override {
        fins::ParamLoader bbs("GlobalLocalization.bbs");
        fins::ParamLoader gicp("GlobalLocalization.gicp_tracking");
        fins::ParamLoader init_cfg("GlobalLocalization.initializer");

        config_.initialization_only = fins::param_server().get("GlobalLocalization.initialization_only", false);
        // ---------------- bbs ----------------
        config_.num_threads = bbs.get("num_threads", 8)
                                   .with_description("Number of threads for parallel processing (GICP covariance and 3dBBS search)")
                                   .within(1, 128);

        config_.score_threshold = bbs.get("score_threshold", 0.3)
                                      .with_description("Minimum score percentage threshold for 3dBBS coarse localization")
                                      .within(0.0, 10.0);

        config_.bbs_timeout_msec = bbs.get("timeout_msec", 30000)
                                       .with_description("Per-search 3dBBS time limit in milliseconds (<=0 disables); a timed-out search counts as no-match")
                                       .greater_than(-1);

        config_.neighbor_score = 2.0;

        config_.accumulation_time = bbs.get("accumulation_time", 1.0)
                                        .with_description("Time duration to accumulate input point clouds before triggering localization (seconds)")
                                        .greater_than(0.0);

        config_.min_level_res = bbs.get("min_level_res", 1.5)
                                    .with_description("Resolution of the lowest level in the 3dBBS multi-resolution pyramid (meters)")
                                    .greater_than(0.0);

        config_.max_level = bbs.get("max_level", 5)
                                .with_description("Maximum number of levels in the 3dBBS multi-resolution pyramid");

        config_.global_leaf_size = bbs.get("global_leaf_size", 0.5)
                                       .with_description("Voxel leaf size for the target global map points (meters)")
                                       .greater_than(0.0);

        config_.bbs_leaf_size = bbs.get("bbs_leaf_size", 0.5)
                                    .with_description("Voxel leaf size for source clouds during 3dBBS coarse localization (meters)")
                                    .within(0.05, 5.0);

        config_.viz_leaf_size = bbs.get("viz_leaf_size", 0.7)
                                    .with_description("Voxel leaf size for visualization (meters)")
                                    .greater_than(0.0);

        config_.nms_dist_thresh = bbs.get("nms_dist_thresh", 2.0)
                                      .with_description("Minimum distance (m) between distinct BBS candidate peaks for NMS")
                                      .greater_than(0.0);

        config_.nms_angle_thresh_deg = bbs.get("nms_angle_thresh_deg", 30.0)
                                            .with_description("Minimum yaw difference (deg) between distinct BBS candidate peaks for NMS")
                                            .within(0.0, 180.0);

        config_.voxel_dir = bbs.get("voxel_dir", std::string(""))
                                .with_description("Directory path for storing preprocessed voxel maps");

        config_.auto_search_range = bbs.get("auto_search_range", true)
                                        .with_description("If true, derive the x/y search range from the map bounding box "
                                                          "(z keeps the configured range); otherwise use bbs.search_range");

        // ---------------- bbs.search_range ----------------
        {
            fins::ParamLoader search("GlobalLocalization.bbs.search_range");
            config_.min_search_xyz.x() = search.get("x_min", -100.0)
                                             .with_description("Minimum x of the 3dBBS translation search range (map frame, meters)");
            config_.max_search_xyz.x() = search.get("x_max", 100.0)
                                             .with_description("Maximum x of the 3dBBS translation search range (map frame, meters)");
            config_.min_search_xyz.y() = search.get("y_min", -100.0)
                                             .with_description("Minimum y of the 3dBBS translation search range (map frame, meters)");
            config_.max_search_xyz.y() = search.get("y_max", 100.0)
                                             .with_description("Maximum y of the 3dBBS translation search range (map frame, meters)");
            config_.min_search_xyz.z() = search.get("z_min", -0.10)
                                             .with_description("Minimum z of the 3dBBS translation search range (map frame, meters)");
            config_.max_search_xyz.z() = search.get("z_max", 0.10)
                                             .with_description("Maximum z of the 3dBBS translation search range (map frame, meters)");
        }

        // ---------------- bbs.angle_range ----------------
        {
            fins::ParamLoader angle("GlobalLocalization.bbs.angle_range");
            config_.min_search_rpy.x() = angle.get("roll_min", 0.0)
                                              .with_description("Minimum roll of the 3dBBS angular search range (radians)");
            config_.max_search_rpy.x() = angle.get("roll_max", 0.0)
                                              .with_description("Maximum roll of the 3dBBS angular search range (radians)");
            config_.min_search_rpy.y() = angle.get("pitch_min", 0.0)
                                              .with_description("Minimum pitch of the 3dBBS angular search range (radians)");
            config_.max_search_rpy.y() = angle.get("pitch_max", 0.0)
                                              .with_description("Maximum pitch of the 3dBBS angular search range (radians)");
            config_.min_search_rpy.z() = angle.get("yaw_min", -M_PI)
                                              .with_description("Minimum yaw of the 3dBBS angular search range (radians)");
            config_.max_search_rpy.z() = angle.get("yaw_max", M_PI)
                                              .with_description("Maximum yaw of the 3dBBS angular search range (radians)");
        }

        normalize_search_range();

        // ---------------- gicp_tracking ----------------
        config_.registered_leaf_size = gicp.get("registered_leaf_size", 0.5)
                                            .with_description("Voxel leaf size for source clouds during GICP registration (meters)")
                                            .greater_than(0.0);

        config_.tracking_interval = gicp.get("tracking_interval", 1.0)
                                         .with_description("Time interval between GICP tracking updates (seconds)")
                                         .greater_than(0.0);

        config_.max_dist_sq_fine = gicp.get("max_dist_sq_fine", 4.0)
                                       .with_description("Maximum squared distance for GICP point correspondences (meters^2)")
                                       .greater_than(0.0);

        config_.max_dist_sq_bbs_refine = gicp.get("max_dist_sq_bbs_refine", 25.0)
                                            .with_description("Maximum squared distance for GICP refinement after 3dBBS coarse localization (meters^2)")
                                            .greater_than(0.0);

        config_.max_gicp_error = gicp.get("max_gicp_error", 60000.0)
                                     .with_description("Threshold for GICP convergence error; results exceeding this are treated as failure")
                                     .greater_than(0.0);

        config_.max_gicp_failures = gicp.get("max_gicp_failures", 6000)
                                        .with_description("Number of consecutive GICP failures before falling back to coarse localization")
                                        .greater_than(0);

        config_.smooth_enabled = gicp.get("smooth_enabled", false)
                                     .with_description("Enable EMA smoothing filter on T_map_odom to prevent localization_pose jumps");

        config_.smooth_alpha = gicp.get("smooth_alpha", 0.3)
                                    .with_description("Smoothing factor for EMA filter (0.0 = frozen, 1.0 = no smoothing). Lower = smoother but more lag.")
                                    .within(0.0, 1.0);

        config_.pose_trigger = gicp.get("pose_trigger", std::string("tf"))
                                   .with_description("Pose publish trigger: 'tf' = every odom callback, 'gicp' = only on GICP update");

        config_.planar_constraint = gicp.get("planar_constraint", false)
                                        .with_description("If true, force roll=0, pitch=0, z=0 on map->odom transform and current_pose output");

        // ---------------- initializer（接受门控 + 陌生区域处理） ----------------
        config_.min_init_inliers = init_cfg.get("min_init_inliers", 100)
                                       .with_description("Minimum GICP inlier count for accepting a BBS+GICP initialization")
                                       .greater_than(0);
        config_.min_init_inlier_ratio = init_cfg.get("min_init_inlier_ratio", 0.25)
                                            .with_description("Minimum GICP inlier count / source size ratio for acceptance")
                                            .within(0.0, 1.0);
        config_.init_verify_max_dist = init_cfg.get("init_verify_max_dist", 0.5)
                                           .with_description("NN distance gate (m) for the independent map-overlap verification")
                                           .greater_than(0.0);
        config_.min_init_overlap_ratio = init_cfg.get("min_init_overlap_ratio", 0.30)
                                             .with_description("Minimum fraction of transformed source points within init_verify_max_dist of the map")
                                             .within(0.0, 1.0);
        config_.max_init_attempts = init_cfg.get("max_init_attempts", 3)
                                        .with_description("Consecutive failed initialization attempts before declaring unknown area")
                                        .greater_than(0);
        config_.unknown_area_action = init_cfg.get("unknown_area_action", std::string("stop"))
                                          .with_description("On unknown area: 'stop' = stop matching and idle, 'exit' = terminate the process");
        config_.unknown_area_retry_sec = init_cfg.get("unknown_area_retry_sec", 0.0)
                                             .with_description("Retry matching after this many seconds in unknown area (0 = never)")
                                             .greater_than(-0.001);

        bbs3d_ = std::make_unique<cpu::BBS3D>();
        if (config_.bbs_timeout_msec > 0) {
            bbs3d_->enable_timeout();
            bbs3d_->set_timeout_duration_in_msec(config_.bbs_timeout_msec);
        } else {
            bbs3d_->disable_timeout();
        }
        accumulated_cloud_odom_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        
        std::string map_path = bbs.get("map_dir", std::string(""))
                                   .with_description("Directory to load map file");
        
        if (!map_path.empty()) load_map(map_path);

        is_worker_running_ = true;
        worker_thread_ = std::thread(&GlobalLocalizationNode::worker_loop, this);
    }

    void run() override {
        is_paused_ = false;
        cv_.notify_one();
    }
    void pause() override {
        is_paused_ = true;
    }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(data_mtx_);
        state_ = map_ready_ ? SystemState::COARSE_LOCALIZATION : SystemState::IDLE;
        accumulated_cloud_odom_->clear();
        accumulation_started_ = false;
        consecutive_gicp_failures_ = 0;
        has_filtered_pose_ = false;
        init_failures_ = 0;
        init_accepted_pending_ = false;
    }

    ~GlobalLocalizationNode() {
        is_worker_running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    /// 保证 min <= max（配置写反时自动交换并告警）
    void normalize_search_range() {
        auto fix = [this](double& lo, double& hi, const char* name) {
            if (lo > hi) {
                logger->warn("[Config] {}_min ({}) > {}_max ({}), swapping the two values.", name, lo, name, hi);
                std::swap(lo, hi);
            }
        };
        fix(config_.min_search_xyz.x(), config_.max_search_xyz.x(), "x");
        fix(config_.min_search_xyz.y(), config_.max_search_xyz.y(), "y");
        fix(config_.min_search_xyz.z(), config_.max_search_xyz.z(), "z");
        fix(config_.min_search_rpy.x(), config_.max_search_rpy.x(), "roll");
        fix(config_.min_search_rpy.y(), config_.max_search_rpy.y(), "pitch");
        fix(config_.min_search_rpy.z(), config_.max_search_rpy.z(), "yaw");
    }

    // ---- 落图复核：源点按候选位姿变换到地图系后，最近邻距离门限内的命中率 ----
    struct OverlapStat { size_t hit = 0; size_t total = 0; double ratio = 0.0; };

    OverlapStat verify_overlap(const pcl::PointCloud<pcl::PointCovariance>::Ptr& src_base,
                               const Eigen::Isometry3d& T_map_base) const {
        OverlapStat out;
        if (!target_tree_ || !target_covs_ || !src_base || src_base->empty()) return out;
        const double max_sq = config_.init_verify_max_dist * config_.init_verify_max_dist;
        const Eigen::Matrix3d R = T_map_base.linear();
        const Eigen::Vector3d t = T_map_base.translation();
        for (const auto& p : src_base->points) {
            const Eigen::Vector3d w = R * p.getVector3fMap().cast<double>() + t;
            size_t idx = 0;
            double sq = std::numeric_limits<double>::max();
            // small_gicp 的 KdTree 距离是 4 维 squaredNorm；地图点云由 load_map
            // 只写 xyz 构建，第 4 维为 0。查询第 4 维也必须填 0，否则即使点完全
            // 重合也会恒定多出 1.0，导致命中率恒为 0（真匹配一并被误杀）。
            size_t found = target_tree_->nearest_neighbor_search(
                Eigen::Vector4d(w.x(), w.y(), w.z(), 0.0), &idx, &sq);
            ++out.total;
            if (!found || idx >= target_covs_->size()) continue;
            // 不依赖第 4 维约定：用返回的最近点索引自己按 3 维重算距离。
            // 第 4 维对所有地图点是同一常数，不影响 argmin，故 idx 就是 3 维最近点。
            const Eigen::Vector3d mp = target_covs_->points[idx].getVector3fMap().cast<double>();
            if ((mp - w).squaredNorm() <= max_sq) ++out.hit;
        }
        out.ratio = out.total ? double(out.hit) / double(out.total) : 0.0;
        return out;
    }

    // ---- 初始化失败计数：连续 max_init_attempts 次 -> 判定陌生区域 ----
    void note_init_failure(const std::string& why) {
        if (!config_.initialization_only) return;
        const int n = ++init_failures_;
        logger->warn("[Initializer] 初始化尝试失败 #{}（{}），上限 {} 次",
                     n, why, config_.max_init_attempts);
        if (n >= config_.max_init_attempts) enter_unknown_area(why);
    }

    void enter_unknown_area(const std::string& why) {
        state_ = SystemState::UNKNOWN_AREA;
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            accumulated_cloud_odom_->clear();
            accumulation_started_ = false;
            job_.has_new_job = false;
        }
        unknown_area_since_ = std::chrono::steady_clock::now();
        logger->warn("[Initializer] 连续 {} 次初始化失败（{}），判定为陌生区域：停止匹配（action={}）",
                     init_failures_.load(), why, config_.unknown_area_action);
        publish_initializer_status("UNKNOWN_AREA", fins::now());
        if (config_.unknown_area_action == "exit") {
            logger->warn("[Initializer] unknown_area_action=exit，退出定位进程"
                         "（注意：localization.py 看门狗会重启它，如需彻底停用请一并处理）");
            std::exit(0);
        }
    }

    void publish_initializer_status(const std::string& s, const fins::AcqTime& ts) {
        if (!required("initializer_status")) return;
        std_msgs::msg::String msg;
        msg.data = s;
        send("initializer_status", msg, ts);
    }

    void load_map(const std::string& path) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(path, *raw_map) == -1) {
            logger->error("Failed to load map file: {}", path);
            return;
        }

        map_viz_cloud_ = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*raw_map, config_.viz_leaf_size);
        auto map_algo = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*map_viz_cloud_, config_.global_leaf_size);
        
        target_covs_.reset(new pcl::PointCloud<pcl::PointCovariance>());
        target_covs_->points.resize(map_algo->size());
        for (size_t i = 0; i < map_algo->size(); ++i) {
            target_covs_->points[i].getVector3fMap() = map_algo->points[i].getVector3fMap();
        }
        small_gicp::estimate_covariances_omp(*target_covs_, 20, config_.num_threads);
        target_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target_covs_, small_gicp::KdTreeBuilderOMP(config_.num_threads));

        std::vector<Eigen::Vector3d> tar_points;
        for (auto& p : map_algo->points) tar_points.push_back(p.getVector3fMap().cast<double>());
        // 邻居体素分数在建体素图时固化，必须在 set_tar_points() 之前设置
        bbs3d_->set_neighbor_score(config_.neighbor_score);
        bbs3d_->set_tar_points(tar_points, config_.min_level_res, config_.max_level);
        bbs3d_->set_score_threshold_percentage(config_.score_threshold);
        bbs3d_->set_nms_thresholds(config_.nms_dist_thresh, config_.nms_angle_thresh_deg);
        bbs3d_->set_num_threads(config_.num_threads);

        // 搜索范围只与地图和静态配置有关，加载地图后设置一次即可
        Eigen::Vector3d min_xyz = config_.min_search_xyz;
        Eigen::Vector3d max_xyz = config_.max_search_xyz;
        if (config_.auto_search_range) {
            Eigen::Vector3d bb_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
            Eigen::Vector3d bb_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
            for (const auto& p : raw_map->points) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
                const Eigen::Vector3d v(p.x, p.y, p.z);
                bb_min = bb_min.cwiseMin(v);
                bb_max = bb_max.cwiseMax(v);
            }
            if (bb_min.allFinite() && bb_max.allFinite() && (bb_max - bb_min).norm() > 1e-6) {
                min_xyz.x() = bb_min.x();
                max_xyz.x() = bb_max.x();
                min_xyz.y() = bb_min.y();
                max_xyz.y() = bb_max.y();
                // z 依旧使用配置值（默认 -0.10 ~ 0.10），地面机器人的高度搜索不需要地面以下/以上的空间
            } else {
                logger->warn("[3dBBS] auto_search_range is on but the map bounding box is invalid, "
                             "falling back to the configured search_range.");
            }
        }
        bbs3d_->set_trans_search_range(min_xyz, max_xyz);
        bbs3d_->set_angular_search_range(config_.min_search_rpy, config_.max_search_rpy);

        // 顶层候选规模估算：地图很大而顶层分辨率又很细时，初始化候选集会膨胀得很快
        // （真实候选数还要再乘上每个角度的划分数，通常是 10~30 倍）
        const double top_res = config_.min_level_res * std::pow(2.0, static_cast<double>(config_.max_level));
        const double init_cells = ((max_xyz.x() - min_xyz.x()) / top_res + 1.0) *
                                  ((max_xyz.y() - min_xyz.y()) / top_res + 1.0) *
                                  ((max_xyz.z() - min_xyz.z()) / top_res + 1.0);
        if (init_cells > 2.0e5) {
            logger->warn("[3dBBS] Initial translation cells ~{:.0f} (top level resolution {:.2f} m); "
                         "the search may be slow - consider increasing bbs.min_level_res / bbs.max_level, "
                         "or turning off bbs.auto_search_range and giving a tighter bbs.search_range.",
                         init_cells, top_res);
        }

        logger->info("[3dBBS] num_threads={}, auto_search_range={}, neighbor_score={} ({})",
                     config_.num_threads, config_.auto_search_range, config_.neighbor_score,
                     config_.neighbor_score >= 2.0 ? "original 3D-BBS" : "weighted 3D-BBS");
        logger->info("[3dBBS] Search range: x[{:.3f}, {:.3f}] y[{:.3f}, {:.3f}] z[{:.3f}, {:.3f}] m | "
                     "rpy[{:.4f}, {:.4f}] [{:.4f}, {:.4f}] [{:.4f}, {:.4f}] rad",
                     min_xyz.x(), max_xyz.x(), min_xyz.y(), max_xyz.y(), min_xyz.z(), max_xyz.z(),
                     config_.min_search_rpy.x(), config_.max_search_rpy.x(),
                     config_.min_search_rpy.y(), config_.max_search_rpy.y(),
                     config_.min_search_rpy.z(), config_.max_search_rpy.z());

        map_ready_ = true;
        state_ = SystemState::COARSE_LOCALIZATION;
        logger->info("Global Map loaded. Algo points: {}, Viz points: {}, Viz Grid: {}m",
                     map_algo->size(), map_viz_cloud_->size(), config_.viz_leaf_size);

        // 初始状态定格：Transient Local 的迟订阅者需要拿到 SEARCHING 初值
        publish_initializer_status("SEARCHING", fins::now());
                     
        if (map_viz_cloud_ && required("global_map_viz")) {
            map_viz_cloud_->header.frame_id = "map";
            send("global_map_viz", map_viz_cloud_, fins::now());
        }
    }

    void on_tracking_status(const std_msgs::msg::String &msg) {
        if (!config_.initialization_only) return;
        std::string failure;   // 锁外计数，避免 note_init_failure 内部加锁与 data_mtx_ 死锁
        bool reinit = false;
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            bool reset_requested = (msg.data == "LOST" && last_tracking_status_ != "LOST") ||
                                   msg.data == "INITIALIZATION_REJECTED";
            last_tracking_status_ = msg.data;
            if (msg.data == "LOCKED") {
                // 跟踪稳定 = 这次初始化成功，清零失败计数
                init_failures_ = 0;
                init_accepted_pending_ = false;
            } else if (reset_requested) {
                if (init_accepted_pending_) {
                    init_accepted_pending_ = false;
                    failure = "接受后未到 LOCKED 即丢失（对齐位置很可能不对）";
                } else if (state_ == SystemState::FINE_LOCALIZATION) {
                    failure = "跟踪丢失";
                }
                if (state_ == SystemState::FINE_LOCALIZATION) {
                    state_ = SystemState::COARSE_LOCALIZATION;
                    accumulated_cloud_odom_->clear();
                    accumulation_started_ = false;
                    job_.has_new_job = false;
                    reinit = true;
                }
            }
        }
        if (reinit) logger->warn("[Initializer] Tracker requested reinitialization: {}", msg.data);
        if (!failure.empty()) note_init_failure(failure);
    }

    void on_odom_tf_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr waiting;
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            latest_T_odom_base_ = tf2::transformToEigen(*msg);
            tf_received_ = true;
            double ts = fins::to_seconds(msg.acq_time);
            odom_history_[ts] = latest_T_odom_base_;
            while (odom_history_.size() > 1000) odom_history_.erase(odom_history_.begin());
            auto it = pending_clouds_.find(ts);
            if (it != pending_clouds_.end()) { waiting = it->second; pending_clouds_.erase(it); }
            if (!config_.initialization_only && map_ready_ && state_ != SystemState::IDLE && config_.pose_trigger == "tf") {
                publish_current_pose(msg.acq_time, latest_T_odom_base_);
            }
        }
        if (waiting) on_cloud_callback(waiting, msg.acq_time);
    }

    void on_cloud_callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_odom_in, fins::AcqTime acq_time) {
        std::unique_lock<std::mutex> lock(data_mtx_);
        if (state_ == SystemState::UNKNOWN_AREA) {
            // 陌生区域：默认停止匹配（不再累积/搜索，BBS 与 GICP 不再烧 CPU/内存）。
            // 配置了 unknown_area_retry_sec>0 时按周期放行一轮重试。
            if (config_.unknown_area_retry_sec > 0.0) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - unknown_area_since_).count()
                        >= config_.unknown_area_retry_sec) {
                    init_failures_ = 0;
                    init_accepted_pending_ = false;
                    accumulation_started_ = false;
                    state_ = SystemState::COARSE_LOCALIZATION;
                    logger->info("[Initializer] 陌生区域停留 {}s 后重试匹配",
                                 config_.unknown_area_retry_sec);
                    publish_initializer_status("SEARCHING", acq_time);
                }
            }
            return;
        }
        if (!map_ready_ || state_ == SystemState::IDLE) return;
        if (config_.initialization_only && state_ != SystemState::COARSE_LOCALIZATION) return;
        double cur_time = fins::to_seconds(acq_time);
        Eigen::Isometry3d cloud_pose = latest_T_odom_base_;
        if (config_.initialization_only) {
            auto it = odom_history_.find(cur_time);
            if (it == odom_history_.end()) {
                pending_clouds_[cur_time] = cloud_odom_in;
                while (pending_clouds_.size() > 30) pending_clouds_.erase(pending_clouds_.begin());
                return;
            }
            cloud_pose = it->second;
        } else if (!tf_received_) return;

        // 精定位阶段：累积 1s 点云后触发 GICP 配准
        if (state_ == SystemState::FINE_LOCALIZATION) {
            // 刚从粗定位切换过来，开始计时累积
            if (last_gicp_run_time_ < 0.0) {
                last_gicp_run_time_ = cur_time;
                accumulated_cloud_odom_->clear();
            }

            auto temp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*cloud_odom_in, 0.2, config_.num_threads);
            *accumulated_cloud_odom_ += *temp;

            double elapsed = cur_time - last_gicp_run_time_;
            if (elapsed >= config_.tracking_interval) {
                if (!is_processing_job_) {
                    job_.cloud_odom.reset(new pcl::PointCloud<pcl::PointXYZI>(*accumulated_cloud_odom_));
                    job_.ts = acq_time;
                    job_.T_odom_base = cloud_pose;
                    job_.has_new_job = true;
                    last_gicp_run_time_ = cur_time;
                    accumulated_cloud_odom_->clear();
                    cv_.notify_one();
                }
            }
            return; // 结束回调
        }

        // 粗定位（BBS）阶段：累积 1.0 秒点云后触发 3dBBS 搜索
        if (!accumulation_started_) {
            accumulation_started_ = true;
            start_accumulation_time_ = cur_time;
            accumulated_cloud_odom_->clear();
        }

        auto temp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*cloud_odom_in, 0.2, config_.num_threads);
        *accumulated_cloud_odom_ += *temp;

        if (cur_time - start_accumulation_time_ > config_.accumulation_time) {
            if (!is_processing_job_) {
                job_.cloud_odom.reset(new pcl::PointCloud<pcl::PointXYZI>(*accumulated_cloud_odom_));
                job_.ts = acq_time;
                job_.T_odom_base = cloud_pose;
                job_.has_new_job = true;
                cv_.notify_one();
            }
            accumulation_started_ = false;
        }
    }
    void worker_loop() {
        while (is_worker_running_) {
            JobData current_job;
            {
                std::unique_lock<std::mutex> lock(data_mtx_);
                cv_.wait(lock, [this] { return !is_worker_running_ || (!is_paused_ && job_.has_new_job); });
                if (!is_worker_running_) break;
                if (is_paused_) continue;
                current_job = job_;
                job_.has_new_job = false;
                is_processing_job_ = true;
            }

            if (state_ == SystemState::COARSE_LOCALIZATION) {
                perform_coarse_localization(current_job);
            } else if (!config_.initialization_only && state_ == SystemState::FINE_LOCALIZATION) {
                perform_fine_tracking(current_job);
            }
            is_processing_job_ = false;
        }
    }

    void perform_coarse_localization(const JobData& job) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*job.cloud_odom, *cloud_base, job.T_odom_base.inverse().matrix().cast<float>());
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bbs(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*cloud_base, *cloud_bbs);
        
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_bbs);
        sor.setLeafSize(config_.bbs_leaf_size, config_.bbs_leaf_size, config_.bbs_leaf_size);
        sor.filter(*cloud_bbs);

        std::vector<Eigen::Vector3d> src_points;
        for (auto& p : cloud_bbs->points) src_points.push_back(p.getVector3fMap().cast<double>());
        
        bbs3d_->set_src_points(src_points);
        // 平移/角度搜索范围已在 load_map() 中按配置（或地图包围盒）设置完毕
        
        bbs3d_->localize();

        auto t_end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (bbs3d_->has_localized()) {
            logger->info("[3dBBS] Localized! Score: {} ({:.2f}%), Time: {:.2f}ms",
                         bbs3d_->get_best_score(),
                         bbs3d_->get_best_score_percentage() * 100.0,
                         duration_ms);

            // Log confidence evaluation
            double confidence = bbs3d_->get_confidence_score();
            double peak_margin = bbs3d_->get_peak_margin();
            size_t num_candidates = bbs3d_->get_top_candidates().size();
            logger->info("[3dBBS] Confidence={:.3f}, PeakMargin={:.3f}, TopK={}",
                         confidence, peak_margin, num_candidates);

            // Publish candidate visualization markers
            publish_bbs_candidates(job.ts);

            Eigen::Isometry3d T_map_base_bbs(bbs3d_->get_global_pose());
            T_map_base_bbs.linear() = Eigen::Quaterniond(T_map_base_bbs.linear()).normalized().toRotationMatrix();
            
            pcl::PointCloud<pcl::PointCovariance>::Ptr source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
                *cloud_base, config_.registered_leaf_size, config_.num_threads);
            
            small_gicp::estimate_covariances_omp(*source_down, 20, config_.num_threads);

            small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
            reg.reduction.num_threads = config_.num_threads;
            reg.rejector.max_dist_sq = config_.max_dist_sq_bbs_refine; 
            reg.optimizer.max_iterations = 20;

            auto t_refine_start = std::chrono::high_resolution_clock::now();
            auto res = reg.align(*target_covs_, *source_down, *target_tree_, T_map_base_bbs);
            auto t_refine_end = std::chrono::high_resolution_clock::now();
            double refine_duration_ms = std::chrono::duration<double, std::milli>(t_refine_end - t_refine_start).count();

            logger->info("[3dBBS->GICP] Refine converged={}, error={:.2f}, iter={}, time={:.2f}ms", 
                         res.converged, res.error, res.iterations, refine_duration_ms);

            Eigen::Isometry3d T_map_base_final = res.converged ? res.T_target_source : T_map_base_bbs;

            if (config_.initialization_only) {
                // 接受门控：res.error 是 covariance 加权残差和、不是米制量，错位候选靠少量
                // 远距对应点也能把它拉低（实测错位 accept 的 error=1134，正确 accept 反而
                // 12000~15500）。必须独立复核「变换后的源点是否真的落在地图上」。
                const double inlier_ratio = source_down->empty() ? 0.0
                    : double(res.num_inliers) / double(source_down->size());
                const OverlapStat ov = verify_overlap(source_down, T_map_base_final);
                const bool accept = res.converged && std::isfinite(res.error)
                    && res.error <= config_.max_gicp_error
                    && res.T_target_source.matrix().allFinite()
                    && int(res.num_inliers) >= config_.min_init_inliers
                    && inlier_ratio >= config_.min_init_inlier_ratio
                    && ov.ratio >= config_.min_init_overlap_ratio;
                if (!accept) {
                    // 注意：fmt 不支持 {:.0%} 格式说明符（invalid format specifier
                    // 会直接 throw 到 terminate），百分比一律 ×100 用 {:.0f} 输出。
                    logger->warn("[Initializer] BBS candidate rejected: inliers={} ({:.0f}%), "
                                 "map overlap={:.0f}% (need >={:.0f}% @ {}m), error={:.1f}",
                                 res.num_inliers, 100.0 * inlier_ratio, 100.0 * ov.ratio,
                                 100.0 * config_.min_init_overlap_ratio,
                                 config_.init_verify_max_dist, res.error);
                    note_init_failure("候选未通过落图复核");
                    return;
                }
                logger->info("[Initializer] 落图复核通过: inliers={} ({:.0f}%), map overlap={:.0f}%",
                             res.num_inliers, 100.0 * inlier_ratio, 100.0 * ov.ratio);
            }

            {
                std::lock_guard<std::mutex> lock(data_mtx_);
                T_map_odom_ = T_map_base_final * job.T_odom_base.inverse();
                if (config_.planar_constraint) T_map_odom_ = apply_planar_constraint(T_map_odom_);
                // 初始化平滑滤波器状态，使后续精定位从当前值开始平滑
                has_filtered_pose_ = false;
                state_ = SystemState::FINE_LOCALIZATION;
                consecutive_gicp_failures_ = 0;
                init_accepted_pending_ = true;   // 等 LOCKED 才算真正成功
                // 清空累积点云，为精定位阶段的 1s 累积做准备
                accumulated_cloud_odom_->clear();
                last_gicp_run_time_ = -1.0;  // 标记：等待首个点云到达时开始计时
                if (!config_.initialization_only && config_.pose_trigger == "gicp") {
                    publish_current_pose(job.ts, job.T_odom_base);
                }
                publish_initial_pose(job.ts, job.T_odom_base);
            }
            if (config_.initialization_only) {
                geometry_msgs::msg::TransformStamped alignment;
                alignment.header.stamp = fins::to_ros_msg_time(job.ts);
                alignment.header.frame_id = "map"; alignment.child_frame_id = "odom";
                alignment.transform = tf2::eigenToTransform(T_map_odom_).transform;
                send("initial_alignment", alignment, job.ts);
                logger->info("[Initializer] BBS + GICP accepted; handing map->odom alignment to FAST-LIO, no periodic GICP");
                publish_initializer_status("INITIALIZED", job.ts);
            }
            publish_results(job.ts, job.cloud_odom, job.T_odom_base);
        } else {
            if (bbs3d_->has_timed_out()) {
                logger->warn("[3dBBS] 搜索超时（上限 {}ms），按未匹配处理。Time: {:.2f}ms",
                             config_.bbs_timeout_msec, duration_ms);
                note_init_failure("3dBBS 搜索超时（按未匹配计）");
            } else {
                logger->warn("[3dBBS] Localization failed. Time: {:.2f}ms", duration_ms);
                note_init_failure("3dBBS 未匹配到候选");
            }
        }
    }

    void perform_fine_tracking(const JobData& job) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. 将 Odom 系下的累积点云转换到当前 base_link 坐标系下
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*job.cloud_odom, *cloud_base, job.T_odom_base.inverse().matrix().cast<float>());

        // 2. 对 base_link 系下的点云进行下采样
        pcl::PointCloud<pcl::PointCovariance>::Ptr source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
            *cloud_base, config_.registered_leaf_size, config_.num_threads);
        
        small_gicp::estimate_covariances_omp(*source_down, 20, config_.num_threads);

        // 3. 计算基于里程计预测的 T_map_base 初始估计
        Eigen::Isometry3d T_map_base_init = T_map_odom_ * job.T_odom_base;

        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
        reg.reduction.num_threads = config_.num_threads;
        reg.rejector.max_dist_sq = config_.max_dist_sq_fine; 
        reg.optimizer.max_iterations = 20;
        
        // 4. 在 base_link 系下进行配准
        auto result = reg.align(*target_covs_, *source_down, *target_tree_, T_map_base_init);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (result.converged && result.error < config_.max_gicp_error) {
            Eigen::Isometry3d T_delta = result.T_target_source * T_map_base_init.inverse();
            Eigen::Vector3d t_diff = T_delta.translation();
            Eigen::Vector3d rpy_diff = T_delta.linear().eulerAngles(0, 1, 2) * 180.0 / M_PI;

            logger->info("[GICP] Result: converged={}, error={:.2f}, iter={}, time={:.2f}ms | Delta: xyz[{:.3f} {:.3f} {:.3f}] rpy[{:.3f} {:.3f} {:.3f} deg]",
                        result.converged, result.error, result.iterations, duration_ms,
                        t_diff.x(), t_diff.y(), t_diff.z(),
                        rpy_diff.x(), rpy_diff.y(), rpy_diff.z());

            {
                std::lock_guard<std::mutex> lock(data_mtx_);
                // 5. 根据配准后的 T_map_base 更新 T_map_odom（原始值），然后应用平滑滤波
                Eigen::Isometry3d T_map_odom_raw = result.T_target_source * job.T_odom_base.inverse();
                T_map_odom_ = apply_smooth_filter(T_map_odom_raw);
                if (config_.planar_constraint) T_map_odom_ = apply_planar_constraint(T_map_odom_);
                consecutive_gicp_failures_ = 0;
                if (config_.pose_trigger == "gicp") {
                    publish_current_pose(job.ts, job.T_odom_base);
                }
            }
            publish_results(job.ts, job.cloud_odom, job.T_odom_base);
        } else {
            consecutive_gicp_failures_++;
            logger->warn("[GICP] Tracking Failed (converged={}, error={:.2f}). consecutive: {}", 
                        result.converged, result.error, consecutive_gicp_failures_);
            if (consecutive_gicp_failures_ >= config_.max_gicp_failures) {
                logger->error("[GICP] Too many failures, Resetting to COARSE.");
                state_ = SystemState::COARSE_LOCALIZATION;
            }
        }
    }

    Eigen::Isometry3d apply_smooth_filter(const Eigen::Isometry3d& raw_T_map_odom) {
        if (!config_.smooth_enabled) {
            return raw_T_map_odom;
        }

        if (!has_filtered_pose_) {
            T_map_odom_filtered_ = raw_T_map_odom;
            has_filtered_pose_ = true;
            return raw_T_map_odom;
        }

        // Extract translation and rotation from previous filtered pose
        Eigen::Vector3d t_prev = T_map_odom_filtered_.translation();
        Eigen::Quaterniond q_prev(T_map_odom_filtered_.rotation());
        q_prev.normalize();

        // Extract translation and rotation from new raw pose
        Eigen::Vector3d t_raw = raw_T_map_odom.translation();
        Eigen::Quaterniond q_raw(raw_T_map_odom.rotation());
        q_raw.normalize();

        // EMA for translation: linear interpolation
        Eigen::Vector3d t_smooth = config_.smooth_alpha * t_raw + (1.0 - config_.smooth_alpha) * t_prev;

        // EMA for rotation: spherical linear interpolation (slerp)
        Eigen::Quaterniond q_smooth = q_prev.slerp(config_.smooth_alpha, q_raw);
        q_smooth.normalize();

        // Build filtered pose
        Eigen::Isometry3d filtered = Eigen::Isometry3d::Identity();
        filtered.translation() = t_smooth;
        filtered.linear() = q_smooth.toRotationMatrix();

        T_map_odom_filtered_ = filtered;

        double trans_delta = (t_raw - t_prev).norm();
        double rot_delta = q_prev.angularDistance(q_raw) * 180.0 / M_PI;
        logger->debug("[Smooth] Raw delta: trans={:.3f}m rot={:.2f}deg | Filtered delta: trans={:.3f}m rot={:.2f}deg | alpha={:.2f}",
                      trans_delta, rot_delta,
                      (t_smooth - t_prev).norm(),
                      q_prev.angularDistance(q_smooth) * 180.0 / M_PI,
                      config_.smooth_alpha);

        return filtered;
    }

    Eigen::Isometry3d apply_planar_constraint(const Eigen::Isometry3d& T) {
        Eigen::Isometry3d result = T;
        result.translation().z() = 0.0;
        Eigen::Vector3d rpy = result.linear().eulerAngles(0, 1, 2);
        result.linear() = Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
        return result;
    }

    void publish_results(fins::AcqTime ts, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_odom, const Eigen::Isometry3d& T_odom_base) {
    /*
        geometry_msgs::msg::TransformStamped tf;
    	tf.header.stamp = fins::to_ros_msg_time(ts);
        tf.header.frame_id = "map";
        tf.child_frame_id = "odom";
        
        Eigen::Vector3d t = T_map_odom_.translation();
        Eigen::Quaterniond q(T_map_odom_.rotation());
        
        tf.transform.translation.x = t.x();
        tf.transform.translation.y = t.y();
        tf.transform.translation.z = t.z();
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        
        send("$T_{map}^{odom}$", tf, ts);*/

        if (required("aligned_cloud")) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*cloud_odom, *aligned, T_map_odom_.matrix().cast<float>());
            aligned->header.frame_id = "map";
            send("aligned_cloud", aligned, ts);
        }
    }

    void publish_current_pose(fins::AcqTime ts, const Eigen::Isometry3d& T_odom_base) {
    
        geometry_msgs::msg::TransformStamped tf;
    	tf.header.stamp = fins::to_ros_msg_time(ts);
        tf.header.frame_id = "map";
        tf.child_frame_id = "odom";
        
        Eigen::Vector3d t = T_map_odom_.translation();
        Eigen::Quaterniond q(T_map_odom_.rotation());
        
        tf.transform.translation.x = t.x();
        tf.transform.translation.y = t.y();
        tf.transform.translation.z = t.z();
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        
        send("$T_{map}^{odom}$", tf, ts);
        
        if (required("current_pose")) {
            Eigen::Isometry3d T_map_base = T_map_odom_ * T_odom_base;
            if (config_.planar_constraint) T_map_base = apply_planar_constraint(T_map_base);
            geometry_msgs::msg::PoseWithCovarianceStamped pose;
            pose.header.frame_id = "map";
            pose.header.stamp = fins::to_ros_msg_time(ts); // Use the provided timestamp

            Eigen::Vector3d t_base = T_map_base.translation();
            Eigen::Quaterniond q_base(T_map_base.rotation());

            pose.pose.pose.position.x = t_base.x();
            pose.pose.pose.position.y = t_base.y();
            pose.pose.pose.position.z = t_base.z();
            pose.pose.pose.orientation.w = q_base.w();
            pose.pose.pose.orientation.x = q_base.x();
            pose.pose.pose.orientation.y = q_base.y();
            pose.pose.pose.orientation.z = q_base.z();

            // Covariance all zeros
            for (int i = 0; i < 36; i++) pose.pose.covariance[i] = 0.0;

            send("current_pose", pose, ts);
        }
    }

    void publish_initial_pose(fins::AcqTime ts, const Eigen::Isometry3d& T_odom_base) {
        if (!required("initial_pose")) return;

        Eigen::Isometry3d T_map_base = T_map_odom_ * T_odom_base;
        if (config_.planar_constraint) T_map_base = apply_planar_constraint(T_map_base);
        geometry_msgs::msg::PoseWithCovarianceStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = fins::to_ros_msg_time(ts);

        Eigen::Vector3d t_base = T_map_base.translation();
        Eigen::Quaterniond q_base(T_map_base.rotation());

        pose.pose.pose.position.x = t_base.x();
        pose.pose.pose.position.y = t_base.y();
        pose.pose.pose.position.z = t_base.z();
        pose.pose.pose.orientation.w = q_base.w();
        pose.pose.pose.orientation.x = q_base.x();
        pose.pose.pose.orientation.y = q_base.y();
        pose.pose.pose.orientation.z = q_base.z();

        for (int i = 0; i < 36; i++) pose.pose.covariance[i] = 0.0;

        send("initial_pose", pose, ts);
    }

    void publish_bbs_candidates(fins::AcqTime ts) {
        if (!required("bbs_candidates")) return;
        if (!bbs3d_->has_localized()) return;

        auto candidates = bbs3d_->get_top_candidates();
        if (candidates.empty()) return;

        visualization_msgs::msg::MarkerArray marker_array;
        auto ros_time = fins::to_ros_msg_time(ts);

        // Leaf-level resolution for converting discrete transforms to continuous poses
        double min_res = config_.min_level_res;
        cpu::AngularInfo leaf_ang = bbs3d_->get_leaf_angular_info();

        // Color scheme: green (best), yellow (2nd), orange (3rd)
        const std::array<std::array<double, 4>, 3> colors = {{
            {0.0, 1.0, 0.0, 1.0},       // Green
            {1.0, 0.8, 0.0, 0.9},       // Yellow-orange
            {1.0, 0.4, 0.0, 0.7},       // Orange, semi-transparent
        }};

        int marker_id = 0;
        for (size_t i = 0; i < candidates.size() && i < 3; ++i) {
            const auto& cand = candidates[i];
            const auto& color = colors[i];

            // Convert discrete transform to continuous 4x4 pose matrix
            Eigen::Matrix4d pose_matrix = cand.create_matrix(
                min_res, leaf_ang.rpy_res, leaf_ang.min_rpy);
            Eigen::Isometry3d T(pose_matrix);
            T.linear() = Eigen::Quaterniond(T.linear()).normalized().toRotationMatrix();

            Eigen::Vector3d pos = T.translation();
            Eigen::Quaterniond q(T.linear());

            double score_ratio = (bbs3d_->get_src_points().size() > 0)
                ? static_cast<double>(cand.score) / (2.0 * bbs3d_->get_src_points().size())
                : 0.0;

            // Arrow marker pointing along yaw direction
            {
                visualization_msgs::msg::Marker arrow;
                arrow.header.frame_id = "map";
                arrow.header.stamp = ros_time;
                arrow.ns = "bbs_candidates_arrow";
                arrow.id = marker_id++;
                arrow.type = visualization_msgs::msg::Marker::ARROW;
                arrow.action = visualization_msgs::msg::Marker::ADD;
                arrow.pose.position.x = pos.x();
                arrow.pose.position.y = pos.y();
                arrow.pose.position.z = pos.z();
                arrow.pose.orientation.w = q.w();
                arrow.pose.orientation.x = q.x();
                arrow.pose.orientation.y = q.y();
                arrow.pose.orientation.z = q.z();
                // Scale: shaft diameter, head diameter, shaft length
                double shaft_len = 1.5 + 1.0 * score_ratio;
                arrow.scale.x = 0.12;     // shaft diameter
                arrow.scale.y = 0.22;     // head diameter
                arrow.scale.z = shaft_len;
                arrow.color.r = color[0];
                arrow.color.g = color[1];
                arrow.color.b = color[2];
                arrow.color.a = color[3];
                marker_array.markers.push_back(arrow);
            }

            // Text marker with rank and score
            {
                visualization_msgs::msg::Marker text;
                text.header.frame_id = "map";
                text.header.stamp = ros_time;
                text.ns = "bbs_candidates_text";
                text.id = marker_id++;
                text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text.action = visualization_msgs::msg::Marker::ADD;
                text.pose.position.x = pos.x();
                text.pose.position.y = pos.y();
                text.pose.position.z = pos.z() + 0.6;
                text.scale.z = 0.35;  // text height
                text.color.r = color[0];
                text.color.g = color[1];
                text.color.b = color[2];
                text.color.a = color[3];
                std::ostringstream oss;
                oss << "#" << (i + 1) << ": " << std::fixed << std::setprecision(1)
                    << (score_ratio * 100.0) << "%";
                text.text = oss.str();
                marker_array.markers.push_back(text);
            }
        }

        send("bbs_candidates", marker_array, ts);
    }

    Config config_;
    std::atomic<SystemState> state_{SystemState::IDLE};
    std::string last_tracking_status_;
    std::map<double, Eigen::Isometry3d> odom_history_;
    std::map<double, pcl::PointCloud<pcl::PointXYZI>::Ptr> pending_clouds_;
    bool map_ready_{false};
    bool tf_received_{false};
    
    Eigen::Isometry3d T_map_odom_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d latest_T_odom_base_{Eigen::Isometry3d::Identity()};

    // Smoothing filter state
    bool has_filtered_pose_{false};
    Eigen::Isometry3d T_map_odom_filtered_{Eigen::Isometry3d::Identity()};
    
    int consecutive_gicp_failures_ = 0;
    bool accumulation_started_ = false;
    double start_accumulation_time_ = 0.0;
    double last_gicp_run_time_ = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_odom_;

    // 初始化尝试计数（陌生区域停搜）；原子量避免 worker / 状态回调并发读写
    std::atomic<int> init_failures_{0};
    std::atomic<bool> init_accepted_pending_{false};
    std::chrono::steady_clock::time_point unknown_area_since_;

    JobData job_;
    std::atomic<bool> is_processing_job_{false};
    std::mutex data_mtx_;
    std::condition_variable cv_;
    
    std::thread worker_thread_;
    std::atomic<bool> is_worker_running_{false};
    std::atomic<bool> is_paused_{false};

    std::unique_ptr<cpu::BBS3D> bbs3d_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_viz_cloud_;
    pcl::PointCloud<pcl::PointCovariance>::Ptr target_covs_;
    std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>> target_tree_;
};

EXPORT_NODE(GlobalLocalizationNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)
