/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <fins/utils/time.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

#include <cpu_bbs3d/bbs3d.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <atomic>

namespace fs = std::filesystem;

enum class SystemState { IDLE, COARSE_LOCALIZATION, FINE_LOCALIZATION };

class GlobalLocalizationNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void define() override {
        set_name("GlobalLocalization");
        set_description("Global Localization using 3dBBS and GICP with Refinement logic");
        set_category("Localization");

        register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &GlobalLocalizationNode::on_cloud_callback);
        register_input<nav_msgs::msg::Odometry>("odom_in", &GlobalLocalizationNode::on_odom_callback);

        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("global_map_viz");
        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("aligned_cloud");
        register_output<geometry_msgs::msg::TransformStamped>("$T_{map}^{odom}$");

        register_parameter<std::string>("global_map_path", &GlobalLocalizationNode::on_map_path_changed, "");
        register_parameter<std::string>("voxel_map_dir", &GlobalLocalizationNode::on_voxel_dir_changed, "");
    }

    void initialize() override {
        logger->info("Initializing Global Localization Node...");

        // 搜索范围参数
        min_search_xyz_ = Eigen::Vector3d(-100.0, -100.0, -2.0);
        max_search_xyz_ = Eigen::Vector3d(100.0, 100.0, 2.0);
        min_search_rpy_ = Eigen::Vector3d(0.0, 0.0, -M_PI);
        max_search_rpy_ = Eigen::Vector3d(0.0, 0.0, M_PI);

        // 算法配置
        num_threads_ = 8;
        score_threshold_ = 0.3;
        accumulation_time_ = 1.5;
        
        min_level_res_ = 1.5;
        max_level_ = 5;
        global_leaf_size_ = 0.8;
        registered_leaf_size_ = 0.5;
        max_dist_sq_fine_ = 4.0;  // 精定位匹配阈值
        max_dist_sq_bbs_refine_ = 25.0; // BBS后精修匹配阈值
        max_gicp_failures_ = 5;

        bbs3d_ = std::make_unique<cpu::BBS3D>();
        accumulated_cloud_odom_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        T_map_odom_ = Eigen::Isometry3d::Identity();
        state_ = SystemState::IDLE;

        is_running_ = true;
        worker_thread_ = std::thread(&GlobalLocalizationNode::worker_loop, this);
    }

    void run() override { running_ = true; logger->info("System Running"); }
    void pause() override { running_ = false; }
    void reset() override {
        std::lock_guard<std::mutex> lock(data_mtx_);
        state_ = SystemState::COARSE_LOCALIZATION;
        accumulated_cloud_odom_->clear();
        accumulation_started_ = false;
        consecutive_gicp_failures_ = 0;
        logger->info("GlobalLocalizationNode: Reset to COARSE mode.");
    }

    ~GlobalLocalizationNode() {
        is_running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    void on_map_path_changed(std::string path) { load_map(path); }
    void on_voxel_dir_changed(std::string dir) { voxel_dir_ = dir; }

    void on_odom_callback(const fins::Msg<nav_msgs::msg::Odometry>& msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        latest_odom_ = *msg;
        odom_received_ = true;
    }

    void on_cloud_callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_raw, fins::AcqTime acq_time) {
        if (state_ == SystemState::IDLE || !odom_received_ || !running_) return;

        std::unique_lock<std::mutex> lock(data_mtx_);
        double cur_time = fins::to_seconds(acq_time);

        // 获取当前里程计位姿并将点云转到 Odom 系
        Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
        const auto& pose = latest_odom_.pose.pose;
        T_odom_base.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
        T_odom_base.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_odom(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*cloud_raw, *cloud_odom, T_odom_base.matrix().cast<float>());

        if (!accumulation_started_) {
            accumulation_started_ = true;
            start_accumulation_time_ = cur_time;
            accumulated_cloud_odom_->clear();
        }

        // 小幅下采样累加 (0.2m) 保证效率
        auto temp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*cloud_odom, 0.2, num_threads_);
        *accumulated_cloud_odom_ += *temp;

        if (cur_time - start_accumulation_time_ > accumulation_time_) {
            if (!processing_job_) {
                pending_cloud_ = accumulated_cloud_odom_;
                pending_time_ = acq_time;
                pending_odom_base_ = T_odom_base;
                has_new_job_ = true;
                cv_.notify_one();
            }
            accumulation_started_ = false;
            accumulated_cloud_odom_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        }
    }

    void load_map(const std::string& path) {
        if (!fs::exists(path)) {
            logger->error("Map not found at: {}", path);
            return;
        }
        
        auto t_start = std::chrono::high_resolution_clock::now();
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(path, *raw_map) == -1) return;

        // 可视化下采样
        global_map_viz_target_ = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*raw_map, 0.7);
        // 算法下采样
        auto downsampled_map = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*global_map_viz_target_, global_leaf_size_);
        
        init_gicp_target_from_cloud(downsampled_map);

        // BBS 初始化
        bool voxel_loaded = false;
        if (!voxel_dir_.empty() && fs::exists(voxel_dir_) && !fs::is_empty(voxel_dir_)) {
            if (bbs3d_->load_voxel_params(voxel_dir_) && bbs3d_->set_multi_buckets(voxel_dir_)) {
                logger->info("Voxelmaps loaded from: {}", voxel_dir_);
                voxel_loaded = true;
            }
        }
        if (!voxel_loaded) {
            std::vector<Eigen::Vector3d> tar_points;
            for (auto& p : downsampled_map->points) tar_points.push_back(p.getVector3fMap().cast<double>());
            bbs3d_->set_tar_points(tar_points, min_level_res_, max_level_);
            if (!voxel_dir_.empty()) {
                fs::create_directories(voxel_dir_);
                bbs3d_->save_voxel_params(voxel_dir_);
            }
        }

        bbs3d_->set_score_threshold_percentage(score_threshold_);
        bbs3d_->set_num_threads(16);

        send("global_map_viz", global_map_viz_target_, fins::now());
        state_ = SystemState::COARSE_LOCALIZATION;
        logger->info("Map loaded. Mode: COARSE. Time: {:.2f}ms", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count());
    }

    void init_gicp_target_from_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud) {
        target_covs_.reset(new pcl::PointCloud<pcl::PointCovariance>());
        target_covs_->points.resize(map_cloud->points.size());
        for (size_t i = 0; i < map_cloud->points.size(); ++i) target_covs_->points[i].getVector3fMap() = map_cloud->points[i].getVector3fMap();
        small_gicp::estimate_covariances_omp(*target_covs_, 20, num_threads_);
        target_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target_covs_, small_gicp::KdTreeBuilderOMP(num_threads_));
    }

    void worker_loop() {
        while (is_running_) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr query;
            fins::AcqTime ts;
            Eigen::Isometry3d T_odom_base_at_acq;

            {
                std::unique_lock<std::mutex> lock(data_mtx_);
                cv_.wait(lock, [this] { return !is_running_ || has_new_job_; });
                if (!is_running_) break;
                query = pending_cloud_;
                ts = pending_time_;
                T_odom_base_at_acq = pending_odom_base_;
                has_new_job_ = false;
                processing_job_ = true;
            }

            if (state_ == SystemState::COARSE_LOCALIZATION) {
                // 1. BBS 粗定位
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bbs(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*query, *cloud_base, T_odom_base_at_acq.inverse().matrix().cast<float>());
                pcl::copyPointCloud(*cloud_base, *cloud_bbs);

                pcl::VoxelGrid<pcl::PointXYZ> sor;
                sor.setInputCloud(cloud_bbs);
                sor.setLeafSize(0.5f, 0.5f, 0.5f);
                sor.filter(*cloud_bbs);

                std::vector<Eigen::Vector3d> src_points;
                for (auto& p : cloud_bbs->points) src_points.push_back(p.getVector3fMap().cast<double>());

                bbs3d_->set_src_points(src_points);
                bbs3d_->set_trans_search_range(min_search_xyz_, max_search_xyz_);
                bbs3d_->set_angular_search_range(min_search_rpy_, max_search_rpy_);
                
                {
                    auto t = this->recorder("bbs3d_localize", Measures.ts);
                    bbs3d_->localize();
                }

                if (bbs3d_->has_localized()) {
                    Eigen::Isometry3d T_map_base_bbs(bbs3d_->get_global_pose());
                    // 归一化旋转防止漂移
                    Eigen::Quaterniond q(T_map_base_bbs.linear());
                    q.normalize();
                    T_map_base_bbs.linear() = q.toRotationMatrix();

                    // --- 逻辑改动 1: BBS 后立即进行 GICP 精修 ---
                    logger->info("[3dBBS] Success. Refining with GICP...");
                    
                    // 将 query 转到 BBS 预测的 map 系下
                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_map_bbs(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::transformPointCloud(*cloud_base, *cloud_in_map_bbs, T_map_base_bbs.matrix().cast<float>());
                    
                    auto source_gicp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(*cloud_in_map_bbs, registered_leaf_size_, num_threads_);
                    small_gicp::estimate_covariances_omp(*source_gicp, 20, num_threads_);

                    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
                    reg.reduction.num_threads = num_threads_;
                    reg.rejector.max_dist_sq = max_dist_sq_bbs_refine_; // 使用较大的阈值 (25.0)

                    auto res = reg.align(*target_covs_, *source_gicp, *target_tree_, Eigen::Isometry3d::Identity());

                    if (res.converged) {
                        logger->info("[3dBBS] Refinement success. Error: {:.2f}", res.error);
                        // 更新 T_map_odom: T_map_odom = T_delta * T_map_base_bbs * T_base_odom
                        T_map_odom_ = res.T_target_source * T_map_base_bbs * T_odom_base_at_acq.inverse();
                    } else {
                        logger->warn("[3dBBS] Refinement failed, using raw BBS pose.");
                        T_map_odom_ = T_map_base_bbs * T_odom_base_at_acq.inverse();
                    }

                    state_ = SystemState::FINE_LOCALIZATION;
                    consecutive_gicp_failures_ = 0; // 重置计数
                    publish_results(ts, query);
                }
            } 
            else if (state_ == SystemState::FINE_LOCALIZATION) {
                // 2. GICP 跟踪
                auto source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(*query, registered_leaf_size_, num_threads_);
                small_gicp::estimate_covariances_omp(*source_down, 20, num_threads_);

                small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
                reg.reduction.num_threads = num_threads_;
                reg.rejector.max_dist_sq = max_dist_sq_fine_; // 正常跟踪阈值 (4.0)
                
                auto result = reg.align(*target_covs_, *source_down, *target_tree_, T_map_odom_);
                
                // --- 逻辑改动 2: 连续失败计数逻辑 ---
                if (result.converged && result.error < 60000.0) { // 增加一个基本的 error 检查
                    T_map_odom_ = result.T_target_source;
                    consecutive_gicp_failures_ = 0; // 成功则重置
                    publish_results(ts, query);
                } else {
                    consecutive_gicp_failures_++;
                    logger->warn("[GICP] Tracking Failure ({}/{})", consecutive_gicp_failures_, max_gicp_failures_);
                    
                    if (consecutive_gicp_failures_ >= max_gicp_failures_) {
                        logger->error("[GICP] Too many failures. Reverting to COARSE.");
                        state_ = SystemState::COARSE_LOCALIZATION;
                    }
                }
            }
            processing_job_ = false;
        }
    }

    void publish_results(fins::AcqTime ts, pcl::PointCloud<pcl::PointXYZI>::Ptr query) {
        geometry_msgs::msg::TransformStamped tf;
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

        if (required("aligned_cloud")) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*query, *aligned, T_map_odom_.matrix().cast<float>());
            send("aligned_cloud", aligned, ts);
        }
    }

    // 状态与同步
    SystemState state_{SystemState::IDLE};
    std::mutex data_mtx_;
    std::condition_variable cv_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> running_{false};
    
    // 算法对象
    std::unique_ptr<cpu::BBS3D> bbs3d_;
    pcl::PointCloud<pcl::PointCovariance>::Ptr target_covs_;
    std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>> target_tree_;
    Eigen::Isometry3d T_map_odom_;

    // 配置参数
    Eigen::Vector3d min_search_xyz_, max_search_xyz_, min_search_rpy_, max_search_rpy_;
    double score_threshold_, accumulation_time_;
    int num_threads_;
    std::string voxel_dir_;
    double min_level_res_;
    int max_level_;
    double global_leaf_size_, registered_leaf_size_;
    double max_dist_sq_fine_, max_dist_sq_bbs_refine_;
    int max_gicp_failures_ = 5;
    int consecutive_gicp_failures_ = 0;

    // 数据缓存
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map_viz_target_;
    bool has_new_job_{false};
    bool processing_job_{false};
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_odom_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pending_cloud_;
    fins::AcqTime pending_time_;
    Eigen::Isometry3d pending_odom_base_;
    
    bool accumulation_started_{false};
    double start_accumulation_time_{0.0};
    nav_msgs::msg::Odometry latest_odom_;
    bool odom_received_{false};

    std::thread worker_thread_;
};

EXPORT_NODE(GlobalLocalizationNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)