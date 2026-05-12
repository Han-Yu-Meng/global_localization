/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <fins/agent/parameter_server.hpp>
#include <fins/utils/time.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

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

    struct Config {
        // 搜索范围保留默认值
        Eigen::Vector3d min_search_xyz{-100.0, -100.0, -2.0};
        Eigen::Vector3d max_search_xyz{100.0, 100.0, 2.0};
        Eigen::Vector3d min_search_rpy{0.0, 0.0, -M_PI};
        Eigen::Vector3d max_search_rpy{0.0, 0.0, M_PI};
        
        int num_threads = 8;
        double score_threshold = 0.3;
        double accumulation_time = 1.0;
        
        double min_level_res = 1.5;
        int max_level = 5;
        double global_leaf_size = 0.8;
        double registered_leaf_size = 0.5;
        double bbs_leaf_size = 0.5;
        double viz_leaf_size = 0.7;    // 新增参数：可视化地图采样粒度
        
        double max_dist_sq_fine = 4.0;
        double max_dist_sq_bbs_refine = 25.0;
        double max_gicp_error = 60000.0; // 新增参数：GICP 容许最大误差阈值
        int max_gicp_failures = 5;
        std::string voxel_dir = "";
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
    }

    void initialize() override {
        fins::ParamLoader config("GlobalLocalization");
        
        config_.num_threads = config.get("num_threads", 8);
        config_.score_threshold = config.get("score_threshold", 0.3);
        config_.accumulation_time = config.get("accumulation_time", 1.0);
        config_.min_level_res = config.get("min_level_res", 1.5);
        config_.max_level = config.get("max_level", 5);
        config_.global_leaf_size = config.get("global_leaf_size", 0.8);
        config_.registered_leaf_size = config.get("registered_leaf_size", 0.5);
        config_.bbs_leaf_size = config.get("bbs_leaf_size", 0.5);
        config_.viz_leaf_size = config.get("viz_leaf_size", 0.7);
        config_.max_dist_sq_fine = config.get("max_dist_sq_fine", 4.0);
        config_.max_gicp_error = config.get("max_gicp_error", 60000.0);
        config_.voxel_dir = config.get("voxel_dir", std::string(""));

        bbs3d_ = std::make_unique<cpu::BBS3D>();
        accumulated_cloud_odom_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        
        std::string map_path = config.get("map_dir", std::string(""));
        if (!map_path.empty()) load_map(map_path);

        is_worker_running_ = true;
        worker_thread_ = std::thread(&GlobalLocalizationNode::worker_loop, this);
    }

    void run() override { is_node_active_ = true; }
    void pause() override { is_node_active_ = false; }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(data_mtx_);
        state_ = map_ready_ ? SystemState::COARSE_LOCALIZATION : SystemState::IDLE;
        accumulated_cloud_odom_->clear();
        accumulation_started_ = false;
        consecutive_gicp_failures_ = 0;
    }

    ~GlobalLocalizationNode() {
        is_worker_running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    void load_map(const std::string& path) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(path, *raw_map) == -1) {
            logger->error("Failed to load map file: {}", path);
            return;
        }

        // 使用配置参数 viz_leaf_size
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
        bbs3d_->set_tar_points(tar_points, config_.min_level_res, config_.max_level);
        bbs3d_->set_score_threshold_percentage(config_.score_threshold);
        bbs3d_->set_num_threads(16);

        map_ready_ = true;
        state_ = SystemState::COARSE_LOCALIZATION;
        logger->info("Global Map loaded. Algo points: {}, Viz points: {}, Viz Grid: {}m", 
                     map_algo->size(), map_viz_cloud_->size(), config_.viz_leaf_size);
    }

    void on_odom_tf_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        latest_T_odom_base_ = tf2::transformToEigen(*msg);
        tf_received_ = true;
    }

    void on_cloud_callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_odom_in, fins::AcqTime acq_time) {
        if (!map_ready_ || state_ == SystemState::IDLE || !is_node_active_ || !tf_received_) return;

        std::unique_lock<std::mutex> lock(data_mtx_);
        double cur_time = fins::to_seconds(acq_time);

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
                job_.T_odom_base = latest_T_odom_base_;
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
                cv_.wait(lock, [this] { return !is_worker_running_ || job_.has_new_job; });
                if (!is_worker_running_) break;
                current_job = job_;
                job_.has_new_job = false;
                is_processing_job_ = true;
            }

            if (state_ == SystemState::COARSE_LOCALIZATION) {
                perform_coarse_localization(current_job);
            } else if (state_ == SystemState::FINE_LOCALIZATION) {
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
        bbs3d_->set_trans_search_range(config_.min_search_xyz, config_.max_search_xyz);
        bbs3d_->set_angular_search_range(config_.min_search_rpy, config_.max_search_rpy);
        
        bbs3d_->localize();

        auto t_end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (bbs3d_->has_localized()) {
            logger->info("[3dBBS] Localized! Score: {} ({:.2f}%), Time: {:.2f}ms", 
                         bbs3d_->get_best_score(), 
                         bbs3d_->get_best_score_percentage() * 100.0, 
                         duration_ms);

            Eigen::Isometry3d T_map_base_bbs(bbs3d_->get_global_pose());
            T_map_base_bbs.linear() = Eigen::Quaterniond(T_map_base_bbs.linear()).normalized().toRotationMatrix();
            
            pcl::PointCloud<pcl::PointCovariance>::Ptr source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
                *cloud_base, config_.registered_leaf_size, config_.num_threads);
            
            small_gicp::estimate_covariances_omp(*source_down, 20, config_.num_threads);

            small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
            reg.reduction.num_threads = config_.num_threads;
            reg.rejector.max_dist_sq = config_.max_dist_sq_bbs_refine; 

            auto res = reg.align(*target_covs_, *source_down, *target_tree_, T_map_base_bbs);
            Eigen::Isometry3d T_map_base_final = res.converged ? res.T_target_source : T_map_base_bbs;

            {
                std::lock_guard<std::mutex> lock(data_mtx_);
                T_map_odom_ = T_map_base_final * job.T_odom_base.inverse();
                state_ = SystemState::FINE_LOCALIZATION;
                consecutive_gicp_failures_ = 0;
            }
            publish_results(job.ts, job.cloud_odom);
        } else {
            logger->warn("[3dBBS] Localization failed. Time: {:.2f}ms", duration_ms);
        }
    }

    void perform_fine_tracking(const JobData& job) {
        auto t_start = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointCovariance>::Ptr source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
            *job.cloud_odom, config_.registered_leaf_size, config_.num_threads);
        
        small_gicp::estimate_covariances_omp(*source_down, 20, config_.num_threads);

        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
        reg.reduction.num_threads = config_.num_threads;
        reg.rejector.max_dist_sq = config_.max_dist_sq_fine; 
        
        auto result = reg.align(*target_covs_, *source_down, *target_tree_, T_map_odom_);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // 使用配置中的 max_gicp_error
        if (result.converged && result.error < config_.max_gicp_error) {
            Eigen::Isometry3d T_delta = result.T_target_source * T_map_odom_.inverse();
            Eigen::Vector3d t_diff = T_delta.translation();
            Eigen::Vector3d rpy_diff = T_delta.linear().eulerAngles(0, 1, 2) * 180.0 / M_PI;

            logger->info("[GICP] Result: converged={}, error={:.2f}, iter={}, time={:.2f}ms | Delta: xyz[{:.3f} {:.3f} {:.3f}] rpy[{:.3f} {:.3f} {:.3f} deg]",
                         result.converged, result.error, result.iterations, duration_ms,
                         t_diff.x(), t_diff.y(), t_diff.z(),
                         rpy_diff.x(), rpy_diff.y(), rpy_diff.z());

            {
                std::lock_guard<std::mutex> lock(data_mtx_);
                T_map_odom_ = result.T_target_source;
                consecutive_gicp_failures_ = 0;
            }
            publish_results(job.ts, job.cloud_odom);
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

    void publish_results(fins::AcqTime ts, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_odom) {
        // 1. 发送 TF
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

        // 2. 发送对齐后的点云
        if (required("aligned_cloud")) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*cloud_odom, *aligned, T_map_odom_.matrix().cast<float>());
            aligned->header.frame_id = "map";
            send("aligned_cloud", aligned, ts);
        }

        // 3. 随着每个结果同步发送可视化全局地图
        if (map_ready_ && map_viz_cloud_ && required("global_map_viz")) {
            map_viz_cloud_->header.frame_id = "map";
            send("global_map_viz", map_viz_cloud_, ts);
        }
    }

    Config config_;
    SystemState state_{SystemState::IDLE};
    bool map_ready_{false};
    bool is_node_active_{false};
    bool tf_received_{false};
    
    Eigen::Isometry3d T_map_odom_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d latest_T_odom_base_{Eigen::Isometry3d::Identity()};
    
    int consecutive_gicp_failures_ = 0;
    bool accumulation_started_ = false;
    double start_accumulation_time_ = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_odom_;

    JobData job_;
    bool is_processing_job_ = false;
    std::mutex data_mtx_;
    std::condition_variable cv_;
    
    std::thread worker_thread_;
    std::atomic<bool> is_worker_running_{false};

    std::unique_ptr<cpu::BBS3D> bbs3d_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_viz_cloud_;
    pcl::PointCloud<pcl::PointCovariance>::Ptr target_covs_;
    std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>> target_tree_;
};

EXPORT_NODE(GlobalLocalizationNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)