/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <fins/utils/time.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

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

    // ==============================================================================
    // 1. 数据结构定义
    // ==============================================================================
    struct Config {
        Eigen::Vector3d min_search_xyz{-100.0, -100.0, -2.0};
        Eigen::Vector3d max_search_xyz{100.0, 100.0, 2.0};
        Eigen::Vector3d min_search_rpy{0.0, 0.0, -M_PI};
        Eigen::Vector3d max_search_rpy{0.0, 0.0, M_PI};
        
        int num_threads = 8;
        double score_threshold = 0.3;
        double accumulation_time = 1.5;
        
        double min_level_res = 1.5;
        int max_level = 5;
        double global_leaf_size = 0.8;
        double registered_leaf_size = 0.5;
        
        double max_dist_sq_fine = 4.0;
        double max_dist_sq_bbs_refine = 25.0;
        int max_gicp_failures = 5;
        std::string voxel_dir = "";
    };

    struct State {
        SystemState mode{SystemState::IDLE};
        Eigen::Isometry3d T_map_odom{Eigen::Isometry3d::Identity()};
        int consecutive_gicp_failures = 0;
        
        bool accumulation_started = false;
        double start_accumulation_time = 0.0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_odom{new pcl::PointCloud<pcl::PointXYZI>()};
        
        geometry_msgs::msg::TransformStamped latest_T_odom_base;
        bool tf_received = false;
        
        std::atomic<bool> is_running{false};
        std::atomic<bool> node_running{false};
    };

    struct JobData {
        bool has_new_job = false;
        bool is_processing = false;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        fins::AcqTime ts;
        Eigen::Isometry3d T_odom_base;
    };

    // ==============================================================================
    // 2. FINS 框架生命周期接口
    // ==============================================================================
    void define() override {
        set_name("GlobalLocalization");
        set_description("Global Localization using 3dBBS and GICP with Refinement logic");
        set_category("Localization");

        register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &GlobalLocalizationNode::on_cloud_callback);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{odom}^{baselink}$", &GlobalLocalizationNode::on_odom_tf_callback);

        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("global_map_viz");
        register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("aligned_cloud");
        register_output<geometry_msgs::msg::TransformStamped>("$T_{map}^{odom}$");

        // 使用成员函数指针注册参数，解决 Lambda 和直接变量地址导致的编译错误
        register_parameter<std::string>("global_map_path", &GlobalLocalizationNode::on_map_path_changed, std::string(""));
        register_parameter<std::string>("voxel_map_dir", &GlobalLocalizationNode::on_voxel_dir_changed, std::string(""));
        register_parameter<int>("max_gicp_failures", &GlobalLocalizationNode::on_max_failures_changed, 5);
    }

    void initialize() override {
        logger->info("Initializing Global Localization Node...");
        bbs3d_ = std::make_unique<cpu::BBS3D>();
        state_.is_running = true;
        worker_thread_ = std::thread(&GlobalLocalizationNode::worker_loop, this);
    }

    void run() override { state_.node_running = true; logger->info("System Running"); }
    void pause() override { state_.node_running = false; }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(data_mtx_);
        state_.mode = SystemState::COARSE_LOCALIZATION;
        state_.accumulated_cloud_odom->clear();
        state_.accumulation_started = false;
        state_.consecutive_gicp_failures = 0;
        logger->info("Node reset to COARSE mode.");
    }

    ~GlobalLocalizationNode() {
        state_.is_running = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    // ==============================================================================
    // 3. 参数 Setter 函数
    // ==============================================================================
    void on_voxel_dir_changed(const std::string& dir) {
        config_.voxel_dir = dir;
        logger->info("Voxel directory set to: {}", dir);
    }

    void on_max_failures_changed(int val) {
        config_.max_gicp_failures = val;
        logger->info("Max GICP failures set to: {}", val);
    }

    // ==============================================================================
    // 4. 数据回调与前端累加逻辑
    // ==============================================================================
    void on_odom_tf_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        state_.latest_T_odom_base = *msg;
        state_.tf_received = true;
    }

    void on_cloud_callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_raw, fins::AcqTime acq_time) {
        if (state_.mode == SystemState::IDLE || !state_.tf_received || !state_.node_running) return;

        std::unique_lock<std::mutex> lock(data_mtx_);
        double cur_time = fins::to_seconds(acq_time);

        Eigen::Isometry3d T_odom_base = extract_pose_from_tf(state_.latest_T_odom_base);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_odom(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*cloud_raw, *cloud_odom, T_odom_base.matrix().cast<float>());

        if (!state_.accumulation_started) {
            state_.accumulation_started = true;
            state_.start_accumulation_time = cur_time;
            state_.accumulated_cloud_odom->clear();
        }

        auto temp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*cloud_odom, 0.2, config_.num_threads);
        *state_.accumulated_cloud_odom += *temp;

        if (cur_time - state_.start_accumulation_time > config_.accumulation_time) {
            if (!job_.is_processing) {
                job_.cloud = state_.accumulated_cloud_odom;
                job_.ts = acq_time;
                job_.T_odom_base = T_odom_base;
                job_.has_new_job = true;
                cv_.notify_one();
            }
            state_.accumulation_started = false;
            state_.accumulated_cloud_odom.reset(new pcl::PointCloud<pcl::PointXYZI>());
        }
    }

    // ==============================================================================
    // 5. 后台工作线程与算法分发
    // ==============================================================================
    void worker_loop() {
        while (state_.is_running) {
            JobData current_job;
            {
                std::unique_lock<std::mutex> lock(data_mtx_);
                cv_.wait(lock, [this] { return !state_.is_running || job_.has_new_job; });
                if (!state_.is_running) break;
                current_job = job_;
                job_.has_new_job = false;
                job_.is_processing = true;
            }

            if (state_.mode == SystemState::COARSE_LOCALIZATION) {
                process_coarse(current_job);
            } else if (state_.mode == SystemState::FINE_LOCALIZATION) {
                process_fine(current_job);
            }
            job_.is_processing = false;
        }
    }

    void process_coarse(const JobData& job) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*job.cloud, *cloud_base, job.T_odom_base.inverse().matrix().cast<float>());
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bbs(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*cloud_base, *cloud_bbs);
        
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_bbs);
        sor.setLeafSize(0.5f, 0.5f, 0.5f);
        sor.filter(*cloud_bbs);

        std::vector<Eigen::Vector3d> src_points;
        for (auto& p : cloud_bbs->points) src_points.push_back(p.getVector3fMap().cast<double>());

        bbs3d_->set_src_points(src_points);
        bbs3d_->set_trans_search_range(config_.min_search_xyz, config_.max_search_xyz);
        bbs3d_->set_angular_search_range(config_.min_search_rpy, config_.max_search_rpy);
        
        bbs3d_->localize();

        if (!bbs3d_->has_localized()) {
            logger->warn("[3dBBS] Localization failed.");
            return;
        }

        Eigen::Isometry3d T_map_base_bbs(bbs3d_->get_global_pose());
        T_map_base_bbs.linear() = Eigen::Quaterniond(T_map_base_bbs.linear()).normalized().toRotationMatrix();
        
        Eigen::Isometry3d T_map_base_refined = refine_with_gicp(cloud_base, T_map_base_bbs, config_.max_dist_sq_bbs_refine);
        
        state_.T_map_odom = T_map_base_refined * job.T_odom_base.inverse();
        state_.mode = SystemState::FINE_LOCALIZATION;
        state_.consecutive_gicp_failures = 0; 
        
        publish_results(job.ts, job.cloud);
    }

    void process_fine(const JobData& job) {
        auto source_down = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
            *job.cloud, config_.registered_leaf_size, config_.num_threads);
        small_gicp::estimate_covariances_omp(*source_down, 20, config_.num_threads);

        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
        reg.reduction.num_threads = config_.num_threads;
        reg.rejector.max_dist_sq = config_.max_dist_sq_fine; 
        
        auto result = reg.align(*target_covs_, *source_down, *target_tree_, state_.T_map_odom);
        
        if (result.converged && result.error < 60000.0) {
            state_.T_map_odom = result.T_target_source;
            state_.consecutive_gicp_failures = 0; 
            publish_results(job.ts, job.cloud);
        } else {
            state_.consecutive_gicp_failures++;
            if (state_.consecutive_gicp_failures >= config_.max_gicp_failures) {
                logger->error("[GICP] Reverting to COARSE.");
                state_.mode = SystemState::COARSE_LOCALIZATION;
            }
        }
    }

    Eigen::Isometry3d refine_with_gicp(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_base, const Eigen::Isometry3d& initial_guess, double max_dist_sq) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_map(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*cloud_base, *cloud_in_map, initial_guess.matrix().cast<float>());
        
        auto source_gicp = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
            *cloud_in_map, config_.registered_leaf_size, config_.num_threads);
        small_gicp::estimate_covariances_omp(*source_gicp, 20, config_.num_threads);

        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
        reg.reduction.num_threads = config_.num_threads;
        reg.rejector.max_dist_sq = max_dist_sq; 

        auto res = reg.align(*target_covs_, *source_gicp, *target_tree_, Eigen::Isometry3d::Identity());
        return res.converged ? (res.T_target_source * initial_guess) : initial_guess;
    }

    // ==============================================================================
    // 6. 地图处理与发布
    // ==============================================================================
    void on_map_path_changed(const std::string& path) {
        if (!fs::exists(path)) return;
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(path, *raw_map) == -1) return;

        auto map_viz = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*raw_map, 0.7);
        auto map_algo = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*map_viz, config_.global_leaf_size);
        
        target_covs_.reset(new pcl::PointCloud<pcl::PointCovariance>());
        target_covs_->points.resize(map_algo->points.size());
        for (size_t i = 0; i < map_algo->points.size(); ++i) target_covs_->points[i].getVector3fMap() = map_algo->points[i].getVector3fMap();
        small_gicp::estimate_covariances_omp(*target_covs_, 20, config_.num_threads);
        target_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target_covs_, small_gicp::KdTreeBuilderOMP(config_.num_threads));

        bool voxel_loaded = false;
        if (!config_.voxel_dir.empty() && fs::exists(config_.voxel_dir) && !fs::is_empty(config_.voxel_dir)) {
            if (bbs3d_->load_voxel_params(config_.voxel_dir) && bbs3d_->set_multi_buckets(config_.voxel_dir)) voxel_loaded = true;
        }
        if (!voxel_loaded) {
            std::vector<Eigen::Vector3d> tar_points;
            for (auto& p : map_algo->points) tar_points.push_back(p.getVector3fMap().cast<double>());
            bbs3d_->set_tar_points(tar_points, config_.min_level_res, config_.max_level);
            if (!config_.voxel_dir.empty()) {
                fs::create_directories(config_.voxel_dir);
                bbs3d_->save_voxel_params(config_.voxel_dir);
            }
        }
        bbs3d_->set_score_threshold_percentage(config_.score_threshold);
        bbs3d_->set_num_threads(16);

        send("global_map_viz", map_viz, fins::now());
        state_.mode = SystemState::COARSE_LOCALIZATION;
    }

    void publish_results(fins::AcqTime ts, const pcl::PointCloud<pcl::PointXYZI>::Ptr& query) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.frame_id = "map";
        tf.child_frame_id = "odom";
        tf.transform.translation.x = state_.T_map_odom.translation().x();
        tf.transform.translation.y = state_.T_map_odom.translation().y();
        tf.transform.translation.z = state_.T_map_odom.translation().z();
        Eigen::Quaterniond q(state_.T_map_odom.rotation());
        tf.transform.rotation.w = q.w(); tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y(); tf.transform.rotation.z = q.z();
        send("$T_{map}^{odom}$", tf, ts);

        if (required("aligned_cloud")) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*query, *aligned, state_.T_map_odom.matrix().cast<float>());
            send("aligned_cloud", aligned, ts);
        }
    }

    Eigen::Isometry3d extract_pose_from_tf(const geometry_msgs::msg::TransformStamped& msg) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z);
        T.linear() = Eigen::Quaterniond(msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z).toRotationMatrix();
        return T;
    }

    Config config_;
    State state_;
    JobData job_;
    std::mutex data_mtx_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::unique_ptr<cpu::BBS3D> bbs3d_;
    pcl::PointCloud<pcl::PointCovariance>::Ptr target_covs_;
    std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>> target_tree_;
};

EXPORT_NODE(GlobalLocalizationNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)