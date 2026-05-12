/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <fins/agent/parameter_server.hpp>
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
        std::atomic<bool> map_ready{false}; // 新增标志位
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
    }

    void initialize() override {
        logger->info("Initializing Global Localization Node...");
        std::cout << "Initialized" << std::endl;
        
        // 使用 ParamLoader 读取参数
        fins::ParamLoader config("GlobalLocalization");
        config_.min_search_xyz = config.get("min_search_xyz", Eigen::Vector3d{-100.0, -100.0, -2.0});
        config_.max_search_xyz = config.get("max_search_xyz", Eigen::Vector3d{100.0, 100.0, 2.0});
        config_.min_search_rpy = config.get("min_search_rpy", Eigen::Vector3d{0.0, 0.0, -M_PI});
        config_.max_search_rpy = config.get("max_search_rpy", Eigen::Vector3d{0.0, 0.0, M_PI});
        config_.num_threads = config.get("num_threads", 8);
        config_.score_threshold = config.get("score_threshold", 0.3);
        config_.accumulation_time = config.get("accumulation_time", 1.5);
        config_.min_level_res = config.get("min_level_res", 1.5);
        config_.max_level = config.get("max_level", 5);
        config_.global_leaf_size = config.get("global_leaf_size", 0.8);
        config_.registered_leaf_size = config.get("registered_leaf_size", 0.5);
        config_.max_dist_sq_fine = config.get("max_dist_sq_fine", 4.0);
        config_.max_dist_sq_bbs_refine = config.get("max_dist_sq_bbs_refine", 25.0);
        config_.max_gicp_failures = config.get("max_gicp_failures", 5);
        config_.voxel_dir = config.get("voxel_dir", std::string(""));
        
        bbs3d_ = std::make_unique<cpu::BBS3D>();
        
        std::string global_map_path = config.get("map_dir", std::string(""));
        logger->info("Map path from config: {}", global_map_path.empty() ? "empty" : global_map_path);
        if (!global_map_path.empty()) {
            load_map(global_map_path);
        } else {
            logger->warn("No map path specified in configuration. Map loading skipped.");
        }
        
        state_.is_running = true;
        worker_thread_ = std::thread(&GlobalLocalizationNode::worker_loop, this);
    }

    void run() override { state_.node_running = true; logger->info("System Running"); }
    void pause() override { state_.node_running = false; }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(data_mtx_);
        if (state_.map_ready) {
            state_.mode = SystemState::COARSE_LOCALIZATION;
        } else {
            state_.mode = SystemState::IDLE;
        }
        state_.accumulated_cloud_odom->clear();
        state_.accumulation_started = false;
        state_.consecutive_gicp_failures = 0;
        logger->info("Node reset to {} mode.", state_.map_ready ? "COARSE" : "IDLE");
    }

    ~GlobalLocalizationNode() {
        state_.is_running = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    // ==============================================================================
    // 3. 地图加载函数
    // ==============================================================================
    void load_map(const std::string& path) {
        state_.map_ready = false; // 进来先锁死
        
        logger->info("Loading map from: {}", path);
        if (!fs::exists(path)) {
            logger->error("Map file does not exist: {}", path);
            return;
        }
        
        // Check if bbs3d_ is initialized, if not, initialize it
        if (!bbs3d_) {
            bbs3d_ = std::make_unique<cpu::BBS3D>();
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(path, *raw_map) == -1) {
            logger->error("Failed to load PCD file: {}", path);
            return;
        }
        
        logger->info("Raw map loaded with {} points", raw_map->points.size());

        auto map_viz = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*raw_map, 0.7);
        auto map_algo = small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointXYZI>>(*map_viz, config_.global_leaf_size);
        
        logger->info("Map downsampled: viz {} points, algo {} points", map_viz->points.size(), map_algo->points.size());
        
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
            logger->info("Building voxel grid in {}.", config_.voxel_dir);
            std::vector<Eigen::Vector3d> tar_points;
            for (auto& p : map_algo->points) tar_points.push_back(p.getVector3fMap().cast<double>());
            bbs3d_->set_tar_points(tar_points, config_.min_level_res, config_.max_level);
            if (!config_.voxel_dir.empty()) {
                fs::create_directories(config_.voxel_dir);
                // 1. 保存配置参数
                bbs3d_->save_voxel_params(config_.voxel_dir); 
                // 2. 关键：保存具体的多层级体素桶数据
                // 注意：set_multi_buckets 会自动保存数据，不需要单独的save函数
                logger->info("Voxel map saved to {}", config_.voxel_dir);
            }
        }
        bbs3d_->set_score_threshold_percentage(config_.score_threshold);
        bbs3d_->set_num_threads(16);

        // Set map_viz coordinate frame to 'map'
        map_viz->header.frame_id = "map";
        // map_viz->header.stamp = fins::to_nanos(fins::now());
        
        logger->info("BBS3D search range: XYZ [{}, {}, {}] to [{}, {}, {}]", 
                   config_.min_search_xyz.x(), config_.min_search_xyz.y(), config_.min_search_xyz.z(),
                   config_.max_search_xyz.x(), config_.max_search_xyz.y(), config_.max_search_xyz.z());
        logger->info("BBS3D angular range: RPY [{}, {}, {}] to [{}, {}, {}]", 
                   config_.min_search_rpy.x(), config_.min_search_rpy.y(), config_.min_search_rpy.z(),
                   config_.max_search_rpy.x(), config_.max_search_rpy.y(), config_.max_search_rpy.z());

        send("global_map_viz", map_viz, fins::now());
        state_.map_ready = true; // 全部完成后才准许定位
        state_.mode = SystemState::COARSE_LOCALIZATION;
        logger->info("Map loaded and BBS3D initialized. Map ready for localization.");
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
        if (!state_.map_ready || state_.mode == SystemState::IDLE) return;
        
        if (!state_.tf_received || !state_.node_running) return;

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
        // 增加安全检查
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            if (!state_.map_ready || !bbs3d_) {
                logger->warn("BBS3D skipped: Map not initialized.");
                return;
            }
        }

        logger->debug("Processing coarse localization with {} input points", job.cloud->points.size());

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*job.cloud, *cloud_base, job.T_odom_base.inverse().matrix().cast<float>());
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bbs(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*cloud_base, *cloud_bbs);
        
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_bbs);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);
        sor.filter(*cloud_bbs);

        std::vector<Eigen::Vector3d> src_points;
        for (auto& p : cloud_bbs->points) src_points.push_back(p.getVector3fMap().cast<double>());
        
        logger->debug("BBS3D input: {} points after voxel filtering", src_points.size());

        // 建议对 bbs3d_ 的调用块加锁，或者确认此处的原子性
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
        
        // Add score and percentage logging like ROS2 version
        logger->info("[3dBBS] Localized! Score: {} ({:.2f})", 
                   bbs3d_->get_best_score(), bbs3d_->get_best_score_percentage() * 100.0);

        // Convert rotation matrix to quaternion, then to Euler angles
        Eigen::Quaterniond q_bbs(T_map_base_bbs.linear());
        Eigen::Vector3d rpy_bbs = q_bbs.toRotationMatrix().eulerAngles(0, 1, 2);
        
        logger->info("[3dBBS] Coarse pose: xyz=[{:.3f}, {:.3f}, {:.3f}], rpy=[{:.3f}, {:.3f}, {:.3f}]",
                   T_map_base_bbs.translation().x(), T_map_base_bbs.translation().y(), T_map_base_bbs.translation().z(),
                   rpy_bbs.x(), rpy_bbs.y(), rpy_bbs.z());
        
        Eigen::Isometry3d T_map_base_refined = refine_with_gicp(cloud_base, T_map_base_bbs, config_.max_dist_sq_bbs_refine);
        
        // Convert refined rotation matrix to quaternion, then to Euler angles
        Eigen::Quaterniond q_refined(T_map_base_refined.linear());
        Eigen::Vector3d rpy_refined = q_refined.toRotationMatrix().eulerAngles(0, 1, 2);
        
        logger->info("[GICP] Refined pose: xyz=[{:.3f}, {:.3f}, {:.3f}], rpy=[{:.3f}, {:.3f}, {:.3f}]",
                   T_map_base_refined.translation().x(), T_map_base_refined.translation().y(), T_map_base_refined.translation().z(),
                   rpy_refined.x(), rpy_refined.y(), rpy_refined.z());
        
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
    // 6. 结果发布
    // ==============================================================================

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