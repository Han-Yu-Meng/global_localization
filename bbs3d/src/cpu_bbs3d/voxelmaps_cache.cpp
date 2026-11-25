// 加权体素图二进制缓存。
//
// 与上游 voxelmaps_io.cpp 的 PCD 缓存不同：这里逐字保存 Buckets 的
// Eigen::Vector4i（x, y, z, w），其中 .w() 是打分权重（core=2 / neighbor=1），
// w=0 是空槽哨兵。上游 PCD 缓存只写 xyz、读回把 w 置 1，会把 core 降权、
// 把空槽变成假点，破坏加权 BBS 的分数语义。
//
// 缓存元数据里记录源 map.pcd 的 size 与 mtime_ns；加载时比对，不一致即失效，
// 由调用方回退到 create_voxelmaps() 重建。这样磁盘地图更新后（例如建图落盘），
// 即使缓存文件还在，也不会拿旧体素图去匹配新地图。

#include <cpu_bbs3d/bbs3d.hpp>
#include <cpu_bbs3d/voxelmaps.hpp>

#include <filesystem>
#include <system_error>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {
constexpr char kCacheFileName[] = "voxelmaps.weighted.bin";
constexpr uint32_t kCacheMagic = 0x42425333u;  // "BBS3"
constexpr uint32_t kCacheVersion = 1u;

int64_t source_mtime_ns(const std::string& path) {
  std::error_code ec;
  fs::file_time_type ft = fs::last_write_time(path, ec);
  if (ec) return -1;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             ft.time_since_epoch()).count();
}

uint64_t source_size(const std::string& path) {
  std::error_code ec;
  uintmax_t s = fs::file_size(path, ec);
  return ec ? 0ull : static_cast<uint64_t>(s);
}
}  // namespace

namespace cpu {

bool BBS3D::save_voxelmaps_cache(const std::string& folder_path,
                                 const std::string& source_map_path) {
  if (!voxelmaps_ptr_) return false;
  std::error_code ec;
  if (!fs::exists(folder_path, ec)) {
    if (!fs::create_directories(folder_path, ec)) {
      std::cerr << "[3dBBS cache] cannot create dir: " << folder_path << std::endl;
      return false;
    }
  }
  const std::string cache_path = folder_path + "/" + kCacheFileName;
  std::ofstream ofs(cache_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "[3dBBS cache] cannot open for write: " << cache_path << std::endl;
    return false;
  }

  const int max_level = voxelmaps_ptr_->get_max_level();
  const uint64_t n_levels = static_cast<uint64_t>(max_level + 1);
  const int64_t mtime_ns = source_mtime_ns(source_map_path);
  const uint64_t sz = source_size(source_map_path);

  const double min_res = voxelmaps_ptr_->get_min_res();
  const double v_rate = v_rate_;
  const double neighbor_score = neighbor_score_;
  const int bucket_scan = voxelmaps_ptr_->get_max_bucket_scan_count();

  ofs.write(reinterpret_cast<const char*>(&kCacheMagic), sizeof(kCacheMagic));
  ofs.write(reinterpret_cast<const char*>(&kCacheVersion), sizeof(kCacheVersion));
  ofs.write(reinterpret_cast<const char*>(&n_levels), sizeof(n_levels));
  ofs.write(reinterpret_cast<const char*>(&min_res), sizeof(min_res));
  ofs.write(reinterpret_cast<const char*>(&v_rate), sizeof(v_rate));
  ofs.write(reinterpret_cast<const char*>(&neighbor_score), sizeof(neighbor_score));
  ofs.write(reinterpret_cast<const char*>(&bucket_scan), sizeof(bucket_scan));
  ofs.write(reinterpret_cast<const char*>(&mtime_ns), sizeof(mtime_ns));
  ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  const uint32_t path_len = static_cast<uint32_t>(source_map_path.size());
  ofs.write(reinterpret_cast<const char*>(&path_len), sizeof(path_len));
  ofs.write(source_map_path.data(), path_len);

  for (uint64_t i = 0; i < n_levels; ++i) {
    const double res = voxelmaps_ptr_->voxelmaps_res_[i];
    ofs.write(reinterpret_cast<const char*>(&res), sizeof(res));
    const auto& buckets = voxelmaps_ptr_->multi_buckets_[i];
    const uint64_t cnt = static_cast<uint64_t>(buckets.size());
    ofs.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
    for (const auto& b : buckets) {
      const int32_t v[4] = {b.x(), b.y(), b.z(), b.w()};
      ofs.write(reinterpret_cast<const char*>(v), sizeof(v));
    }
  }

  ofs.close();
  if (!ofs) {
    std::cerr << "[3dBBS cache] write failed: " << cache_path << std::endl;
    return false;
  }
  std::cout << "[3dBBS cache] saved weighted voxelmaps: " << cache_path
            << " (levels=" << n_levels << ", src_size=" << sz << ")" << std::endl;
  return true;
}

bool BBS3D::load_voxelmaps_cache(const std::string& folder_path,
                                 const std::string& source_map_path,
                                 double expected_min_res,
                                 int expected_max_level,
                                 double expected_neighbor_score) {
  if (folder_path.empty()) return false;
  const std::string cache_path = folder_path + "/" + kCacheFileName;
  std::ifstream ifs(cache_path, std::ios::binary);
  if (!ifs) return false;

  uint32_t magic = 0, version = 0;
  uint64_t n_levels = 0;
  double min_res = 0.0, v_rate = 0.0, neighbor_score = 0.0;
  int bucket_scan = 0;
  int64_t mtime_ns = -2;
  uint64_t sz = 0;
  uint32_t path_len = 0;

  auto read = [&ifs](void* dst, size_t n) -> bool {
    ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<size_t>(ifs.gcount()) == n;
  };

  if (!read(&magic, sizeof(magic)) || magic != kCacheMagic) return false;
  if (!read(&version, sizeof(version)) || version != kCacheVersion) return false;
  if (!read(&n_levels, sizeof(n_levels)) || n_levels == 0) return false;
  if (!read(&min_res, sizeof(min_res))) return false;
  if (!read(&v_rate, sizeof(v_rate))) return false;
  if (!read(&neighbor_score, sizeof(neighbor_score))) return false;
  if (!read(&bucket_scan, sizeof(bucket_scan))) return false;
  if (!read(&mtime_ns, sizeof(mtime_ns))) return false;
  if (!read(&sz, sizeof(sz))) return false;
  if (!read(&path_len, sizeof(path_len))) return false;

  std::string cached_path(path_len, '\0');
  if (path_len && !read(&cached_path[0], path_len)) return false;

  // 源地图签名：路径 / size / mtime 必须全部一致
  if (cached_path != source_map_path) return false;
  if (sz == 0 || sz != source_size(source_map_path)) return false;
  if (mtime_ns != source_mtime_ns(source_map_path)) return false;

  // 体素图参数签名：分辨率 / 层数 / 邻居权重变了就必须重建
  if (std::abs(min_res - expected_min_res) > 1e-9) return false;
  if (static_cast<int>(n_levels) - 1 != expected_max_level) return false;
  if (std::abs(neighbor_score - expected_neighbor_score) > 1e-9) return false;

  // ---- payload ----
  std::unique_ptr<VoxelMaps> vm(new VoxelMaps);
  vm->set_min_res(min_res);
  vm->set_max_level(static_cast<int>(n_levels) - 1);
  vm->set_max_bucket_scan_count(bucket_scan);
  vm->voxelmaps_res_.resize(n_levels);
  vm->multi_buckets_.resize(n_levels);

  for (uint64_t i = 0; i < n_levels; ++i) {
    double res = 0.0;
    uint64_t cnt = 0;
    if (!read(&res, sizeof(res))) return false;
    if (!read(&cnt, sizeof(cnt))) return false;
    vm->voxelmaps_res_[i] = res;
    auto& buckets = vm->multi_buckets_[i];
    buckets.resize(cnt);
    for (uint64_t j = 0; j < cnt; ++j) {
      int32_t v[4] = {0, 0, 0, 0};
      if (!read(v, sizeof(v))) return false;
      buckets[j] = Eigen::Vector4i(v[0], v[1], v[2], v[3]);
    }
  }
  if (!ifs && !ifs.eof()) return false;

  set_voxel_expantion_rate(v_rate);
  set_neighbor_score(neighbor_score);
  voxelmaps_ptr_ = std::move(vm);

  std::cout << "[3dBBS cache] loaded weighted voxelmaps from cache: " << cache_path
            << " (levels=" << n_levels << ")" << std::endl;
  return true;
}

}  // namespace cpu
