#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <index.h>
#include <cstddef>
#include <future>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define NUM_INSERT_THREADS 2
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 3
// #define NUM_INSERT_THREADS 1
// #define NUM_DELETE_THREADS 1
// #define NUM_SEARCH_THREADS 1
#define DeleteQPS 1000

int            begin_time = 0;
diskann::Timer globalTimer;
// read io info
static inline void get_io_info(std::string info = "") {
  static std::map<std::string, long long> previousDataMap;
  std::string                             pid = std::to_string(getpid());
  std::string ioStatsFilePath = "/proc/" + pid + "/io";

  std::ifstream                    file(ioStatsFilePath);
  std::map<std::string, long long> currentDataMap;

  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      size_t delimiterPos = line.find(':');
      if (delimiterPos != std::string::npos) {
        std::string key = line.substr(0, delimiterPos);
        std::string valueString = line.substr(delimiterPos + 1);
        long long   value = std::stoull(valueString);
        currentDataMap[key] = value;
      }
    }

    file.close();

    // print diff value
    for (const auto& entry : currentDataMap) {
      printf(" %s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
             entry.second / 1024.0 / 1024 / 1024);
      if (previousDataMap.count(entry.first) > 0) {
        long long diff = entry.second - previousDataMap[entry.first];
        printf(" @%s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
               diff / 1024.0 / 1024 / 1024);
      } else {
        printf(" @%s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
               entry.second / 1024.0 / 1024 / 1024);
      }
    }

    previousDataMap = currentDataMap;
  } else {
    std::cerr << "Failed to open file: " << ioStatsFilePath << std::endl;
  }
}

// acutually also shows disk size
void ShowMemoryStatus() {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int           tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  std::cout << "memory current time: " << current_time << " RSS : " << rss
            << " KB" << std::endl;
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string& truthFilePrefix, int vectorCount) {
  std::string fileName(truthFilePrefix);
  fileName += "-";
  if (vectorCount < 1000) {
    fileName += std::to_string(vectorCount);
  } else if (vectorCount < 1000000) {
    fileName += std::to_string(vectorCount / 1000);
    fileName += "k";
  } else if (vectorCount < 1000000000) {
    if (vectorCount % 1000000 == 0) {
      fileName += std::to_string(vectorCount / 1000000);
      fileName += "M";
    } else {
      float vectorCountM = ((float) vectorCount) / 1000000;
      fileName += convertFloatToString(vectorCountM, 2);
      fileName += "M";
    }
  } else {
    fileName += std::to_string(vectorCount / 1000000000);
    fileName += "B";
  }
  return fileName;
}

template<typename T>
inline uint64_t save_bin_test(const std::string& filename, T* id, float* dist,
                              size_t npts, size_t ndims, size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  diskann::cout << "Writing bin: " << filename.c_str() << std::endl;
  writer.seekp(offset, writer.beg);
  int    npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  writer.write((char*) &npts_i32, sizeof(int));
  writer.write((char*) &ndims_i32, sizeof(int));
  diskann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                << ", size = " << bytes_written << "B" << std::endl;

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char*) (id + i * ndims + j), sizeof(T));
      writer.write((char*) (dist + i * ndims + j), sizeof(float));
    }
  }
  writer.close();
  diskann::cout << "Finished writing bin." << std::endl;
  return bytes_written;
}

template<typename T, typename TagT>
void sync_search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                        const int recall_at, _u64 L,
                        diskann::MergeInsert<T, TagT>& sync_index,
                        std::string&                   truthset_file,
                        tsl::robin_set<TagT>& inactive_tags, int curCount,
                        bool merged, bool calRecall) {
  auto search_timer = std::chrono::high_resolution_clock::now();

  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  unsigned* gt_tags = nullptr;
  size_t    gt_num, gt_dim;

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
                           &gt_tags);
    std::cout << "load truthset over" << std::endl;
  }
  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double>  latency_stats(query_num, 0);
  diskann::QueryStats* stats = new diskann::QueryStats[query_num];
  std::string          recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(12) << "50 Latency"
            << std::setw(12) << "90 Latency" << std::setw(12) << "95 Latency"
            << std::setw(12) << "99 Latency" << std::setw(12) << "99.9 Latency"
            << std::setw(12) << recall_string << std::setw(12)
            << "Mean disk IOs" << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();

    stats[i].n_current_used = std::numeric_limits<double>::max();
    sync_index.search_sync(query + i * query_aligned_dim, recall_at, L,
                           query_result_tags + i * recall_at,
                           query_result_dists + i * recall_at, stats + i);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float                         qps = (query_num / diff.count());
  float                         recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    if (merged) {
      std::string cur_result_path =
          "/data/linsy/Cout/overall_res/"
          "result_overall_diskann" +
          std::to_string(current_time) + "merged.bin";
      save_bin_test<TagT>(cur_result_path, query_result_tags,
                          query_result_dists, query_num, recall_at);
    } else {
      std::string cur_result_path =
          "/data/linsy/Cout/overall_res/"
          "result_overall_diskann" +
          std::to_string(current_time) + ".bin";
      save_bin_test<TagT>(cur_result_path, query_result_tags,
                          query_result_dists, query_num, recall_at);
    }

    recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                       query_result_tags, recall_at, recall_at,
                                       inactive_tags);
  }

  auto search_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_search = search_time - search_timer;
  std::cout << "search current time: " << diff_search.count()
            << " search use time: " << diff.count() << " recall compute use "
            << diff_search.count() - diff.count() << std::endl;

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats& stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(),
                                        latency_stats.end(), 0)) /
                   (float) query_num
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.50 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
            << std::setw(12) << recall << std::setw(12) << mean_ios
            << std::endl;

  delete[] query_result_dists;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
void merge_kernel(diskann::MergeInsert<T, TagT>& sync_index,
                  std::string& save_path, uint32_t id_map = 0) {
  diskann::Timer merget;
  sync_index.final_merge(id_map);
  double mt = ((double) merget.elapsed()) / (1000000.0);
  std::cout << "Merge_kernel use " << mt << " s." << std::endl;
  std::cout << "_________________________Over_merge_kernel________________"
               "_________"
            << std::endl;
  std::cout << "Merge Done " << std::endl;
}

template<typename T, typename TagT>
void deletion_kernel(T* data_load, diskann::MergeInsert<T, TagT>& sync_index,
                     std::vector<TagT>& delete_vec, size_t aligned_dim,
                     _u64 L) {
  diskann::Timer      timer;
  size_t              npts = delete_vec.size();
  std::vector<double> delete_latencies(npts, 0);
  std::cout << "Begin Delete" << std::endl;
  std::atomic_size_t success(0);
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
  for (_s64 i = 0; i < (_s64) delete_vec.size(); i++) {
    diskann::Timer      delete_timer;
    diskann::QueryStats stats;
    sync_index.lazy_delete(delete_vec[i]);
    success++;
    delete_latencies[i] = ((double) delete_timer.elapsed());
    if (success == DeleteQPS) {
      // std::this_thread::sleep_for(std::chrono::seconds(1));
      success = 0;
    }
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(delete_latencies.begin(), delete_latencies.end());
  std::cout << "Deleted " << delete_vec.size() << " / " << success.load()
            << " points in " << time_secs << "s" << std::endl;
  std::cout << "Mem index deletion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile deletion time : "
            << delete_latencies[(size_t) (0.10 * ((double) npts))]
            << " microsec" << std::endl
            << "50th percentile deletion time : "
            << delete_latencies[(size_t) (0.5 * ((double) npts))] << " microsec"
            << std::endl
            << "90th percentile deletion time : "
            << delete_latencies[(size_t) (0.90 * ((double) npts))]
            << " microsec" << std::endl
            << "99th percentile deletion time : "
            << delete_latencies[(size_t) (0.99 * ((double) npts))]
            << " microsec" << std::endl
            << "99.9th percentile deletion time : "
            << delete_latencies[(size_t) (0.999 * ((double) npts))]
            << " microsec" << std::endl;
}

template<typename T, typename TagT>
void insertion_kernel(T* data_load, diskann::MergeInsert<T, TagT>& sync_index,
                      std::vector<TagT>& insert_vec, size_t aligned_dim) {
  diskann::Timer      timer;
  size_t              npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  std::cout << "Begin Insert" << std::endl;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    diskann::Timer insert_timer;
    sync_index.insert(data_load + aligned_dim * i, insert_vec[i]);
    insert_latencies[i] = ((double) insert_timer.elapsed());
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(insert_latencies.begin(), insert_latencies.end());
  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs
            << "s" << std::endl;
  std::cout << "Mem index insertion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile insertion time : "
            << insert_latencies[(size_t) (0.10 * ((double) npts))]
            << " microsec" << std::endl
            << "50th percentile insertion time : "
            << insert_latencies[(size_t) (0.5 * ((double) npts))] << " microsec"
            << std::endl
            << "90th percentile insertion time : "
            << insert_latencies[(size_t) (0.90 * ((double) npts))]
            << " microsec" << std::endl
            << "99th percentile insertion time : "
            << insert_latencies[(size_t) (0.99 * ((double) npts))]
            << " microsec" << std::endl
            << "99.9th percentile insertion time : "
            << insert_latencies[(size_t) (0.999 * ((double) npts))]
            << " microsec" << std::endl;
}

template<typename T, typename TagT = uint32_t>
void get_trace(tsl::robin_set<uint32_t>& inactive_set,
               std::vector<TagT>& delete_tags, std::vector<TagT>& insert_tags,
               std::string& file_name, T*& data_load, std::string& data_path,
               size_t& dim, size_t& rounded_dim) {
  /**Loading delete_tags & insert_tags & inactive_set*/
  std::cout << "Loading " << file_name << std::endl;
  std::ifstream base_reader;
  int           update_size;
  base_reader.open(file_name, std::ios::binary | std::ios::ate);
  base_reader.seekg(0, std::ios::beg);
  base_reader.read((char*) &update_size, sizeof(int));
  std::cout << "update_size:" << update_size << std::endl;
  delete_tags.clear();
  delete_tags.resize(update_size);
  insert_tags.clear();
  insert_tags.resize(update_size);
  base_reader.seekg(sizeof(int), std::ios::beg);
  base_reader.read((char*) delete_tags.data(), sizeof(int) * update_size);
  base_reader.seekg((update_size + 1) * sizeof(int), std::ios::beg);
  base_reader.read((char*) insert_tags.data(), sizeof(int) * update_size);
  delete[] data_load;

  diskann::cout << "Reading trace vectors from bin file " << data_path << "..."
                << std::flush;
  std::ifstream reader(data_path, std::ios::binary | std::ios::ate);
  int           npts_i32, dim_i32;
  reader.seekg(0);
  reader.read((char*) &npts_i32, sizeof(int));
  reader.read((char*) &dim_i32, sizeof(int));
  dim = (unsigned) dim_i32;
  rounded_dim = ROUND_UP(dim, 8);
  size_t allocSize = update_size * rounded_dim * sizeof(T);

  diskann::alloc_aligned(((void**) &data_load), allocSize, 8 * sizeof(T));

  for (size_t i = 0; i < update_size; i++) {
    reader.seekg(2 * sizeof(int) + dim * sizeof(T) * insert_tags[i],
                 reader.beg);
    reader.read((char*) (data_load + i * rounded_dim), dim * sizeof(T));
    memset(data_load + i * rounded_dim + dim, 0,
           (rounded_dim - dim) * sizeof(T));
  }

  std::cout << "Loaded full data for driver: (" << update_size << "," << dim
            << ") vectors." << std::endl;
}

template<typename T, typename TagT>
void update(const std::string& data_path, const unsigned L_mem,
            const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
            const unsigned R_disk, const float alpha_disk, int step,
            const size_t base_num, const unsigned num_pq_chunks,
            const unsigned nodes_to_cache, std::string& save_path,
            const std::string& query_file, std::string& truthset_file,
            const int recall_at, _u64 Lsearch, const unsigned beam_width,
            std::string& trace_file_prefix, diskann::Distance<T>* dist_cmp,
            uint32_t id_map = 0) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 160);
  paras.Set<unsigned>("beamwidth", beam_width);
  // paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_search_threads", NUM_SEARCH_THREADS);
  bool   firstMerge = true;
  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::Timer timer;
  // dim = 128;
  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);
  std::cout << "Loaded full data for driver: (" << num_points << "," << dim
            << ") vectors." << std::endl;

  diskann::Metric               metric = diskann::Metric::L2;
  diskann::MergeInsert<T, TagT> sync_index(paras, dim, save_path + "_mem",
                                           save_path, save_path + "_merge",
                                           dist_cmp, metric, false, save_path);
  if (id_map == 2) {
    sync_index._disk_index_prefix_out = save_path;
  }

  std::cout << "Loading queries " << std::endl;
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  std::cout << "Searching before inserts: " << std::endl;
  tsl::robin_set<TagT> inactive_tags;
  // truthset_file="/data/linsy/dataset/deep/gt/deep_gt_K10_"
  std::string currentFileName = truthset_file + std::to_string(0) + ".fbin";
  std::cout << "Current_GT_File: " << currentFileName << std::endl;
  begin_time = globalTimer.elapsed() / 1.0e6f;
  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                     sync_index, currentFileName, inactive_tags, base_num,
                     false, true);

  int               batch = step;
  int               inMmeorySize = 0;
  int               res = base_num;
  std::future<void> merge_future;

  for (int i = 0; i < batch; i++) {
    std::cout << "Batch: " << i << " Total Batch : " << step << std::endl;

    diskann::Timer        batch_timer;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    std::string trace_file_name = trace_file_prefix + std::to_string(i);
    std::string all_data = data_path;
    get_trace<T, TagT>(inactive_tags, delete_vec, insert_vec, trace_file_name,
                       data_load, all_data, dim, aligned_dim);

    deletion_kernel(data_load, sync_index, delete_vec, aligned_dim, L_mem);
    std::cout << "_________________________Over_deletion_kernel________________"
                 "_________"
              << std::endl;

    insertion_kernel(data_load, sync_index, insert_vec, aligned_dim);
    std::cout << "_________________________Over_insertion_kernel_______________"
                 "__________"
              << std::endl;

    inMmeorySize += insert_vec.size();

    if (inMmeorySize >= MERGE_TH) {
      get_io_info("begin");
      std::cout << "Begin Merge" << std::endl;
      merge_kernel<T, TagT>(sync_index, save_path, id_map);
      get_io_info("end");
      std::cout << "IO_all" << std::endl;
      inMmeorySize = 0;
    }
    double e2e_time = ((double) batch_timer.elapsed()) / (1000000.0);
    diskann::cout << "Batch #" << i << " use " << e2e_time << " s."
                  << std::endl;
    auto copy_file = [](const std::string& src, const std::string& dest) {
      diskann::Timer copytimer;
      diskann::cout << "COPY :: " << src << " --> " << dest << "\n";
      std::ofstream dest_writer(dest, std::ios::binary);
      std::ifstream src_reader(src, std::ios::binary);
      dest_writer << src_reader.rdbuf();
      dest_writer.close();
      src_reader.close();
      double e2e_time = ((double) copytimer.elapsed()) / (1000000.0);
      std::cout << "copy_file_time: " << e2e_time << std::endl;
    };

    std::string currentFileName =
        truthset_file + std::to_string(i + 1) + ".fbin";
    std::cout << "Current_GT_File: " << currentFileName << std::endl;
    if ((i + 1) % 10 == 0)
      sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                         Lsearch, sync_index, currentFileName, inactive_tags,
                         res, true, true);
  }
  std::cout << "Update over" << std::endl;
  delete[] data_load;
}

template<typename T, typename TagT>
void build(const std::string& data_path, const unsigned L_mem,
           const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
           const unsigned R_disk, const float alpha_disk,
           const size_t num_start, const size_t num_shards,
           const unsigned num_pq_chunks, const unsigned nodes_to_cache,
           const std::string& save_path) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 500);
  paras.Set<unsigned>("beamwidth", 5);
  paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::cout << "Loaded full data for driver." << std::endl;
  // diskann::SyncIndex<T, TagT> sync_index(num_poiznts + 5000, dim,
  // num_shards,
  //                                        paras, 2, save_path);
  std::cout << "Ran constructor." << std::endl;
  std::vector<TagT> tags(num_start);
  std::iota(tags.begin(), tags.end(), 0);
  diskann::Timer timer;
  // sync_index.build(data_path.c_str(), num_start, tags);
  std::cout << "Sync Index build time: " << timer.elapsed() / 1000000 << "s\n";

  std::string tag_file = save_path + "_disk.index.tags";

  diskann::save_bin<TagT>(tag_file, tags.data(), num_start, 1);
  delete[] data_load;
}

int main(int argc, char** argv) {
  if (argc < 14) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <base_data_file> <L_mem> <R_mem> "
                 "<alpha_mem>"
              << " <L_disk> <R_disk> <alpha_disk>"
              << " <num_start> <num_shards> <#pq_chunks> <#nodes_to_cache>"
              << " <save_graph_file> <update> <build> <full_data_file> "
                 "<query_file> <truthset_file> <recall@>"
              << " <Lsearch> <#beam_width>"
              << " <trace_file_prefix>"
              << " <step>" << std::endl;
    exit(-1);
  }

  int         arg_no = 3;
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) std::atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) std::atof(argv[arg_no++]);
  size_t      num_start = (size_t) std::atoi(argv[arg_no++]);
  size_t      num_shards = (size_t) std::atoi(argv[arg_no++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[arg_no++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);
  bool        updateIndex = false;
  bool        buildIndex = false;

  if (std::string(argv[arg_no++]) == std::string("true"))
    updateIndex = true;
  if (std::string(argv[arg_no++]) == std::string("true"))
    buildIndex = true;

  std::string full_data_path(argv[arg_no++]);
  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int         recall_at = (int) std::atoi(argv[arg_no++]);
  _u64        Lsearch = std::atoi(argv[arg_no++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[arg_no++]);
  std::string trace_file_prefix(argv[arg_no++]);
  int         step = (int) std::atoi(argv[arg_no++]);

  uint32_t id_map = std::atoi(argv[24]);

  if (!updateIndex || buildIndex) {
    if (std::string(argv[1]) == std::string("int8"))
      build<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                              alpha_disk, num_start, num_shards, num_pq_chunks,
                              nodes_to_cache, save_path);
    else if (std::string(argv[1]) == std::string("uint8"))
      build<uint8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                               alpha_disk, num_start, num_shards, num_pq_chunks,
                               nodes_to_cache, save_path);
    else if (std::string(argv[1]) == std::string("float"))
      build<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                             alpha_disk, num_start, num_shards, num_pq_chunks,
                             nodes_to_cache, save_path);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  }

  if (!updateIndex)
    return 0;

  if (updateIndex) {
    if (std::string(argv[1]) == std::string("int8")) {
      diskann::DistanceL2Int8 dist_cmp;
      update<int8_t, unsigned>(
          full_data_path, L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
          step, num_start, num_pq_chunks, nodes_to_cache, save_path, query_file,
          truthset, recall_at, Lsearch, beam_width, trace_file_prefix,
          &dist_cmp, id_map);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      diskann::DistanceL2UInt8 dist_cmp;
      update<uint8_t, unsigned>(
          full_data_path, L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
          step, num_start, num_pq_chunks, nodes_to_cache, save_path, query_file,
          truthset, recall_at, Lsearch, beam_width, trace_file_prefix,
          &dist_cmp, id_map);
    } else if (std::string(argv[1]) == std::string("float")) {
      diskann::DistanceL2 dist_cmp;
      update<float, unsigned>(full_data_path, L_mem, R_mem, alpha_mem, L_disk,
                              R_disk, alpha_disk, step, num_start,
                              num_pq_chunks, nodes_to_cache, save_path,
                              query_file, truthset, recall_at, Lsearch,
                              beam_width, trace_file_prefix, &dist_cmp, id_map);
    } else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  }
}