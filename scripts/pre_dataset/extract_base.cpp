#include <time.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <pwd.h>
#include <cstdio>
#include <unistd.h>
#include <stdexcept>

#define ROUND_UP(x, y) (((x) + (y) - 1) / (y) * (y))

/// @brief 从fbin文件中加载向量
/// @tparam T
/// @param bin_file
/// @param offset_points 其实向量id
/// @param points_to_read 终止向量id
/// @return
template<typename T>
std::vector<std::vector<T>> load_aligned_bin_part(const std::string &bin_file,
                                                  size_t offset_points,
                                                  size_t points_to_read) {
  std::ifstream reader;
  try {
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
  } catch (const std::ios_base::failure &e) {
    // 错误处理提示
    std::cerr << "文件读取失败: " << e.what() << ", no file: " << bin_file
              << std::endl;
  }
  size_t actual_file_size = reader.tellg();
  reader.seekg(0, std::ios::beg);

  int npts_i32, dim_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));
  size_t npts = static_cast<size_t>(npts_i32);
  size_t dim = static_cast<size_t>(dim_i32);

  size_t expected_actual_file_size =
      npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
  if (actual_file_size != expected_actual_file_size) {
    std::stringstream stream;
    stream << "Error. File size mismatch. Actual size is " << actual_file_size
           << " while expected size is " << expected_actual_file_size
           << " npts = " << npts << " dim = " << dim
           << " size of <T>= " << sizeof(T) << std::endl;
    std::cout << stream.str();
    throw std::runtime_error(stream.str());
  }

  if (offset_points + points_to_read > npts) {
    std::stringstream stream;
    stream << "Error. Not enough points in file. Requested " << offset_points
           << " offset and " << points_to_read << " points, but have only "
           << npts << " points" << std::endl;
    std::cout << stream.str();
    throw std::runtime_error(stream.str());
  }

  reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

  const size_t rounded_dim = ROUND_UP(dim, 8);

  std::vector<std::vector<T>> data(points_to_read, std::vector<T>(rounded_dim));

  for (size_t i = 0; i < points_to_read; i++) {
    reader.read(reinterpret_cast<char *>(data[i].data()), dim * sizeof(T));

    memset(data[i].data() + dim, 0, (rounded_dim - dim) * sizeof(T));
  }

  reader.close();

  // Output the read data
  // std::cout << "Read " << points_to_read << " points. Data: " << std::endl;
  // for (size_t i = 0; i < 5; i++) {
  //   std::cout << "Point " << (i + offset_points) << ": ";
  //   for (size_t j = 0; j < dim; j++) {
  //     std::cout << data[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  //   // break;
  // }

  return data;  // Return the 2D vector containing the data
}

/// @brief 将指定数据写入fbin文件
/// @tparam T
/// @param bin_file
/// @param offset_points 其实向量id
/// @param points_to_read 终止向量id
/// @return
template<typename T>
void gen_vector(const std::string &out_file, std::vector<std::vector<T>> data,
                int dim_i32, size_t offset_points, size_t points_to_read) {
  std::ofstream writer;
  // Set to throw if failbit gets set
  writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);

  try {
    writer.open(out_file, std::ios::binary);
    // Further operations with writer...
  } catch (const std::ios_base::failure &e) {
    // Error handling
    std::cerr << "File write failed: " << e.what() << ", file: " << out_file
              << std::endl;
  }

  int npts_i32 = points_to_read - offset_points;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &dim_i32, sizeof(int));
  size_t npts = static_cast<size_t>(npts_i32);
  size_t dim = static_cast<size_t>(dim_i32);

  for (size_t i = offset_points; i < points_to_read; i++) {
    writer.write(reinterpret_cast<char *>(data[i].data()), dim * sizeof(T));
  }

  writer.close();
}

inline void get_bin_metadata_impl(std::basic_istream<char> &reader,
                                  size_t &nrows, size_t &ncols,
                                  size_t offset = 0) {
  int nrows_32, ncols_32;
  reader.seekg(offset, reader.beg);
  reader.read((char *) &nrows_32, sizeof(int));
  reader.read((char *) &ncols_32, sizeof(int));
  nrows = nrows_32;
  ncols = ncols_32;
}

/* FILE READER*/
/* nrows:numbers
 * ncols:dims
 */
inline void get_bin_metadata(const std::string &bin_file, size_t &nrows,
                             size_t &ncols, size_t offset = 0) {
  std::ifstream reader(bin_file.c_str(), std::ios::binary);
  get_bin_metadata_impl(reader, nrows, ncols, offset);
}

int main(int argc, char **argv) {
  int         top_K = 10;  // 需要关连修改: M_LOW 和 EF in hnsw.h
  std::string infile = argv[1];

  std::cout << "@file_name:" << infile << std::endl;
  size_t dims = -1;
  size_t numbers = -1;
  get_bin_metadata(infile, numbers, dims);
  std::cout << "@numbers:" << numbers << std::endl;
  std::cout << "@dims:" << dims << std::endl;

  std::vector<std::vector<float>> data =
      load_aligned_bin_part<float>(infile, 0, numbers);
  size_t data_size = numbers * dims * sizeof(float) / 1024 / 1024;  // MB
  std::cout << "load data done, data size(MB)=" << data_size << std::endl;

  // std::string data_name_suffix = "50w"; // siftsmall, sift, gist
  std::string write_file_name = argv[2];
  int         neednum = atoi(argv[3]) * 0.95;
  gen_vector(write_file_name, data, dims, 0, neednum);
  std::cout << " save to " << write_file_name << std::endl;

  return 0;
}