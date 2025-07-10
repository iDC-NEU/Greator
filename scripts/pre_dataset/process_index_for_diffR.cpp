#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <string>

#define SECTOR_SIZE 4096
#define R 32

// Define the metadata structure and node structure
struct Metadata {
  int      meta_npts;
  int      meta_ndims;
  uint64_t npts;
  uint64_t ndims;
  uint64_t medoid;
  uint64_t max_node_len;
  uint64_t nnodes_per_sector;
  uint64_t vamana_frozen_num;
  uint64_t vamana_frozen_loc;
  uint64_t append_reorder_data;
  uint64_t n_sectors_add1;
  uint64_t ndims_reorder_file;
  uint64_t n_data_nodes_per_sector;
  uint64_t disk_index_file_size;
};

// 节点信息结构体
template<typename T>
struct Node {
  std::vector<T>        node_data;  // 节点数据，模板T 类型
  uint32_t              nnbrs;      // 邻居数量，uint32_t 类型
  std::vector<uint32_t> nhood;      // 邻居列表，固定大小 R，每个点为 uint32_t

  Node(size_t ndims, int tr) : node_data(ndims, 0), nnbrs(0), nhood(tr, 0) {
  }  // 初始化，节点数据为 0，邻居列表为 0
};

// Function to read metadata and nodes from the disk index file
template<typename T>
void read_disk_index(const std::string     &file_name,
                     std::vector<Metadata> &metadata_vector,
                     std::vector<Node<T>> &node_vector, int tr,
                     std::string data_type) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file: " << file_name << std::endl;
    return;
  }

  // Read metadata (assuming metadata is at the beginning of the file)
  int meta_npts, meta_ndims;
  file.read(reinterpret_cast<char *>(&meta_npts), sizeof(int));
  file.read(reinterpret_cast<char *>(&meta_ndims), sizeof(int));

  uint64_t npts, ndims;
  file.read(reinterpret_cast<char *>(&npts), sizeof(uint64_t));
  file.read(reinterpret_cast<char *>(&ndims), sizeof(uint64_t));

  uint64_t medoid, width_u32, max_node_len, nnodes_per_sector,
      vamana_frozen_num, vamana_frozen_loc, append_reorder_data;
  uint64_t n_sectors_add1 = 0, ndims_reorder_file = 0,
           n_data_nodes_per_sector = 0;
  uint64_t disk_index_file_size;
  file.read(reinterpret_cast<char *>(&medoid), sizeof(uint64_t));  // width
  file.read(reinterpret_cast<char *>(&max_node_len),
            sizeof(uint64_t));  // medoid
  file.read(reinterpret_cast<char *>(&nnodes_per_sector),
            sizeof(uint64_t));  // max_node_len
  file.read(reinterpret_cast<char *>(&vamana_frozen_num),
            sizeof(uint64_t));  // nnodes_per_sector
  file.read(reinterpret_cast<char *>(&vamana_frozen_loc), sizeof(uint64_t));
  file.read(reinterpret_cast<char *>(&append_reorder_data), sizeof(uint64_t));
  if (append_reorder_data) {
    file.read(reinterpret_cast<char *>(&n_sectors_add1), sizeof(uint64_t));
    file.read(reinterpret_cast<char *>(&ndims_reorder_file), sizeof(uint64_t));
    file.read(reinterpret_cast<char *>(&n_data_nodes_per_sector),
              sizeof(uint64_t));
  }
  file.read(reinterpret_cast<char *>(&disk_index_file_size), sizeof(uint64_t));

  Metadata meta;
  meta.meta_npts = meta_npts;
  meta.meta_ndims = meta_ndims;
  meta.npts = npts;
  meta.ndims = ndims;
  meta.medoid = medoid;
  meta.max_node_len = max_node_len;
  meta.nnodes_per_sector = nnodes_per_sector;
  meta.vamana_frozen_num = vamana_frozen_num;
  meta.vamana_frozen_loc = vamana_frozen_loc;
  meta.append_reorder_data = append_reorder_data;
  if (append_reorder_data) {
    meta.n_sectors_add1 = n_sectors_add1;
    meta.ndims_reorder_file = ndims_reorder_file;
    meta.n_data_nodes_per_sector = n_data_nodes_per_sector;
  }
  meta.disk_index_file_size = disk_index_file_size;

  metadata_vector.push_back(meta);

  uint64_t n_sectors = npts / nnodes_per_sector;
  if (npts % nnodes_per_sector)
    n_sectors++;
  uint64_t nodes_read = 0;
  for (uint64_t sector = 1; sector <= n_sectors && nodes_read < npts;
       sector++) {
    if (sector % 100000 == 0) {
      std::cout << "Sector #" << sector << "read" << std::endl;
    }
    // Read data for the current sector
    uint64_t          sector_size = SECTOR_SIZE;
    std::vector<char> sector_buffer(sector_size);
    // Move the file pointer to the correct position for the current sector
    file.seekg(sector * SECTOR_SIZE, std::ios::beg);
    file.read(sector_buffer.data(), sector_size);
    if (!file) {
      std::cerr << "Error reading sector " << sector << " from file!"
                << std::endl;
      break;
    }

    // Process the sector data (reading nodes)
    for (uint64_t i = 0; i < nnodes_per_sector && nodes_read < npts;
         i++, nodes_read++) {
      // Create a node and read its data from the sector buffer
      Node<T> node(meta.ndims, tr);
      char   *node_data_start = sector_buffer.data() + i * max_node_len;

      std::memcpy(&node.node_data[0], node_data_start, meta.ndims * sizeof(T));
      std::memcpy(&node.nnbrs, node_data_start + meta.ndims * sizeof(T),
                  sizeof(uint32_t));
      std::memcpy(&node.nhood[0],
                  node_data_start + meta.ndims * sizeof(T) + sizeof(uint32_t),
                  tr * sizeof(uint32_t));

      // Add node to node_vector
      node_vector.push_back(node);
    }
  }

  file.close();
}

template<typename T>
void print_Node(const std::vector<Node<T>> &node_vector, int i,
                std::string data_type) {
  Node<T> tmp = node_vector[i];
  std::cout << "Node " << i << ":\n";

  // std::cout<<"Vector: ";
  // for (auto j:tmp.node_data) std::cout<<j<<" ";
  std::cout << "Vector Size:" << tmp.node_data.size();
  std::cout << "\n";

  std::cout << "Number of Neighbor:" << tmp.nhood.size() << "\n";
  std::cout << "True Number of Neighbor:" << tmp.nnbrs << "\n";
  std::cout << "Neighbors List:";
  for (auto j : tmp.nhood)
    std::cout << j << " ";
  std::cout << "\n\n";
}

// Function to print metadata and the first node's information
template<typename T>
void print_metadata_and_node(const std::vector<Metadata> &metadata_vector,
                             int index, const std::vector<Node<T>> &node_vector,
                             std::string data_type) {
  // Print metadata
  const Metadata &meta = metadata_vector[index];
  std::cout << "Metadata: \n";
  std::cout << "Number of Meta Points (npts): " << meta.meta_npts << "\n";
  std::cout << "Number of Meta Dimensions (ndims): " << meta.meta_ndims << "\n";
  std::cout << "Number of Points (npts): " << meta.npts << "\n";
  std::cout << "Number of Dimensions (ndims): " << meta.ndims << "\n";
  std::cout << "Medoid: " << meta.medoid << "\n";
  std::cout << "Max Node Length: " << meta.max_node_len << "\n";
  std::cout << "Nodes per Sector: " << meta.nnodes_per_sector << "\n";
  std::cout << "Vamana Frozen Number: " << meta.vamana_frozen_num << "\n";
  std::cout << "Vamana Frozen Location: " << meta.vamana_frozen_loc << "\n";
  std::cout << "Append Reorder Data: " << meta.append_reorder_data << "\n";
  if (meta.append_reorder_data) {
    std::cout << "Number of Sectors: " << meta.n_sectors_add1 << "\n";
    std::cout << "Reorder File Dimensions: " << meta.ndims_reorder_file << "\n";
    std::cout << "Data Nodes per Sector: " << meta.n_data_nodes_per_sector
              << "\n";
  }
  std::cout << "Disk Index File Size: " << meta.disk_index_file_size << "\n\n";

  std::cout << typeid(T).name() << "\n";
  std::cout << "Number of Points (npts): " << meta.npts << "\n";
  std::cout << "Actual Number of Points: " << node_vector.size() << "\n\n";

  // print_Node(node_vector, 0, data_type);
  // print_Node(node_vector, 6, data_type);
  // print_Node(node_vector, 7, data_type);
  // print_Node(node_vector, node_vector.size() - 1, data_type);
  // for (uint64_t i=0;i<14;i++) {
  //     print_Node(node_vector,i,data_type);
  // }
}

template<typename T>
void change_r(std::vector<Metadata> &metadata_vector,
              std::vector<Node<T>> &node_vector, int r,
              std::string output_file_name, std::string data_type) {
  Metadata meta = metadata_vector[0];
  r = r - R;

  for (uint64_t i = 0; i < meta.npts; i++) {
    for (uint64_t j = 0; j < r; j++)
      node_vector[i].nhood.push_back(0);
  }
  meta.max_node_len += (r * sizeof(uint32_t));
  meta.nnodes_per_sector = SECTOR_SIZE / meta.max_node_len;

  std::cout << "max_node_len:" << meta.max_node_len << "\n";
  std::cout << "nnodes_per_sector:" << meta.nnodes_per_sector << "\n";

  std::ofstream file(output_file_name, std::ios::binary | std::ios::out);
  // 检查文件是否打开成功
  if (!file) {
    std::cerr << "Error opening file: " << output_file_name << std::endl;
    return;
  }

  std::vector<char> sector_buf(SECTOR_SIZE);
  std::memset(sector_buf.data(), 0, SECTOR_SIZE);
  file.write(sector_buf.data(), SECTOR_SIZE);

  // std::vector<char>node_buf(meta.max_node_len);
  // std::memset(node_buf.data(), 0, meta.max_node_len);

  uint64_t n_sectors = meta.npts / meta.nnodes_per_sector;
  if (meta.npts % meta.nnodes_per_sector)
    n_sectors++;
  meta.disk_index_file_size = (n_sectors + 1) * SECTOR_SIZE;

  uint64_t nodes_write = 0;
  for (uint64_t sector = 1; sector <= n_sectors && nodes_write < meta.npts;
       sector++) {
    if (sector % 100000 == 0) {
      std::cout << "Sector #" << sector << "written" << std::endl;
    }
    // Read data for the current sector
    uint64_t          sector_size = SECTOR_SIZE;
    std::vector<char> sector_buffer(sector_size);
    // Move the file pointer to the correct position for the current sector
    file.seekp(sector * SECTOR_SIZE, std::ios::beg);

    // 检查 seekp 是否成功
    if (file.fail()) {
      std::cerr << "Error seeking to position: " << sector * SECTOR_SIZE
                << std::endl;
      file.close();
      return;
    }

    std::memset(sector_buf.data(), 0, SECTOR_SIZE);

    // Process the sector data (reading nodes)
    for (uint64_t i = 0; i < meta.nnodes_per_sector && nodes_write < meta.npts;
         i++, nodes_write++) {
      Node<T> node = node_vector[nodes_write];
      char   *node_data_start = sector_buffer.data() + i * meta.max_node_len;

      std::memcpy(node_data_start, &node.node_data[0], meta.ndims * sizeof(T));
      std::memcpy(node_data_start + meta.ndims * sizeof(T), &node.nnbrs,
                  sizeof(uint32_t));
      std::memcpy(node_data_start + meta.ndims * sizeof(T) + sizeof(uint32_t),
                  &node.nhood[0], (R + r) * sizeof(uint32_t));
    }

    file.write(sector_buffer.data(), sector_size);
  }

  file.seekp(0, std::ios::beg);
  file.write((char *) &meta.meta_npts, sizeof(int));
  file.write((char *) &meta.meta_ndims, sizeof(int));
  file.write((char *) &meta.npts, sizeof(uint64_t));
  file.write((char *) &meta.ndims, sizeof(uint64_t));
  file.write((char *) &meta.medoid, sizeof(uint64_t));
  file.write((char *) &meta.max_node_len, sizeof(uint64_t));
  file.write((char *) &meta.nnodes_per_sector, sizeof(uint64_t));
  file.write((char *) &meta.vamana_frozen_num, sizeof(uint64_t));
  file.write((char *) &meta.vamana_frozen_loc, sizeof(uint64_t));
  file.write((char *) &meta.append_reorder_data, sizeof(uint64_t));
  if (meta.append_reorder_data) {
    file.write((char *) &meta.n_sectors_add1, sizeof(uint64_t));
    file.write((char *) &meta.ndims_reorder_file, sizeof(uint64_t));
    file.write((char *) &meta.n_data_nodes_per_sector, sizeof(uint64_t));
  }
  file.write((char *) &meta.disk_index_file_size, sizeof(uint64_t));

  file.close();
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "error argc\n";
    std::cout << "Please enter the input file path, output file path, R and "
                 "data type in order\n";
    return 0;
  }

  std::string intput_file_name =
      argv[1];  //"disk_index_sift_learn_R32_L50_A1.2_disk.index";
  std::cout << "intput_file_name:" << intput_file_name << "\n";
  std::string output_file_name = argv[2];
  std::cout << "output_file_name:" << output_file_name << "\n";
  int r = std::stoi(argv[3]);
  std::cout << "New R:" << r << "\n\n";
  std::string data_type = argv[4];

  if (r <= R) {
    std::cout << "error R\n";
    return 0;
  }

  std::vector<Metadata> metadata_vector;

  if (data_type == "float") {
    std::vector<Node<float>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "double") {
    std::vector<Node<double>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "long double") {
    std::vector<Node<long double>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "int8_t") {
    std::vector<Node<int8_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "uint8_t") {
    std::vector<Node<uint8_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "int16_t") {
    std::vector<Node<int16_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "uint16_t") {
    std::vector<Node<uint16_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "int32_t") {
    std::vector<Node<int32_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "uint32_t") {
    std::vector<Node<uint32_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "int64_t") {
    std::vector<Node<int64_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  } else if (data_type == "uint64_t") {
    std::vector<Node<uint64_t>> node_vector1, node_vector2;
    read_disk_index(intput_file_name, metadata_vector, node_vector1, R,
                    data_type);
    print_metadata_and_node(metadata_vector, 0, node_vector1, data_type);
    change_r(metadata_vector, node_vector1, r, output_file_name, data_type);
    read_disk_index(output_file_name, metadata_vector, node_vector2, r,
                    data_type);
    print_metadata_and_node(metadata_vector, 1, node_vector2, data_type);
  }
  return 0;
}
