#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>
#include <string>
#include <set>
#include <map>
#include <random>
#include <unistd.h>

#define sint32 sizeof(uint32_t)
#define sfloat sizeof(float)
#define sint64 sizeof(uint64_t)
#define suint sizeof(uint8_t)
#define sdata sfloat
#define R 34
using data_t = uint32_t;
using offset_t = uint64_t;
using _u64 = uint64_t;

void topogy_extraction(const std::string &indir, const std::string &outdir, offset_t SECTOR_LEN)
{
    std::ifstream infile(indir, std::ios::binary);
    if (!infile.is_open())
    {
        std::cerr << "无法打开文件: " << indir << std::endl;
        throw std::ios_base::failure("File not found");
    }

    // 获取文件大小
    infile.seekg(0, std::ios::end);
    std::streampos file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::cout << "file_size: " << file_size << std::endl;

    std::ofstream outfile(outdir, std::ios::binary);
    if (!outfile.is_open())
    {
        std::cerr << "无法打开文件: " << outdir << std::endl;
        throw std::ios_base::failure("File not found");
    }

    // 分配缓冲区
    // 读
    std::vector<char> read_buf(SECTOR_LEN);
    infile.read(read_buf.data(), SECTOR_LEN);
    // 写
    std::vector<char> write_buf(SECTOR_LEN);

    // 获取npts和dim
    _u64 npts = 0;
    _u64 dim = 0;
    memcpy(&npts, read_buf.data() + sint64, sint64);
    memcpy(&dim, read_buf.data() + sint64 * 2, sint64);
    std::cout << "npts: " << npts << " dim: " << dim << std::endl;

    // 写入npts和R
    _u64 range = (_u64)R;

    memcpy(write_buf.data(), &npts, sint64);
    memcpy(write_buf.data() + sint64, &range, sint64);
    outfile.seekp(0, std::ios::beg);
    outfile.write(write_buf.data(), SECTOR_LEN);
    // read_sector
    _u64 size_per_node_read = (_u64)(1 + R) * sint32 + dim * sdata;
    _u64 nnodes_per_sector_read = SECTOR_LEN / size_per_node_read;
    _u64 page_num_read = (_u64)(npts % nnodes_per_sector_read) ? (1 + npts / nnodes_per_sector_read) : (npts / nnodes_per_sector_read);
    std::cout << "Read_Secotr's size_per_node: " << size_per_node_read << " nnodes_per_sector: " << nnodes_per_sector_read << " page_num: " << 1 + page_num_read << "\n";

    // write_sector
    _u64 size_per_node_write = (_u64)(1 + R) * sint32;
    _u64 nnodes_per_sector_write = SECTOR_LEN / size_per_node_write;
    _u64 page_num_write = (_u64)(npts % nnodes_per_sector_write) ? (1 + npts / nnodes_per_sector_write) : (npts / nnodes_per_sector_write);
    std::cout << "Write_Sector's size_per_node: " << size_per_node_write << " nnodes_per_sector: " << nnodes_per_sector_write << " page_num: " << 1 + page_num_write << "\n";
    data_t cur_node_id = 0;
    data_t cur_sector_id = 1;

    for (size_t i = 0; i < page_num_read; i++)
    {
        infile.clear();
        offset_t offset_read = (offset_t)SECTOR_LEN * (i + 1);
        infile.seekg(offset_read, std::ios::beg);
        infile.read(read_buf.data(), SECTOR_LEN);

        if (!infile)
        {
            std::cout << "Page_Num_Read: " << i << " Wrong" << std::endl;
            return;
        }

        for (size_t j = 0; j < nnodes_per_sector_read; j++)
        {
            size_t id_read = j + i * nnodes_per_sector_read;
            if (id_read >= npts)
                break;

            offset_t offset_node = (offset_t)(j * size_per_node_read + dim * sdata);

            // 解析 nnbrs
            data_t nnbrs_read = 0;
            memcpy(&nnbrs_read, read_buf.data() + offset_node, sint32);
            offset_node += sint32;

            // 解析 nbrs
            std::vector<data_t> nbrs_read(R);
            memcpy(nbrs_read.data(), read_buf.data() + offset_node, nnbrs_read * sint32);
            for (int nn = nnbrs_read; nn < R; nn++)
            {
                nbrs_read[nn] = 0;
            }
            // 加入 Write_buf
            if (cur_node_id == nnodes_per_sector_write) // write_buf 满
            {

                offset_t offset_in_buf = (offset_t)cur_sector_id * SECTOR_LEN;
                // std::cout << "write #" << cur_sector_id << " & " << offset_in_buf << std::endl;
                outfile.seekp(offset_in_buf, std::ios::beg);
                outfile.write(write_buf.data(), SECTOR_LEN);
                cur_sector_id++;
                cur_node_id = 0;
            }
            offset_t offset_in_sector = cur_node_id * size_per_node_write;
            memcpy(write_buf.data() + offset_in_sector, &nnbrs_read, sint32);
            offset_in_sector += sint32;
            memcpy(write_buf.data() + offset_in_sector, nbrs_read.data(), R * sint32);
            cur_node_id++;
        }
    }
    if (cur_node_id != 0)
    {
        offset_t offset_in_buf = (offset_t)cur_sector_id * SECTOR_LEN;
        outfile.seekp(offset_in_buf, std::ios::beg);
        outfile.write(write_buf.data(), SECTOR_LEN);
    }

    infile.close();
    outfile.close();
    std::cout << outdir << " 's page_num: " << cur_sector_id << " , node_num: " << (cur_sector_id - 1) * nnodes_per_sector_write + cur_node_id << "\n";
}

int main(int agrs, char *agrvs[])
{
    // 获取当前时间点（起始时间）
    auto start = std::chrono::high_resolution_clock::now();

    std::string indir = agrvs[1];
    std::string outdir = agrvs[2];
    int SECTOR_LEN = std::atoi(agrvs[3]);
    offset_t sector_len = (offset_t)SECTOR_LEN;
    topogy_extraction(indir, outdir, sector_len);

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算两个时间点之间的差值（持续时间）
    std::chrono::duration<double> duration = end - start;

    // 输出持续时间，单位是秒
    std::cout << "Topological extraction elapsed time: " << duration.count() << " seconds\n";
    return 0;
}