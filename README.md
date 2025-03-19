# Greator

Greator is a system that supports large amounts of vector search and small batch update (deletion and insertion), with priority on update performance and search accuracy. The system is based on FreshDiskANN[1-3], and introduces a topology-aware local update strategy for graph-based ANN index for the first time, greatly improving update performance while ensuring high search accuracy. This code is forked from [code for DiskANN](https://github.com/microsoft/DiskANN)[4] algorithm.

## Build:

### Fetch Dependencies
Install the following packages through apt-get

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```
### Install Intel MKL

```bash
sudo apt install libmkl-full-dev
```
### Install Intel TBB
See [Installation from Sources](https://github.com/uxlfoundation/oneTBB) to learn how to install oneTBB.

### Build
```bash
mkdir build && cd build && cmake .. && make -j 
```

## Usage
1. To generate an SSD-based index, use the `tests/build_disk_index` program:
   ```bash
   ./tests/build_disk_index [data_type<float/int8/uint8>] [data_file.bin] [index_prefix_path] [R] [L] [B] [M] [T]
   ```

   The arguments are as follows:

   - **(i) data_type**: The datatype of the dataset, either byte indices (signed int8 or unsigned uint8) or float indices.
   - **(ii) data_file**: Input data in `.bin` format. The first 8 bytes represent the number of points and the dimension of the data. The rest contains the data points.
   - **(iii) index_prefix_path**: Prefix path for generated index files, e.g., `~/index_test_pq_pivots.bin`.
   - **(iv) R**: Degree of the graph index, typically between 60-150. Larger values improve search quality but increase index size and time.
   - **(v) L**: Size of the search list during index building, usually between 75-200. Larger values improve recall but take more time.
   - **(vi) B**: Memory footprint limit during search. Specifies RAM usage; excess data will be stored on disk.
   - **(vii) M**: Memory limit for index building. Lower values split the process into sub-graphs, which may slow down the build.
   - **(viii) T**: Number of threads for index building. More threads improve indexing speed, subject to available cores.

2. To update the SSD-based index, use `tests/overall_performance` program.
    ```
    ./tests/overall_performace [index_type<float/int8/uint8>] [data_file.bin] [data_file.bin] [L_mem] [R_mem] [alpha_mem] [L_disk] [R_disk] [alpha_disk] [num_start] [num_shards] [num_pq_chunks] [num_nodes_to_cache] [index_prefix_path] [update] [build] [data_file.bin] [query_file.bin] [truthset.bin] [K] [L] [beamwidth] [trace_file_prefix] [step] [use_topo].
    ```

    The important arguments are as follows:

    - **(i) update**: Whether to update (true or false).

    - **(ii) build**: Whether to build (true or false).

    - **(iii) trace_file_prefix**: The location of deletion and insertion points. The first 8 bytes represent the number of deleted or inserted points *NUM*, then *NUM* * 8 bytes represent the ID of deleted points, and finally *NUM* * 8 bytes represent the ID of inserted points

    - **(iv) step**: The number of rounds of updates.

    - **(v) use_topo**: Whether to use topology strategy (default is 2).

3. To search the SSD-based index, use the `tests/search_disk_index` program.
    ```
    ./tests/search_disk_index [index_type<float/int8/uint8>] [index_prefix_path] [num_nodes_to_cache] [num_threads] [beamwidth (use 0 to optimize internally)] [query_file.bin] [truthset.bin (use "null" for none)] [K] [result_output_prefix] [L1] [L2] etc.
    ```

    The arguments are as follows:

    - **(i) data_type**: Same as in (i) above in building index.

    - **(ii) index_prefix_path**: Same as in (iii) above in building index.

    - **(iii) num_nodes_to_cache**: The program stores the entire graph on disk. To improve search performance, a few nodes (closest to the starting point) can be cached in memory.

    - **(iv) num_threads**: Search using the specified number of threads in parallel, one thread per query. More threads may result in more IOs, so balance based on SSD bandwidth.

    - **(v) beamwidth**: Maximum number of IO requests each query will issue per search iteration. Larger beamwidth reduces IO round-trips but may result in higher SSD IO requests. Specifying 0 optimizes beamwidth based on the number of threads performing the search.

    - **(vi) query_file.bin**: Search on these queries, same format as the data file (ii) above. The query file must match the data type specified in (i).

    - **(vii) truthset.bin**: Must be in the following format, or specify "null": n (number of queries), followed by d (number of ground truth elements per query), followed by n*d entries of the closest IDs per query, followed by n*d entries of corresponding distances (float). Total file size is 8 + 4*n*d + 4*n*d. If not available, groundtruth can be calculated using the program `tests/utils/compute_groundtruth`. If only measuring latency, enter "null".

    - **(viii) K**: Measure recall@k, i.e., accuracy of retrieving top-k nearest neighbors.

    - **(ix) result_output_prefix**: Search results are stored in files with the specified prefix, in binary format.

    - **(x, xi, ...)**: Various search list sizes to perform the search. Larger sizes result in slower latencies but higher accuracies. Must be at least the recall@ value in (vi).

## References

[1] https://github.com/microsoft/DiskANN/tree/diskv2

[2] https://github.com/Yuming-Xu/DiskANN_Baseline

[3] Singh A, Subramanya S J, Krishnaswamy R, et al. Freshdiskann: A fast and accurate graph-based ann index for streaming similarity search[J]. arXiv preprint arXiv:2105.09613, 2021.

[4] Jayaram Subramanya S, Devvrit F, Simhadri H V, et al. Diskann: Fast accurate billion-point nearest neighbor search on a single node[J]. Advances in neural information processing Systems, 2019, 32.