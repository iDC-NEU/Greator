### Step 1: Run `pre_dataset.sh`

The example in this paper uses the SIFT1M dataset. Please download and prepare the files `sift_base.fbin` and `sift_query.fbin` before running the scripts.

This shell script consists of multiple smaller scripts, each serving a specific purpose. For detailed information about each script, please refer to `./scripts/pre_dataset/readme.md`.  

In this step, we generate the update trace for each iteration, the base 95% vectors file, and the ground truth files for each round.  
- The directory `./scripts/dataset/sift/gt` contains `num_iterations` ground truth files.  
- The directory `./scripts/dataset/sift` contains `sift_base_95.fbin`, `sift_base.fbin`, and `sift_query.fbin`.  
- The directory `./scripts/trace/sift_trace_0.001` contains `num_iterations` trace files.

---


### Step 2: Run `build_indices.sh`

In this step, we build the `index_R32` and `index_R34` (i.e., 33/34) index files.  
- The directory `./scripts/indices/sift_R32/disk_init` contains 6 files.  
- The directory `./scripts/indices/sift_R34/disk_init` contains 7 files (including the topology file):`_index_disk.index`, `_index_disk.index.tags`, `_index_disk.index_with_only_nbrs`, `_index_pq_compressed.bin`,  `_index_pq_pivots.bin`,  `_index_sample_data.bin`, `_index_sample_ids.bin`.

---

### Step 3: Run `overall_performance.sh`

In this step, we evaluate the overall performance of the `index_R34` (i.e., 33/34) index. For each update round, we calculate the recall and update time.  
We focus on the output values of **Recall** and the update time indicated by messages like:  
`Merge_kernel use 6.09201 s.`

It is worth noting that Greator’s update algorithms support asynchronous updates. However, to more accurately evaluate the update algorithm’s performance and the recall after updates, we adopt a synchronous update approach. Using asynchronous updates makes it difficult to precisely assess the update throughput and the quality of the updated index.

