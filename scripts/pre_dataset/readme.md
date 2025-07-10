### Streaming Update Trace Generator (make_trace.cpp)

This script is designed to generate trace files for **streaming update scenarios** in dynamic vector indexing experiments.

To ensure **standardization** and **reproducibility** of results, we adopt the following experimental setup:

- **Only the first 95% of the vectors** (i.e., IDs from `0` to `0.95×N − 1`) are used to construct the initial index.
- In each update round:
  - The earliest **0.1%** of vectors in the current index are **deleted**.
  - A new **0.1%** of vectors are **inserted**.
- This simulates a typical **sliding-window update pattern** and supports 50 rounds of streaming updates.

#### Example

Suppose the total number of vectors is **1,000,000**:

- The initial index is built from vectors with IDs ranging from `0` to `949,999`.
- In the **first update**:
  - The system **deletes** vectors with IDs from `0` to `999`.
  - Then **inserts** vectors with IDs from `950,000` to `950,999`.

### Base_95 Generator (extract_base.cpp)

To accurately reproduce the experimental results, no randomization is introduced during the preprocessing stage. Therefore, the extract_base.cpp file simply extracts the first 95% of vectors from the dataset (assuming 50 update rounds with 0.1% updates per round).

### Ground Truth File Generator for Each Update Round (compute_knn.cpp)

To accurately calculate the recall after each update round, the ground truth files for each round need to be generated in advance.

### Tags File Generator (make_tags.cpp)

This script is designed to generate tags files for **streaming update scenarios** in dynamic vector indexing experiments.

### Process Index_R32 as Index_R34 (process_index_for_diffR.cpp)

This script is designed to process the index_R32 file to generate index_R34 file.

### Extract Topology File from Index_R34 (topology_extraction.cpp)

This script is designed to extract the topology file from the index_R34 file.