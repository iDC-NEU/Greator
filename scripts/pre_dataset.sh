mkdir ./dataset
mkdir ./indices
mkdir ./trace
cd ./pre_dataset

bash make_trace.sh

bash extract_base.sh

bash compute_knn.sh