
clear
project_dir=/data/linsy/Greator
id_map=2
delete_dir="$project_dir/scripts/indices/sift_R34"
batchsize=0.001
find "$delete_dir" -mindepth 1 ! -path "$delete_dir/disk_init*" -exec rm -rf {} +

cp "$delete_dir/disk_init"/* "$delete_dir"/

rm -r "$delete_dir"/_index_temp
mkdir "$delete_dir"/_index_temp


cd /data/linsy/Greator/build && make -j 
cd /data/linsy/Greator/run

name=sift
mydir="/data/linsy/Greator/scripts"
index_type="float"
base_data_file="$mydir"/dataset/"$name"/"$name"_base_95.fbin
L_mem=75
R_mem=34
alpha_mem=1.2
L_disk=75
R_disk=34
alpha_disk=1.2
num_start=0
num_shards=100
num_pq_chunks=100
num_nodes_to_cache=10000
save_graph_file="$delete_dir"/_index
update=true
build=false
full_data_bin="$mydir"/dataset/"$name"/"$name"_base.fbin
query_bin="$mydir"/dataset/"$name"/"$name"_query.fbin
truthset="$mydir"/dataset/"$name"/gt/"$name"_gt_K10_
recall_k=10
search_L1=120
beamwidth=2
trace_file_prefix="$mydir"/trace/"$name"_trace_"$batchsize"/_trace
step=5
C=160


"$project_dir"/build/tests/overall_performance "$index_type" "$base_data_file" "$L_mem" "$R_mem" "$alpha_mem" "$L_disk" "$R_disk" "$alpha_disk" "$num_start" "$num_shards" "$num_pq_chunks" "$num_nodes_to_cache" "$save_graph_file" "$update" "$build" "$full_data_bin" "$query_bin" "$truthset" "$recall_k" "$search_L1" "$beamwidth" "$trace_file_prefix" "$step" "$id_map" 

