project_dir=/data/linsy/Greator
# Part1: build_disk_index_R32 for FreshDiskANN

dataset=sift
cindir="$project_dir"/scripts/dataset/"$dataset"/"$dataset"_base_95.fbin
coutdir="$project_dir"/scripts/indices/"$dataset"_R32/disk_init
mkdir $coutdir
cd "$project_dir"/build && make -j
cd "$project_dir"/run

"$project_dir"/build/tests/build_disk_index float   "$cindir"  "$coutdir"/_index 32  128  98  98  64 l2 0 

# part2: make_tags for disk_index_R32 and disk_index_R34

cd "$project_dir"/scripts/pre_dataset
bash make_tags.sh

# Part3: process disk_index_R32 as disk_index_R34 for Greator (ie, 33/34)

cd "$project_dir"/scripts/pre_dataset
bash process_index_for_diffR.sh


