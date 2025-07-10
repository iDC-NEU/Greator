g++ process_index_for_diffR.cpp -o pifdr


dataset=sift
old_R=32
new_R=34
old_prefix=/data/linsy/Greator/scripts/indices/"$dataset"_R"$old_R"
new_prefix=/data/linsy/Greator/scripts/indices/"$dataset"_R"$new_R"

mkdir $new_prefix 
mkdir $new_prefix/disk_init/

cp -r "$old_prefix"/disk_init/ "$new_prefix"/
rm "$new_prefix"/disk_init/_index_disk.index


./pifdr "$old_prefix"/disk_init/_index_disk.index "$new_prefix"/disk_init/_index_disk.index "$new_R" float


# extract topology file

g++ topology_extraction.cpp -o te
sector_len=4096

./te "$new_prefix"/disk_init/_index_disk.index "$new_prefix"/disk_init/_index_disk.index.index_with_only_nbrs "$sector_len"
