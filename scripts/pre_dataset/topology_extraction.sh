g++ topology_extraction.cpp -o te

sector_len=4096
new_prefix=/data/linsy/Greator/scripts/indices/"$dataset"_R"$new_R"
./te /data/linsy/Cout/dynamic/disk_glove_R34/disk_init/_index_disk.index /data/linsy/Cout/dynamic/disk_glove_R34/disk_init/_index_disk.index_with_only_nbrs "$sector_len"
