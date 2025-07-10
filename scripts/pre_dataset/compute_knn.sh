g++ compute_knn.cpp -fopenmp -o ck

dataset=sift
basefile=/data/linsy/Greator/scripts/dataset/"$dataset"/"$dataset"_base.fbin
queryfile=/data/linsy/Greator/scripts/dataset/"$dataset"/"$dataset"_query.fbin
gt_prefix=/data/linsy/Greator/scripts/dataset/"$dataset"/gt
mkdir $gt_prefix
# 从0到50的数字进行循环
for i in {0..50}
do
    echo "当前轮次是 $i"

    #0~949999
    startid=$((0 + i * 1000))
    endid=$((949999 + i * 1000))
    gtfile=$gt_prefix/"$dataset"_gt_K10_"$i".fbin
    ./ck "$basefile" "$queryfile" "$gtfile" "$startid" "$endid"
done

