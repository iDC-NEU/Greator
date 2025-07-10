g++ make_trace.cpp -o mtrace

dataset=sift
npts=950000
delta=1000 #num_points_per_update
filename=/data/linsy/Greator/scripts/trace/"$dataset"_trace_0.001
update_iteration=5
mkdir $filename

./mtrace $filename/_trace $npts $delta $update_iteration


