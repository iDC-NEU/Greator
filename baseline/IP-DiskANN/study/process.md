# 加油吧！！！
# Overall_performance 运行过程
## 1. 设置参数
## 2. Build
> build(learn.bin,L_mem, R_mem, alpha_mem, L_disk, R_disk,alpha_disk, num_start, num_shards, num_pq_chunks,nodes_to_cache, save_path)
### 2.1 读取data_learn.bin -> data_load
### 2.2 创建tags(num=num_start,value=0)，并保存到save_path+"_disk.index.tags"
## 3. update
> update(full_data_path, L_mem, R_mem, alpha_mem, L_disk,R_disk, alpha_disk, step, num_start,num_pq_chunks, nodes_to_cache, save_path,query_file, truthset, recall_at, Lsearch,beam_width, trace_file_prefix, &dist_cmp)
### 3.1 定义MergeInsert类型变量sync_index，并读取disk/_index所有索引文件
> (paras, dim, save_path + "_mem",save_path, save_path + "_merge",dist_cmp, metric, false, save_path)
### 3.2 读取query文件  #pts = 10000, #dims = 128, aligned_dim = 128
### 3.3 获取当前的truthfile_name（不计算recall就不用读）
### 3.4 搜索设置每个查询点的邻居——sync_search_kernel()，并保存到result_overall_spacev_diskann{cur_time}.bin中
### 3.5 批次处理迭代——for batch
### 3.5.1 通过_trace_i文件获得update_size、insert_ids、delete_ids，然后通过learn.bin获得ids对应的向量
### 3.5.2 异步执行 insertion_kernel() 和 deletion_kernel()
#### 3.5.2.1 insertion_kernel()
> 1.遍历insert_vec里的每个元素，执行insert操作。
>>  * 等到写锁消失，即可以insert(写)
>>  * 上写锁
>>  * 寻找可以用的mem_index（0/1）
>>  * 执行insert_point 操作：
>>>     A. 对于重复插入的点，先删除（delete_set.insert+location_to_tag.erase+tag_to_location.erase）
>>>>    B. 获取插入的locations，更新tag_to_location和tag_to_location_size
>>>>    C. 获得插入点的邻居节点，剪枝处理
>>>>    D. 将剪枝后的邻居加到graph[location]中，并且添加反向边。
#### 3.5.2.2 delete_kernel()
> 1.遍历delete_vec里的每个元素，执行delete操作。
>>  * 上删锁
>>  * 寻找可以用的mem_index（0/1）
>>  * 执行lazy_delete 操作：添加到_deletion_set_1/0中，和mem_index_0/1的_delete_set




save_del_set():将merge_insert的_deletion_set_1/0 保存到_deleted_tags_vector,保存前先清空。
switch_index()中的save()是将mem_index的_delete_set 保存到内存文件，但不清空delete_set.
故deleted_tags存的是内存中

mem_data存的实际上就是插入的点。


process_deletes() 问题：
1. #586 对于检查medoid，仅检查了第0个。且对于含删除标记的medoid的处理欠妥，代码的做法是从medoid[0]的邻居中选择第0个作为新的medoid。这里直接选择邻居是没问题的，因为已经去除了所有点的含删除标记的邻居。



