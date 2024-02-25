
__kernel void dot_product(global const double *a, global const double *b, local double *prods, global double *c){

    // Global id 
    int gid = get_global_id(0);

    // Number of items per work-group
    int num_items = get_local_size(0);

    // Id of thread in work-group
    int lid = get_local_id(0);

    // Id of workgroup 
    int wg_num = get_group_id(0);
    prods[lid] = a[gid] * b[gid];

    for(int offset = 1; offset <num_items; offset*=2 ){
        int mask = 2*offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if((lid & mask ) == 0){
            prods[lid] += prods[lid + offset];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid==0)
        c[wg_num] = prods[0];

}