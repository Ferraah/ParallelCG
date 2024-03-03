
__kernel void dot_product(global const double *a, global const double *b, local double *prods, global double *c, const uint size){


    // Global id 
    int gid = get_global_id(0);

    // Number of items per work-group
    int num_items = get_local_size(0);

    // Id of thread in work-group
    int lid = get_local_id(0);

    // Id of workgroup 
    int wg_num = get_group_id(0);

    int num_of_wg = get_num_groups(0);
    
    
    prods[lid] = 0.0;
    if(gid < size)
        prods[lid] = a[gid] * b[gid];
   
    for(int offset = 1; offset < num_items; offset*=2 ){

            int mask = 2*offset - 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if((lid & mask ) == 0 ){
                prods[lid] += prods[lid + offset];
            }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid==0)
        c[wg_num] = prods[0];
}
/**
* Compute the matrix-vector product (on the right). The matrix
* is linearized. This first kernel is rather simple: each row is mapped
* to a specific thread
* 
*/
__kernel void A_times_x_kernel(
    global const double* dA,
    global const double* dB,
    global double* res,
    const uint num_rows,
    const uint num_cols
)
{
    uint tid = get_global_id(0);
    uint stride = get_local_size(0)*get_num_groups(0);


    // The idea here is that each thread of each block is assigned to a specific row of
    // the matrix.
    for (unsigned int row = tid; row < num_rows; row += stride)
    {
        double accumulator = 0.0;
        // And here there is no use of shared memory.
        for (unsigned int col = 0; col < num_cols; ++col)
        {
            accumulator += dA[row * num_cols + col] * dB[col];
        }
        // Finally, store the result in the corresponsing element
        // of the vector.
        res[row] = accumulator;
    }
}


__kernel void vec_sum(
    const double alpha,
    global const double *dA,
    const double beta,
    global double *dB,
    const uint size
){
    uint gid = get_global_id(0);

    if(gid < size){
        dB[gid] = alpha*dA[gid] + beta*dB[gid];
    } 

}
