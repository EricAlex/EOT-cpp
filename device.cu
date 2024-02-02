#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "device.h"

void sort_on_device(thrust::host_vector<int>& h_vec)
{
    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end());
    
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}

double dev_sum(std::vector<double>& V, double init){
    thrust::device_vector<double> d_vec(V);
    thrust::plus<int> binary_op;
    return thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
}