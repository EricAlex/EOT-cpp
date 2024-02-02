#pragma once

#include <thrust/host_vector.h>
#include <vector>

// function prototype
void sort_on_device(thrust::host_vector<int>& V);
double dev_sum(std::vector<double>& V, double init);