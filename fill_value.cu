#include <iostream>

__global__ void fill_value_kernel(int* res)
{
    *res = *res * 100;
}

extern "C" void fill_value(int* res)
{
    std::cout << "Entered function" << std::endl;
    int* res_;
    cudaMallocManaged(&res_, sizeof(int));
    cudaMemcpy(res_,res,sizeof(int),cudaMemcpyHostToDevice);
    fill_value_kernel<<<1, 1>>>(res_);
    cudaMemcpy(res,res_,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(res_);
    std::cout << "Done." << std::endl;
}
