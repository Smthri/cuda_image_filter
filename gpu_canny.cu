#include <iostream>
#include <opencv2/core.hpp>
#include <cuda.h>
#include <cstdio>
#include <cmath>

namespace cuda
{
    __device__ int im2col(const float* src, const int src_w, const int src_h, const int k, const int y, const int x, float* dst) {
        if (y + k > src_h || x + k > src_w || y < 0 || x < 0) {
            return 1;
        }

        const float* src_ = src + y * src_w + x;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                *dst++ = src_[j];
            }
            src_ += src_w;
        }

        return 0;
    }

    __device__ float dot_product(const float* src, const float* kernel, const int N) {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += src[i] * kernel[i];
        }
        return sum;
    }

    __global__ void gradient_gpu_kernel(
            const float* src_,
            const int src_h,
            const int src_w,
            const int dst_h,
            const int dst_w,
            const float sigma,
            const int k,
            float* dst_,
            float* kernelx,
            float* kernely) {
        float* src_vec = (float *) malloc(k * k * sizeof(float));
        const int N = k * k;

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int im2col_res = im2col(src_, src_w, src_h, k, i, j, src_vec);
        float sumx = dot_product(src_vec, kernelx, N);
        float sumy = dot_product(src_vec, kernely, N);
        dst_[i * dst_w + j] = sqrtf(sumx * sumx + sumy * sumy);
        free(src_vec);
    }
}

void parseCudaResult(std::string label, cudaError_t res) {
    if (res) {
        std::cout << label << ": " << cudaGetErrorString(res) << std::endl;
    }
}

void allocKernel(const int k, const float sigma, float** kernelx, float** kernely) {
    cudaError_t res;
    res = cudaMalloc(kernelx, k * k * sizeof(float));
    parseCudaResult("kernelx alloc", res);
    res = cudaMalloc(kernely, k * k * sizeof(float));
    parseCudaResult("kernely alloc", res);

    float* kernelx_ = (float*) malloc(k * k * sizeof(float));
    float* kernely_ = (float*) malloc(k * k * sizeof(float));
    float* kx_ = kernelx_;
    float* ky_ = kernely_;
    float norm = -1.0 / (2.0 * M_PI * sigma * sigma * sigma * sigma);
    float expnorm = 2.0 * sigma * sigma;
    for (int i = 0; i < k; ++i) {
        float y = k / 2 - i;
        for (int j = 0; j < k; ++j) {
            float x = k / 2 - j;
            float scary = norm * std::exp(-(x * x + y * y) / expnorm);
            *kx_++ = x * scary;
            *ky_++ = y * scary;
        }
    }

    res = cudaMemcpy(*kernelx, kernelx_, k * k * sizeof(float), cudaMemcpyHostToDevice);
    parseCudaResult("kernelx memcpy", res);
    res = cudaMemcpy(*kernely, kernely_, k * k * sizeof(float), cudaMemcpyHostToDevice);
    parseCudaResult("kernely memcpy", res);
    free(kernelx_);
    free(kernely_);
}

void freeKernel(float* kernelx, float* kernely) {
    cudaError_t res;
    res = cudaFree(kernelx);
    parseCudaResult("kernelx free", res);
    res = cudaFree(kernely);
    parseCudaResult("kernely free", res);
}

extern "C" int canny_gpu(cv::Mat& src, const float sigma, cv::Mat& dst) {
    cv::Mat src_ = src;
    if (!src.isContinuous() || (src.type() & CV_MAT_DEPTH_MASK) != CV_32F) {
        src.convertTo(src_, CV_32F);
    }
    cv::Mat _src_;
    const int k = (int) std::ceil(sigma) * 6 + 1;
    cv::copyMakeBorder(src_, _src_, k / 2, k / 2, k / 2, k / 2, CV_HAL_BORDER_REFLECT);

    const int src_h = _src_.size[0];
    const int src_w = _src_.size[1];
    if (k >= src_h + 1 || k >= src_w + 1) {
        return 1;
    }
    const int dst_h = src_h - k + 1;
    const int dst_w = src_w - k + 1;
    dst = cv::Mat(dst_h, dst_w, CV_32F);

    float* cuda_src;
    float* cuda_dst;
    float* kernelx;
    float* kernely;
    allocKernel(k, sigma, &kernelx, &kernely);
    cudaError_t res;

    res = cudaMalloc(&cuda_src, src_h * src_w * sizeof(float));
    parseCudaResult("malloc src", res);
    res = cudaMalloc(&cuda_dst, dst_h * dst_w * sizeof(float));
    parseCudaResult("malloc dst", res);
    res = cudaMemcpy(cuda_src, _src_.ptr(), src_h * src_w * sizeof(float), cudaMemcpyHostToDevice);
    parseCudaResult("memcpy src", res);

    const int thread_per_dim = 20;
    const dim3 grid_size(dst_w / thread_per_dim, dst_h / thread_per_dim);
    const dim3 block_size(thread_per_dim, thread_per_dim);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    cuda::gradient_gpu_kernel<<<grid_size, block_size>>>(
            cuda_src,
                    src_h, src_w,
                    dst_h, dst_w,
                    sigma, k,
                    cuda_dst,
                    kernelx,
                    kernely
    );
    res = cudaDeviceSynchronize();
    parseCudaResult("synchronize", res);

    freeKernel(kernelx, kernely);

    res = cudaMemcpy(dst.ptr(), cuda_dst, dst_h * dst_w * sizeof(float), cudaMemcpyDeviceToHost);
    parseCudaResult("memcpy dst", res);
    res = cudaFree(cuda_src);
    parseCudaResult("free src", res);
    res = cudaFree(cuda_dst);
    parseCudaResult("free dst", res);

    return (int) res;
}
