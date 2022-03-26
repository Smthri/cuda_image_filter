#include <iostream>
#include <opencv2/core.hpp>
#include <cuda.h>
#include <cstdio>
#include <cmath>

#define CHECK_BOUNDS(y, x, h, w) \
    ((x) < 0 || (y) < 0 || (x) >= (w) || (y) >= (h))

namespace cuda {
    __device__ int im2col(
            const float* src,
            const int src_w,
            const int src_h,
            const int k,
            const int y,
            const int x,
            float* dst
    ) {
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
            const float* kernelx,
            const float* kernely,
            unsigned char* directions) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
            printf("grad thread exit\n");
            return;
        }

        const int N = k * k;
        float sumx = 0;
        float sumy = 0;
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                sumx += src_[(i + ky) * src_w + j + kx] * kernelx[ky * k + kx];
                sumy += src_[(i + ky) * src_w + j + kx] * kernely[ky * k + kx];
            }
        }
        dst_[i * dst_w + j] = sqrtf(sumx * sumx + sumy * sumy);
        float angle = atan2f(sumy, sumx);
        if (sumx != 0 || sumy != 0) {
            double dists[9] = {
                    fabs(angle - M_PI),
                    fabs(angle - 3 * M_PI_4),
                    fabs(angle - M_PI_2),
                    fabs(angle - M_PI_4),
                    fabs(angle),
                    fabs(angle + M_PI),
                    fabs(angle + 3 * M_PI_4),
                    fabs(angle + M_PI_2),
                    fabs(angle + M_PI_4)
            };
            int min_index = 0;
            double min = FLT_MAX;
            for (int el = 0; el < 8; ++el) {
                if (dists[el] < min) {
                    min = dists[el];
                    min_index = el;
                }
            }
            switch (min_index) {
                case 1:
                case 8:
                    directions[i * dst_w + j] = 255;
                    break;
                case 2:
                case 7:
                    directions[i * dst_w + j] = 128;
                    break;
                case 3:
                case 6:
                    directions[i * dst_w + j] = 192;
                    break;
                default:
                    directions[i * dst_w + j] = 64;
            }
        }
    }

    __global__ void zero_pad(
            const float* src,
            const int src_h,
            const int src_w,
            float* dst,
            const int dst_h,
            const int dst_w
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int xoffset = (dst_h - src_h) / 2;
        int yoffset = (dst_w - src_w) / 2;
        if (CHECK_BOUNDS(i, j, src_h, src_w) || CHECK_BOUNDS(i + yoffset, j + xoffset, dst_h, dst_w)) {
            printf("padding thread exit\n");
            return;
        }

        dst[(i + yoffset) * dst_w + j + xoffset] = src[i * src_w + j];
    }

    __global__ void nonmax(
            const unsigned char* directions,
            const float* grad,
            const int grad_h,
            const int grad_w,
            const int dst_h,
            const int dst_w,
            float* dst
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
            printf("thread exit\n");
            return;
        }

        float m = 0;
        switch (directions[i * dst_w + j]) {
            case 128:
                m = fmax(grad[i * grad_w + j + 1], grad[(i + 2) * grad_w + j + 1]);
                break;
            case 64:
                m = fmax(grad[(i + 1) * grad_w + j], grad[(i + 1) * grad_w + j + 2]);
                break;
            case 255:
                m = fmax(grad[i * grad_w + j + 2], grad[(i + 2) * grad_w + j]);
                break;
            case 192:
                m = fmax(grad[(i + 2) * grad_w + j + 2], grad[i * grad_w + j]);
                break;
        }

        if (grad[(i + 1) * grad_w + j + 1] > m) {
            dst[i * dst_w + j] = grad[(i + 1) * grad_w + j + 1];
        } else {
            dst[i * dst_w + j] = 0;
        }
    }

    __global__ void hysteresis(
            float* dst,
            const int dst_h,
            const int dst_w,
            const float low_thr,
            const float high_thr
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (CHECK_BOUNDS(i, j, dst_h, dst_w)) {
            printf("hyst bounds check\n");
        }

        int dst_index = i * dst_w + j;
        int neighbor_idxs[8] = {
                (i - 1) * dst_w + j - 1,
                (i - 1) * dst_w + j,
                (i - 1) * dst_w + j + 1,
                i * dst_w + j - 1,
                i * dst_w + j + 1,
                (i + 1) * dst_w + j - 1,
                (i + 1) * dst_w + j,
                (i + 1) * dst_w + j + 1
        };

        if (dst[dst_index] < low_thr) {
            dst[dst_index] = 0;
        } else if (dst[dst_index] < high_thr) {
            dst[dst_index] = 128;
        } else {
            dst[dst_index] = 255;
        }
        __syncthreads();

        __shared__ int changed;
        do {
            changed = 0;
            __syncthreads();

            if (dst[dst_index] == 128) {
                if (i > 0) {
                    if (j > 0) {
                        if (dst[neighbor_idxs[0]] == 255) {
                            dst[dst_index] = 255;
                        }
                    }
                    if (dst[neighbor_idxs[1]] == 255) {
                        dst[dst_index] = 255;
                    }
                    if (j < dst_w - 1) {
                        if (dst[neighbor_idxs[2]] == 255) {
                            dst[dst_index] = 255;
                        }
                    }
                }

                if (j > 0) {
                    if (dst[neighbor_idxs[3]] == 255) {
                        dst[dst_index] = 255;
                    }
                }
                if (j < dst_w - 1) {
                    if (dst[neighbor_idxs[4]] == 255) {
                        dst[dst_index] = 255;
                    }
                }

                if (i < dst_h - 1) {
                    if (j > 0) {
                        if (dst[neighbor_idxs[5]] == 255) {
                            dst[dst_index] = 255;
                        }
                    }
                    if (dst[neighbor_idxs[6]] == 255) {
                        dst[dst_index] = 255;
                    }
                    if (j < dst_w - 1) {
                        if (dst[neighbor_idxs[7]] == 255) {
                            dst[dst_index] = 255;
                        }
                    }
                }

                if (dst[dst_index] == 255) {
                    changed = 1;
                }
            }
            __syncthreads();
        } while (changed);

        if (dst[dst_index] == 128) {
            dst[dst_index] = 0;
        }
    }
}

void parseCudaResult(std::string label, cudaError_t res) {
    if (res) {
        std::cout << label << ": " << cudaGetErrorString(res) << std::endl;
    }
}

void allocKernel(const int k, const float sigma, float** kernelx, float** kernely) {
    parseCudaResult("kernelx alloc", cudaMalloc(kernelx, k * k * sizeof(float)));
    parseCudaResult("kernely alloc", cudaMalloc(kernely, k * k * sizeof(float)));

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

    parseCudaResult("kernelx memcpy", cudaMemcpy(*kernelx, kernelx_, k * k * sizeof(float), cudaMemcpyHostToDevice));
    parseCudaResult("kernely memcpy", cudaMemcpy(*kernely, kernely_, k * k * sizeof(float), cudaMemcpyHostToDevice));
    free(kernelx_);
    free(kernely_);
}

void freeKernel(float* kernelx, float* kernely) {
    parseCudaResult("kernelx free", cudaFree(kernelx));
    parseCudaResult("kernely free", cudaFree(kernely));
}

extern "C" int canny_gpu(cv::Mat& src, const float sigma, const float low_thr, const float high_thr, cv::Mat& dst) {
    cudaEvent_t all_start, exec_start, all_stop, exec_stop;
    float all_ms, exec_ms;
    cudaEventCreate(&all_start);
    cudaEventCreate(&exec_start);
    cudaEventCreate(&all_stop);
    cudaEventCreate(&exec_stop);
    cudaEventRecord(all_start);

    parseCudaResult("select device", cudaSetDevice(0));

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
    const int cuda_dst_h = dst_h + dst_h % BLOCK_SIZE;
    const int cuda_dst_w = dst_w + dst_w % BLOCK_SIZE;
    dst = cv::Mat(cuda_dst_h, cuda_dst_w, CV_32FC1);

    float* cuda_src;
    float* cuda_dst;
    unsigned char* cuda_directions;
    float* cuda_grads;
    float* kernelx;
    float* kernely;

    allocKernel(k, sigma, &kernelx, &kernely);

    parseCudaResult("malloc src", cudaMalloc(&cuda_src, src_h * src_w * sizeof(float)));
    parseCudaResult("malloc dst", cudaMalloc(&cuda_dst, cuda_dst_h * cuda_dst_w * sizeof(float)));
    parseCudaResult("malloc grads", cudaMalloc(&cuda_grads, (cuda_dst_h + 2) * (cuda_dst_w + 2) * sizeof(float)));
    parseCudaResult("malloc directions", cudaMalloc(&cuda_directions, cuda_dst_h * cuda_dst_w * sizeof(unsigned char)));
    parseCudaResult("memcpy src",
                    cudaMemcpy(cuda_src, _src_.ptr(), src_h * src_w * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid_size(cuda_dst_w / BLOCK_SIZE, cuda_dst_h / BLOCK_SIZE);
    const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * 1024 * 1024);
    cudaEventRecord(exec_start);
    cuda::gradient_gpu_kernel<<<grid_size, block_size>>>(
            cuda_src,
                    src_h, src_w,
                    cuda_dst_h, cuda_dst_w,
                    sigma, k,
                    cuda_dst,
                    kernelx,
                    kernely,
                    cuda_directions
    );
    cuda::zero_pad<<<grid_size, block_size>>>(
            cuda_dst,
                    cuda_dst_h,
                    cuda_dst_w,
                    cuda_grads,
                    cuda_dst_h + 2,
                    cuda_dst_w + 2
    );
    cuda::nonmax<<<grid_size, block_size>>>(
            cuda_directions,
                    cuda_grads,
                    cuda_dst_h + 2,
                    cuda_dst_w + 2,
                    cuda_dst_h,
                    cuda_dst_w,
                    cuda_dst
    );
    cuda::hysteresis<<<grid_size, block_size>>>(
            cuda_dst,
                    cuda_dst_h,
                    cuda_dst_w,
                    low_thr,
                    high_thr
    );
    parseCudaResult("record stop", cudaEventRecord(exec_stop));

    freeKernel(kernelx, kernely);

    parseCudaResult("memcpy dst",
                    cudaMemcpy(dst.ptr(), cuda_dst, cuda_dst_h * cuda_dst_w * sizeof(float), cudaMemcpyDeviceToHost));
    dst = dst(cv::Rect(0, 0, dst_w, dst_h));

    parseCudaResult("free src", cudaFree(cuda_src));
    parseCudaResult("free directions", cudaFree(cuda_directions));
    parseCudaResult("free grads", cudaFree(cuda_grads));
    parseCudaResult("free dst", cudaFree(cuda_dst));

    parseCudaResult("record all stop", cudaEventRecord(all_stop));
    parseCudaResult("device sync", cudaEventSynchronize(all_stop));
    parseCudaResult("calc total elapsed time", cudaEventElapsedTime(&all_ms, all_start, all_stop));
    parseCudaResult("calc execution time", cudaEventElapsedTime(&exec_ms, exec_start, exec_stop));
    std::cout << "GPU timings" << std::endl << "    total time (ms): " << all_ms << std::endl
              << "    execution time (ms): " << exec_ms << std::endl << "    data copy time (ms): " << all_ms - exec_ms
              << std::endl;

    parseCudaResult("destroy event all_start", cudaEventDestroy(all_start));
    parseCudaResult("destroy event exec_start", cudaEventDestroy(exec_start));
    parseCudaResult("destroy event all_stop", cudaEventDestroy(all_stop));
    parseCudaResult("destroy event exec_stop", cudaEventDestroy(exec_stop));

    return 0;
}


