#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <deque>

namespace cpu {
    int im2col(const float* src, const int src_w, const int src_h, const int k, const int y, const int x, float* dst) {
        if (y + k > src_h || x + k > src_w || y < 0 || x < 0) {
            return 1;
        }

        unsigned chunk = k * sizeof(float);
        const float* src_ = src + y * src_w + x;
        for (int i = 0; i < k; ++i) {
            memcpy(dst, src_, chunk);
            src_ += src_w;
            dst += k;
        }

        return 0;
    }

    float dot_product(const float* src, const float* kernel, const int N) {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += src[i] * kernel[i];
        }
        return sum;
    }

    int gradient(
            cv::Mat &src,
            const float sigma,
            const int k,
            cv::Mat &dst,
            cv::Mat &dirs
    ) {
        const int src_h = src.size[0];
        const int src_w = src.size[1];
        const int dst_h = dst.size[0];
        const int dst_w = dst.size[1];
        const float* src_ = src.ptr<float>();
        float* dst_ = dst.ptr<float>();
        unsigned char* dirs_ = dirs.ptr<unsigned char>();

        float* kernelx = (float*) malloc(k * k * sizeof(float));
        float* kernely = (float*) malloc(k * k * sizeof(float));

        float* kx_ = kernelx;
        float* ky_ = kernely;
        float norm = -1.0 / (2.0 * M_PI * sigma * sigma * sigma * sigma);
        float expnorm = 2.0 * sigma * sigma;
        for (int i = 0; i < k; ++i) {
            float y = k / 2 - i;
            for (int j = 0; j < k; ++j) {
                float x = k / 2 - j;
                float scary = norm * std::pow(M_E, -(x * x + y * y) / expnorm);
                *kx_++ = x * scary;
                *ky_++ = y * scary;
            }
        }

        int result = 0;
        const int N = k * k;
#pragma omp parallel default(none) shared(dst_h, dst_w, k, src_, src_w, src_h, kernelx, kernely, N, result, dst_, dirs_) num_threads(NUM_THREADS)
        {
            float* src_vec = (float*) malloc(k * k * sizeof(float));

#pragma omp for
            for (int i = 0; i < dst_h; ++i) {
                int index = i * dst_w;
                for (int j = 0; j < dst_w; ++j, ++index) {
                    int im2col_res = im2col(src_, src_w, src_h, k, i, j, src_vec);
#pragma omp atomic
                    result |= im2col_res;

                    float sumx = dot_product(src_vec, kernelx, N);
                    float sumy = dot_product(src_vec, kernely, N);
                    dst_[index] = std::sqrt(sumx * sumx + sumy * sumy);
                    if (sumx != 0.0f || sumy != 0.0f) {
                        float angle = std::atan2(sumy, sumx);
                        std::vector<double> dists = {
                                std::fabs(angle - M_PI),
                                std::fabs(angle - 3 * M_PI_4),
                                std::fabs(angle - M_PI_2),
                                std::fabs(angle - M_PI_4),
                                std::fabs(angle),
                                std::fabs(angle + M_PI),
                                std::fabs(angle + 3 * M_PI_4),
                                std::fabs(angle + M_PI_2),
                                std::fabs(angle + M_PI_4)
                        };
                        int min_index = (int) (std::min_element(dists.begin(), dists.end()) - dists.begin());
                        switch (min_index) {
                            case 1:
                            case 8:
                                dirs_[index] = 255;
                                break;
                            case 2:
                            case 7:
                                dirs_[index] = 128;
                                break;
                            case 3:
                            case 6:
                                dirs_[index] = 192;
                                break;
                            default:
                                dirs_[index] = 64;
                        }
                    }
                }
            }

            free(src_vec);
        }

        free(kernelx);
        free(kernely);
        return result;
    }

    int nonmax(cv::Mat &gradient, cv::Mat &directions, cv::Mat &dst) {
        const int dst_h = dst.size[0];
        const int dst_w = dst.size[1];
        const int grad_h = gradient.size[0];
        const int grad_w = gradient.size[1];
        float* dst_ = dst.ptr<float>();
        unsigned char* dirs_ = directions.ptr<unsigned char>();
        float* grad_ = gradient.ptr<float>();

#pragma omp parallel default(none) shared(dst_h, dst_w, grad_h, grad_w, dst_, dirs_, grad_, std::cout) num_threads(NUM_THREADS)
        {

#pragma omp for
            for (int i = 0; i < dst_h; ++i) {
                int dst_index = i * dst_w;
                int grad_index = i * grad_w;
                for (int j = 0; j < dst_w; ++j, ++dst_index, ++grad_index) {
                    float m = 0;
                    switch (dirs_[dst_index]) {
                        case 128:
                            m = std::fmax(grad_[i * grad_w + j + 1], grad_[(i + 2) * grad_w + j + 1]);
                            break;
                        case 64:
                            m = std::fmax(grad_[(i + 1) * grad_w + j], grad_[(i + 1) * grad_w + j + 2]);
                            break;
                        case 255:
                            m = std::fmax(grad_[i * grad_w + j + 2], grad_[(i + 2) * grad_w + j]);
                            break;
                        case 192:
                            m = std::fmax(grad_[(i + 2) * grad_w + j + 2], grad_[i * grad_w + j]);
                            break;
                    }

                    if (grad_[(i + 1) * grad_w + j + 1] > m) {
                        dst_[dst_index] = grad_[(i + 1) * grad_w + j + 1];
                    }
                }
            }
        }

        return 0;
    }

    int standartize(cv::Mat& src, cv::Mat& dst) {
        const int dst_h = dst.size[0];
        const int dst_w = dst.size[1];
        float* dst_ = dst.ptr<float>();
        float* src_ = src.ptr<float>();

        float min = FLT_MAX, max = FLT_MIN;
        for (int i = 0; i < dst_h * dst_w; ++i) {
            if (src_[i] > max) {
                max = src_[i];
            }
            if (src_[i] < min) {
                min = src_[i];
            }
        }

        for (int i = 0; i < dst_h * dst_w; ++i) {
            dst_[i] -= min;
            dst_[i] /= max;
            dst_[i] *= 255;
        }

        return 0;
    }

    int hysteresis(cv::Mat& src, cv::Mat& dst, const float low_thr, const float high_thr) {
        const int dst_h = dst.size[0];
        const int dst_w = dst.size[1];
        float* src_ = src.ptr<float>();
        float* dst_ = dst.ptr<float>();

#pragma omp parallel default(none) shared(dst_h, dst_w, dst_, src_, low_thr, high_thr) num_threads(NUM_THREADS)
        {

#pragma omp for
            for (int i = 0; i < dst_h; ++i) {
                for (int j = 0; j < dst_w; ++j) {
                    if (src_[i * dst_w + j] < low_thr) {
                        dst_[i * dst_w + j] = 0;
                    } else if (src_[i * dst_w + j] < high_thr) {
                        dst_[i * dst_w + j] = 128;
                    } else {
                        dst_[i * dst_w + j] = 255;
                    }
                }
            }
        }

        std::deque<std::pair<int, int>> to_check;
        for (int i = 0; i < dst_h; ++i) {
            for (int j = 0; j < dst_w; ++j) {
                if (dst_[i * dst_w + j] == 255) {
                    to_check.emplace_back(i, j);
                }
            }
        }

        while (!to_check.empty()) {
            std::pair<int, int> idx = to_check.front();
            to_check.pop_front();
            int i = idx.first;
            int j = idx.second;
            if (i > 0) {
                if (j > 0) {
                    if (dst_[(i - 1) * dst_w + j - 1] == 128) {
                        dst_[(i - 1) * dst_w + j - 1] = 255;
                        to_check.emplace_back(i - 1, j - 1);
                    }
                }
                if (dst_[(i - 1) * dst_w + j] == 128) {
                    dst_[(i - 1) * dst_w + j] = 255;
                    to_check.emplace_back(i - 1, j);
                }
                if (j < dst_w - 1) {
                    if (dst_[(i - 1) * dst_w + j + 1] == 128) {
                        dst_[(i - 1) * dst_w + j + 1] = 255;
                        to_check.emplace_back(i - 1, j + 1);
                    }
                }
            }

            if (j > 0) {
                if (dst_[i * dst_w + j - 1] == 128) {
                    dst_[i * dst_w + j - 1] = 255;
                    to_check.emplace_back(i, j - 1);
                }
            }
            if (j < dst_w - 1) {
                if (dst_[i * dst_w + j + 1] == 128) {
                    dst_[i * dst_w + j + 1] = 255;
                    to_check.emplace_back(i, j + 1);
                }
            }

            if (i < dst_h - 1) {
                if (j > 0) {
                    if (dst_[(i + 1) * dst_w + j - 1] == 128) {
                        dst_[(i + 1) * dst_w + j - 1] = 255;
                        to_check.emplace_back(i + 1, j - 1);
                    }
                }
                if (dst_[(i + 1) * dst_w + j] == 128) {
                    dst_[(i + 1) * dst_w + j] = 255;
                    to_check.emplace_back(i + 1, j);
                }
                if (j < dst_w - 1) {
                    if (dst_[(i + 1) * dst_w + j + 1] == 128) {
                        dst_[(i + 1) * dst_w + j + 1] = 255;
                        to_check.emplace_back(i + 1, j + 1);
                    }
                }
            }
        }

        for (int i = 0; i < dst_h * dst_w; ++i) {
            if (dst_[i] == 128) {
                dst_[i] = 0;
            }
        }

        return 0;
    }
}

extern "C" int canny_cpu(cv::Mat& src, const float sigma, const float low_thr, const float high_thr, cv::Mat& dst) {
    cv::Mat src_ = src;
    if (!src.isContinuous() || (src.type() & CV_MAT_DEPTH_MASK) != CV_32F) {
        src.convertTo(src_, CV_32F);
    }
    cv::Mat _src_;
    const int k = (int) std::ceil(sigma) * 6 + 1;
    cv::copyMakeBorder(src_, _src_, k/2, k/2, k/2, k/2, CV_HAL_BORDER_REFLECT);

    const int src_h = _src_.size[0];
    const int src_w = _src_.size[1];
    if (k >= src_h + 1 || k >= src_w + 1) {
        return 1;
    }
    const int dst_h = src_h - k + 1;
    const int dst_w = src_w - k + 1;
    dst = cv::Mat(dst_h, dst_w, CV_32FC1, cv::Scalar(0));
    cv::Mat directions(dst_h, dst_w, CV_8UC1, cv::Scalar(0));
    cv::Mat grads;

    int ret = cpu::gradient(_src_, sigma, k, dst, directions);
    cv::copyMakeBorder(dst, grads, 1, 1, 1, 1, CV_HAL_BORDER_REFLECT);
    dst = cv::Mat(dst_h, dst_w, CV_32FC1, cv::Scalar(0));
    ret += cpu::nonmax(grads, directions, dst);

    ret += cpu::hysteresis(dst, dst, low_thr, high_thr);

    return ret;
}
