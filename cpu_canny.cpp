#include <opencv2/core.hpp>
#include <cmath>

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

int gradient(cv::Mat& src, const float sigma, const int k, cv::Mat& dst) {
    const int src_h = src.size[0];
    const int src_w = src.size[1];
    const int dst_h = dst.size[0];
    const int dst_w = dst.size[1];
    const float* src_ = src.ptr<float>();
    float* dst_ = dst.ptr<float>();

    float* src_vec = (float *) malloc(k * k * sizeof(float));
    float* kernelx = (float *) malloc(k * k * sizeof(float));
    float* kernely = (float *) malloc(k * k * sizeof(float));

    float* kx_ = kernelx;
    float* ky_ = kernely;
    float norm = -1.0 / (2.0 * M_PI * sigma * sigma * sigma * sigma);
    float expnorm = 2.0 * sigma * sigma;
    for (int i = 0; i < k; ++i) {
        float y = k/2 - i;
        for (int j = 0; j < k; ++j) {
            float x = k/2 - j;
            float scary = norm * std::pow(M_E, -(x * x + y * y) / expnorm);
            *kx_++ = x * scary;
            *ky_++ = y * scary;
        }
    }

    int result = 0;
    const int N = k * k;
    for (int i = 0; i < dst_h; ++i) {
        for (int j = 0; j < dst_w; ++j) {
            result |= im2col(src_, src_w, src_h, k, i, j, src_vec);
            float sumx = dot_product(src_vec, kernelx, N);
            float sumy = dot_product(src_vec, kernely, N);
            *dst_++ = std::sqrt(sumx * sumx + sumy * sumy);
            float angle = std::atan2(sumy, sumx);
        }
    }

    free(src_vec);
    free(kernelx);
    free(kernely);
    return result;
}

extern "C" int canny_cpu(cv::Mat& src, const float sigma, cv::Mat& dst) {
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
    dst = cv::Mat(dst_h, dst_w, CV_32F);

    return gradient(_src_, sigma, k, dst);
}
