#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C" int canny_cpu(cv::Mat& src, const float sigma, const float low_thr, const float high_thr, cv::Mat& dst, const int nthreads);
extern "C" int canny_gpu(cv::Mat& src, const float sigma, const float low_thr, const float high_thr, cv::Mat& dst);

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./cuda_test <src_img> <sigma> <low_thr> <high_thr>" << std::endl;
    }

    std::string fname_in(argv[1]);
    float sigma = std::stof(argv[2]);
    float low_thr = std::stof(argv[3]);
    float high_thr = std::stof(argv[4]);

    cv::Mat src_img = cv::imread(fname_in, cv::IMREAD_GRAYSCALE);
    cv::Mat float_test, float_dest;
    src_img.convertTo(float_test, CV_32FC1);
    int result = 0;
    result |= canny_cpu(float_test, sigma, low_thr, high_thr, float_dest, 1);
    cv::imwrite("out_cpu.png", float_dest);

    result |= canny_cpu(float_test, sigma, low_thr, high_thr, float_dest, NUM_THREADS);

    result |= canny_gpu(float_test, sigma, low_thr, high_thr, float_dest);
    cv::imwrite("out_gpu.png", float_dest);

    return 0;
}
