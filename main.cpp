#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C" int canny_cpu(cv::Mat& src, const float sigma, const float low_thr, const float high_thr, cv::Mat& dst);
extern "C" int canny_gpu(cv::Mat& src, const float sigma, cv::Mat& dst);

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./cuda_test <cuda | cpu> <src_img> <dst_img> <sigma> <low_thr> <high_thr>" << std::endl;
    }

    std::string device(argv[1]);
    std::string fname_in(argv[2]);
    std::string fname_out(argv[3]);
    float sigma = std::stof(argv[4]);
    float low_thr = std::stof(argv[5]);
    float high_thr = std::stof(argv[6]);

    cv::Mat src_img = cv::imread(fname_in, cv::IMREAD_GRAYSCALE);
    cv::Mat float_test, float_dest;
    src_img.convertTo(float_test, CV_32F);
    int result = 1;
    std::cout << "Device: " << device << std::endl;
    if (!device.compare("cpu")) {
        result = canny_cpu(float_test, sigma, low_thr, high_thr, float_dest);
    } else if (!device.compare("cuda")) {
        result = canny_gpu(float_test, sigma, float_dest);
    } else {
        std::cout << "Unknown device " << device << std::endl;
    }

    std::cout << "Result: " << result << std::endl;

    cv::imwrite(fname_out, float_dest);
    return 0;
}
