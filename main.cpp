#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C" int canny_cpu(cv::Mat& src, const float sigma, cv::Mat& dst);

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./cuda_test <src_img> <dst_img> <sigma>" << std::endl;
    }

    std::string fname_in(argv[1]);
    std::string fname_out(argv[2]);
    float sigma = std::stof(argv[3]);

    cv::Mat src_img = cv::imread(fname_in, cv::IMREAD_GRAYSCALE);
    cv::Mat float_test, float_dest;
    src_img.convertTo(float_test, CV_32F);
    int result = canny_cpu(float_test, sigma, float_dest);

    std::cout << "Result: " << result << std::endl;

    cv::imwrite(argv[2], float_dest);
    return 0;
}
