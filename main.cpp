#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C" void fill_value(int* res);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Please specify input / output images" << std::endl;
    }

    std::string fname_in(argv[1]);
    cv::Mat src_img = cv::imread(fname_in, cv::IMREAD_COLOR);
    cv::imwrite("out.png", src_img);
    return 0;
}
