#include <iostream>

extern "C" void fill_value(int* res);

int main() {
    std::cout << "Going to call GPU function..." << std::endl;
    int res = -1;
    fill_value(&res);
    std::cout << "Result: " << res << std::endl;
    return 0;
}
