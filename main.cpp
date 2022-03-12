#include <iostream>

extern "C" void fill_value(int* res);

int main() {
    std::cout << "Input number" << std::endl;
    int x;
    std::cin >> x;
    int res = x;
    fill_value(&res);
    std::cout << x << " * 100 = " << res << std::endl;
    return 0;
}
