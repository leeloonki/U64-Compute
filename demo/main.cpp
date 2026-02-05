#include <iostream>
#include "u64_compute.h"
using namespace u64comp::np;

int main() {
    std::cout << "hello U64 compute" << std::endl;

    U64Array a({3, 4});
    a.print();
    U64Array b({3, 4});
    b.print();

    a.fill(2);
    a.print();

    b.fill(3);
    b.print();
    U64Array c = a + b;
    c.print("1", true);

    return 0;
}