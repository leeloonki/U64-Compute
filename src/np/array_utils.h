#pragma once
#include "array.h"
#include <iostream>
using UInt64Array = u64comp::np::U64Array;
using DeviceType = u64comp::device::DeviceType;
void print_array(const UInt64Array &arr, const std::string &name, bool show_all);