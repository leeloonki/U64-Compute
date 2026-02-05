#include "array_utils.h"
void print_array(const UInt64Array &arr, const std::string &name, bool show_all) {
    std::cout << name << " (shape: ";
    auto shape = arr.shape();
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        std::cout << shape.dims[i];
        if (i < shape.dims.size() - 1)
            std::cout << "x";
    }
    std::cout << ", size: " << arr.size() << ", device: ";
    std::cout << (arr.device() == DeviceType::CPU ? "CPU" : "GPU");
    std::cout << "):" << std::endl;

    if (arr.device() == DeviceType::GPU) {
        std::cout << "  [GPU data - use .to(CPU) to view contents]" << std::endl;
        return;
    }

    size_t show_count = show_all ? arr.size() : std::min(arr.size(), static_cast<size_t>(20));

    if (shape.ndim() == 1) {
        // 1D数组
        std::cout << "  [";
        for (size_t i = 0; i < show_count; ++i) {
            std::cout << arr[i];
            if (i < show_count - 1)
                std::cout << ", ";
        }
        if (!show_all && arr.size() > show_count) {
            std::cout << ", ... (+" << (arr.size() - show_count) << " more)";
        }
        std::cout << "]" << std::endl;
    } else if (shape.ndim() == 2) {
        // 2D数组
        size_t rows = shape.dims[0];
        size_t cols = shape.dims[1];
        size_t show_rows = show_all ? rows : std::min(rows, static_cast<size_t>(10));
        size_t show_cols = show_all ? cols : std::min(cols, static_cast<size_t>(10));

        std::cout << "  [";
        for (size_t i = 0; i < show_rows; ++i) {
            if (i > 0)
                std::cout << "   ";
            std::cout << "[";
            for (size_t j = 0; j < show_cols; ++j) {
                std::cout << arr.at({i, j});
                if (j < show_cols - 1)
                    std::cout << ", ";
            }
            if (!show_all && cols > show_cols) {
                std::cout << ", ... (+" << (cols - show_cols) << " more)";
            }
            std::cout << "]";
            if (i < show_rows - 1)
                std::cout << "," << std::endl;
        }
        if (!show_all && rows > show_rows) {
            std::cout << "," << std::endl << "   ... (+" << (rows - show_rows) << " more rows)";
        }
        std::cout << "]" << std::endl;
    } else {
        // 高维数组，简化显示
        std::cout << "  [";
        for (size_t i = 0; i < show_count; ++i) {
            std::cout << arr[i];
            if (i < show_count - 1)
                std::cout << ", ";
            if ((i + 1) % 10 == 0 && i < show_count - 1)
                std::cout << std::endl << "   ";
        }
        if (!show_all && arr.size() > show_count) {
            std::cout << ", ... (+" << (arr.size() - show_count) << " more)";
        }
        std::cout << "]" << std::endl;
    }
}