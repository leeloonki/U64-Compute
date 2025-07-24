# UInt64 Compute Library Makefile
# High-Performance mod 2^64 Ring Arithmetic Library

# Compiler settings - Auto-detect real GCC
GCC_CANDIDATES = gcc-15 gcc-14 gcc-13 gcc-12 gcc-11 gcc-10 gcc-9 gcc
GCC_FOUND := $(shell for gcc in $(GCC_CANDIDATES); do \
	if command -v $$gcc >/dev/null 2>&1 && $$gcc --version | grep -q "Free Software Foundation"; then \
		echo $$gcc; break; \
	fi; \
done)
CXX = $(if $(GCC_FOUND),$(GCC_FOUND),g++)
NVCC = nvcc

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -isysroot $(shell xcrun --show-sdk-path)
    OPENMP_LIBS = -fopenmp -lstdc++
else
    CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra
    OPENMP_LIBS = -fopenmp -lstdc++
endif
NVCCFLAGS = -std=c++17 -O3 -arch=sm_60 -Xcompiler -fopenmp

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Include paths
INCLUDES = -I$(INCDIR)

# Source files
CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
CPP_OBJECTS = $(CPP_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CU_OBJECTS = $(CU_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
ALL_OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

# Main executable
TARGET = $(BINDIR)/functional_test
MAIN_SOURCE = functional_test.cpp
MAIN_OBJECT = $(OBJDIR)/functional_test.o

# Library files
STATIC_LIB = $(BINDIR)/libuint64compute.a
SHARED_LIB = $(BINDIR)/libuint64compute.so

# CUDA detection
CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo "yes" || echo "no")

# Conditional CUDA support
ifeq ($(CUDA_AVAILABLE),yes)
    CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
    CXXFLAGS += -DCUDA_ENABLED -I$(CUDA_PATH)/include
    NVCCFLAGS += -DCUDA_ENABLED
    CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas
    USE_CUDA = true
else
    USE_CUDA = false
    CUDA_LIBS =
    CU_OBJECTS =
endif

# OpenMP libs already defined above in platform-specific section

# SIMD flags (automatically detected by -march=native)
# Additional SIMD flags can be added here if needed
# SIMD_FLAGS = -mavx2 -mfma

# All targets
.PHONY: all clean test install uninstall help info debug release simdinfo simd-tool benchmark

# Default target
all: info $(TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(BINDIR):
	@mkdir -p $(BINDIR)

# Build information
info:
	@echo "=== UInt64 Compute Library Build Info ==="
	@echo "Compiler: $(CXX)"
	@echo "C++ Flags: $(CXXFLAGS)"
	@echo "CUDA Available: $(CUDA_AVAILABLE)"
ifeq ($(USE_CUDA),true)
	@echo "NVCC: $(NVCC)"
	@echo "NVCC Flags: $(NVCCFLAGS)"
	@echo "CUDA Libraries: $(CUDA_LIBS)"
endif
	@echo "OpenMP: Enabled"
	@echo "SIMD: Auto-detected with -march=native"
	@echo "==========================================="
	@echo

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files (if CUDA is available)
ifeq ($(USE_CUDA),true)
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
endif

# Compile main test file
$(MAIN_OBJECT): $(MAIN_SOURCE) | $(OBJDIR)
	@echo "Compiling main test file..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build static library
$(STATIC_LIB): $(ALL_OBJECTS) | $(BINDIR)
	@echo "Creating static library..."
	ar rcs $@ $(ALL_OBJECTS)

# Build shared library
$(SHARED_LIB): $(ALL_OBJECTS) | $(BINDIR)
	@echo "Creating shared library..."
ifeq ($(USE_CUDA),true)
	$(CXX) -shared -o $@ $(ALL_OBJECTS) $(OPENMP_LIBS) $(CUDA_LIBS)
else
	$(CXX) -shared -o $@ $(ALL_OBJECTS) $(OPENMP_LIBS)
endif

# Build main executable
$(TARGET): $(MAIN_OBJECT) $(STATIC_LIB) | $(BINDIR)
	@echo "Linking executable..."
ifeq ($(USE_CUDA),true)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BINDIR) -luint64compute $(OPENMP_LIBS) $(CUDA_LIBS)
else
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BINDIR) -luint64compute $(OPENMP_LIBS)
endif
	@echo "Build completed successfully!"
	@echo "Run './$(TARGET)' to test the library"

# Build only the libraries
libs: $(STATIC_LIB) $(SHARED_LIB)

# Test target
test: $(TARGET)
	@echo "Running functional tests..."
	./$(TARGET)


# Debug build
debug: CXXFLAGS += -g -DDEBUG -O0
debug: NVCCFLAGS += -g -DDEBUG -O0
debug: $(TARGET)

# Release build (default is already optimized)
release: $(TARGET)

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "Clean completed."

# Install (system-wide installation)
install: $(STATIC_LIB) $(SHARED_LIB)
	@echo "Installing UInt64 Compute Library..."
	sudo mkdir -p /usr/local/include/uint64_compute
	sudo cp -r $(INCDIR)/* /usr/local/include/uint64_compute/
	sudo cp $(STATIC_LIB) /usr/local/lib/
	sudo cp $(SHARED_LIB) /usr/local/lib/
	sudo ldconfig
	@echo "Installation completed."

# Uninstall
uninstall:
	@echo "Uninstalling UInt64 Compute Library..."
	sudo rm -rf /usr/local/include/uint64_compute
	sudo rm -f /usr/local/lib/libuint64compute.a
	sudo rm -f /usr/local/lib/libuint64compute.so
	sudo ldconfig
	@echo "Uninstallation completed."

# Show help
help:
	@echo "UInt64 Compute Library - Build System"
	@echo "====================================="
	@echo "Available targets:"
	@echo "  all       - Build everything (default)"
	@echo "  libs      - Build only static and shared libraries"
	@echo "  test      - Build and run functional tests"
	@echo "  debug     - Build with debug information"
	@echo "  release   - Build optimized release version"
	@echo "  clean     - Remove all build files"
	@echo "  install   - Install library system-wide (requires sudo)"
	@echo "  uninstall - Remove system-wide installation (requires sudo)"
	@echo "  info      - Show build configuration"
	@echo "  simdinfo  - Show detailed SIMD detection information"
	@echo "  simd-tool - Build and run SIMD information tool"
	@echo "  benchmark - Build and run performance benchmarks"
	@echo "  sysinfo   - Show complete system information"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Build features:"
	@echo "  • C++17 standard"
	@echo "  • Automatic SIMD optimization (-march=native)"
	@echo "  • OpenMP parallel processing"
	@echo "  • Optional CUDA support (auto-detected)"
	@echo "  • High optimization (-O3)"
	@echo ""
	@echo "Examples:"
	@echo "  make          # Build everything"
	@echo "  make test     # Build and test"
	@echo "  make clean    # Clean build files"
	@echo "  make install  # System-wide install"

# Show detailed compiler SIMD flags
simdinfo:
	@echo "=== Compiler SIMD Detection ==="
	@echo "Compiler: $(CXX)"
	@echo "OS: $(UNAME_S)"
	@echo "Architecture: $$(uname -m)"
	@echo "Target Architecture:"
	@$(CXX) -march=native -v -E - < /dev/null 2>&1 | grep -E "Target:" | head -1 || echo "  Unable to detect target"
	@echo ""
	@echo "ARM64 SIMD Features Detected:"
	@$(CXX) -march=native -dM -E - < /dev/null | grep -E "__ARM_NEON|__ARM_FEATURE" | sed 's/#define /  /' || echo "  No ARM SIMD features detected"
	@echo ""
	@echo "x86-64 SIMD Features (if applicable):"
	@$(CXX) -march=native -dM -E - < /dev/null | grep -E "__SSE|__AVX|__FMA|__BMI" | sed 's/#define /  /' | sort || echo "  No x86 SIMD features detected"
	@echo ""
	@echo "CPU Architecture Summary:"
	@echo "  Target: $$($(CXX) -march=native -v -E - < /dev/null 2>&1 | grep 'Target:' | cut -d' ' -f2 || echo 'Unknown')"
	@echo "  NEON Support: $$($(CXX) -march=native -dM -E - < /dev/null | grep -q '__ARM_NEON' && echo 'YES' || echo 'NO')"
	@echo "  SVE Support: $$($(CXX) -march=native -dM -E - < /dev/null | grep -q '__ARM_FEATURE_SVE' && echo 'YES' || echo 'NO')"
	@echo "  AVX-512 Support: $$($(CXX) -march=native -dM -E - < /dev/null | grep -q '__AVX512F__' && echo 'YES' || echo 'NO')"
	@echo "  Vector Width: $$($(CXX) -march=native -dM -E - < /dev/null | grep '__ARM_NEON' > /dev/null && echo '128 bits' || ($(CXX) -march=native -dM -E - < /dev/null | grep '__AVX512F__' > /dev/null && echo '512 bits' || ($(CXX) -march=native -dM -E - < /dev/null | grep '__AVX' > /dev/null && echo '256 bits' || echo 'Unknown')))"
ifeq ($(UNAME_S),Darwin)
	@echo "  macOS CPU Info: $$(sysctl -n machdecode.cpu.brand_string 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
	@echo "  CPU Features: $$(sysctl -n machdep.cpu.features 2>/dev/null | tr ' ' '\n' | sort | tr '\n' ' ' || echo 'Unknown')"
	@echo "  Leaf7 Features: $$(sysctl -n machdep.cpu.leaf7_features 2>/dev/null | tr ' ' '\n' | sort | tr '\n' ' ' || echo 'Unknown')"
else
	@echo "  CPU Model: $$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | sed 's/^ *//' || echo 'Unknown')"
	@echo "  CPU Flags: $$(grep '^flags' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | tr ' ' '\n' | grep -E 'avx|sse|neon|sve' | sort | tr '\n' ' ' || echo 'Unknown')"
endif
	@echo "=========================="
	@echo

# Show detailed system and build information  
# Build SIMD information tool
simd-tool: $(STATIC_LIB) | $(BINDIR)
	@echo "Building SIMD information tool..."
ifeq ($(USE_CUDA),true)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/simd_info simd_info.cpp -L$(BINDIR) -luint64compute $(OPENMP_LIBS) $(CUDA_LIBS)
else
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/simd_info simd_info.cpp -L$(BINDIR) -luint64compute $(OPENMP_LIBS)
endif
	@echo "SIMD info tool built successfully!"
	@echo "Run './bin/simd_info' to see detailed SIMD information"

# Build and run benchmark
benchmark: $(STATIC_LIB) | $(BINDIR)
	@echo "Building benchmark tool..."
ifeq ($(USE_CUDA),true)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/benchmark benchmark.cpp -L$(BINDIR) -luint64compute $(OPENMP_LIBS) $(CUDA_LIBS)
else
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/benchmark benchmark.cpp -L$(BINDIR) -luint64compute $(OPENMP_LIBS)
endif
	@echo "Running benchmark..."
	@echo "This may take a few minutes..."
	@./$(BINDIR)/benchmark

sysinfo: info simdinfo
	@echo "=== System Information ==="
	@echo "OS: $(UNAME_S)"
	@echo "Architecture: $$(uname -m)"
ifeq ($(UNAME_S),Darwin)
	@echo "macOS Version: $$(sw_vers -productVersion 2>/dev/null || echo 'Unknown')"
	@echo "CPU Info: $$(sysctl -n machdecode.cpu.brand_string 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
	@echo "CPU Cores: $$(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
	@echo "Available Memory: $$(echo "scale=2; $$(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024" | bc 2>/dev/null || echo 'Unknown')GB"
else
	@echo "Linux Distro: $$(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo 'Unknown')"
	@echo "CPU Info: $$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | sed 's/^ *//' || echo 'Unknown')"
	@echo "CPU Cores: $$(nproc 2>/dev/null || echo 'Unknown')"
	@echo "Available Memory: $$(free -h 2>/dev/null | grep '^Mem:' | awk '{print $$2}' || echo 'Unknown')"
endif
ifeq ($(CUDA_AVAILABLE),yes)
	@echo "GPU Info:"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "  CUDA available but nvidia-smi not found"
endif
	@echo "=========================="