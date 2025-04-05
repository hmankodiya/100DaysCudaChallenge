# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -lcublas -allow-unsupported-compiler

# Directories
INCLUDE_DIRS = -Iinclude -I../include
SRC_DIR = .
KERNEL_DIR = kernel
BUILD_DIR = build

# Sources
KERNEL_SRC = $(KERNEL_DIR)/cublas_vector_scalar_add.cu
UTILS_SRC = ../utils.cu
RUNNER_SRC = run_cublas_vector_scalar_add.cu  # optional test file (if you create it)

# Object files
KERNEL_OBJ = $(BUILD_DIR)/cublas_vector_scalar_add.obj
UTILS_OBJ = $(BUILD_DIR)/utils.obj
RUNNER_OBJ = $(BUILD_DIR)/run_cublas_vector_scalar_add.obj

OBJECTS = $(KERNEL_OBJ) $(UTILS_OBJ) $(RUNNER_OBJ)

# Output binary
TARGET = run_cublas_vector_scalar_add

# Default rule
all: $(TARGET)

# Linking final binary
$(TARGET): $(OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Compile kernel source
$(KERNEL_OBJ): $(KERNEL_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile shared utils
$(UTILS_OBJ): $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Optional: Compile runner (if you have a test file)
$(RUNNER_OBJ): $(RUNNER_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
