# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -lcublas -allow-unsupported-compiler

# Directories
INCLUDE_DIRS = -I./include -I../include
SRC_DIR = .
KERNEL_DIR = kernels
BUILD_DIR = build

# Sources
KERNEL_SRC = $(KERNEL_DIR)/center_vector.cu
UTILS_SRC = ../utils.cu
RUNNER_SRC = run_center_vector.cu  # optional test file (if you create it)

# Object files
KERNEL_OBJ = $(BUILD_DIR)/center_vector.obj
UTILS_OBJ = $(BUILD_DIR)/utils.obj
RUNNER_OBJ = $(BUILD_DIR)/run_center_vector.obj

OBJECTS = $(KERNEL_OBJ) $(UTILS_OBJ) $(RUNNER_OBJ)

# Output binary
TARGET = run_center_vector

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
