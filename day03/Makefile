# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -allow-unsupported-compiler -lcublas

# Directories
INCLUDE_DIRS = -Iinclude -I../include -I../include
KERNELS_DIR = kernels
SRC_DIR = .
BUILD_DIR = build

# Source files
KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCE = run_vector_vector_add.cu
UTILS_SOURCE = ../utils.cu

# Object files
KERNEL_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%.obj,$(KERNEL_SOURCES))
ROOT_OBJECT = $(BUILD_DIR)/run_vector_vector_add.obj
UTILS_OBJECT = $(BUILD_DIR)/utils.obj

OBJECTS = $(KERNEL_OBJECTS) $(ROOT_OBJECT) $(UTILS_OBJECT)

# Output binary
TARGET = vector_vector_add

# Default rule
all: $(TARGET)

# Linking the final binary
$(TARGET): $(OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Rule to compile run_vector_vector_add.cu
$(BUILD_DIR)/run_vector_vector_add.obj: $(SRC_DIR)/run_vector_vector_add.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Rule to compile utils.cu
$(BUILD_DIR)/utils.obj: ../utils.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Rule to compile kernel .cu files
$(BUILD_DIR)/%.obj: $(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
