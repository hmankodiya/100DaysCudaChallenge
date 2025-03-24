# CUDA compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++11

# Folders
INCLUDE_DIR = include
KERNEL_DIR = kernels

# Source files
SRCS = main.cu $(KERNEL_DIR)/utils.cu $(KERNEL_DIR)/mat_add.cu
OBJS = $(SRCS:.cu=.o)

# Output binary
TARGET = main

# Default rule
all: $(TARGET)

# Linking all object files
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile individual .cu files to .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Clean object files and binary
clean:
	rm -f $(OBJS) $(TARGET)
