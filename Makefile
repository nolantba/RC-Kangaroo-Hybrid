# RCKangaroo Hybrid+SOTA+ Makefile
# Combines GPU+CPU execution with SOTA+ optimizations
# Usage:
#   make clean
#   make SM=86 USE_JACOBIAN=1 PROFILE=release -j
#   ./rckangaroo -cpu 64 -dp 16 -range 84 ...

TARGET := rckangaroo

# Toolchains
CC    := g++
NVCC  := /usr/local/cuda-12.6/bin/nvcc

# CUDA
CUDA_PATH ?= /usr/local/cuda-12.6
SM        ?= 86
USE_JACOBIAN ?= 1
USE_PERSISTENT_KERNELS ?= 0
USE_SOTA_PLUS ?= 0
PROFILE   ?= release

# Separate optimization: host vs device
# Host optimizations: aggressive inlining, native arch, link-time optimization
HOST_COPT_release := -O3 -DNDEBUG -ffunction-sections -fdata-sections -march=native -mtune=native -flto -finline-functions -funroll-loops -fopenmp
HOST_COPT_debug   := -O0 -g
HOST_COPT := $(HOST_COPT_$(PROFILE))

DEV_COPT_release := -O3
DEV_COPT_debug   := -O0 -g
DEV_COPT := $(DEV_COPT_$(PROFILE))

# Flags
CCFLAGS    := -std=c++17 -I$(CUDA_PATH)/include $(HOST_COPT) -DUSE_JACOBIAN=$(USE_JACOBIAN) -DUSE_PERSISTENT_KERNELS=$(USE_PERSISTENT_KERNELS) -DUSE_SOTA_PLUS=$(USE_SOTA_PLUS)
# Ampere (SM 86) MAX PERFORMANCE optimizations
# - L2 cache control (ca=cache-all, wb=write-back)
# - Fast math, aggressive vectorization
# - Device LTO for maximum optimization
# - Expensive optimizations for maximum speed
NVCCFLAGS  := -std=c++17 -arch=sm_$(SM) $(DEV_COPT) -Xptxas -O3 -Xptxas -dlcm=ca -Xptxas --def-load-cache=ca -Xptxas --def-store-cache=wb -Xptxas=-allow-expensive-optimizations=true -Xfatbin=-compress-all -DUSE_JACOBIAN=$(USE_JACOBIAN) -DUSE_PERSISTENT_KERNELS=$(USE_PERSISTENT_KERNELS) -DUSE_SOTA_PLUS=$(USE_SOTA_PLUS) --use_fast_math --extra-device-vectorization -dlto -rdc=true
NVCCXCOMP  := -Xcompiler -ffunction-sections -Xcompiler -fdata-sections

LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart -pthread -lcudadevrt
# Optional: Enable NVML for GPU monitoring (requires nvidia-ml-dev package)
ifdef USE_NVML
    CCFLAGS  += -DUSE_NVML
    LDFLAGS  += -lnvidia-ml
endif

# Sources (including CPU worker, save/resume system, and GPU monitoring)
SRC_CPP := RCKangaroo.cpp GpuKang.cpp CpuKang.cpp Ec.cpp Lambda.cpp utils.cpp WorkFile.cpp XorFilter.cpp GpuMonitor.cpp

# CUDA source
CU_DIR ?= .
SRC_CU := $(wildcard $(CU_DIR)/RCGpuCore.cu)

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(patsubst %.cu,%.o,$(SRC_CU))

ifeq ($(strip $(OBJ_CU)),)
  $(warning [Makefile] No RCGpuCore.cu found in $(CU_DIR). Building CPU-only.)
  OBJS := $(OBJ_CPP)
else
  OBJS := $(OBJ_CPP) $(OBJ_CU)
endif

.PHONY: all clean print-vars

all: $(TARGET)

$(TARGET): $(OBJS)
	@# Device link step for CUDA LTO
	$(NVCC) -arch=sm_$(SM) -dlto -dlink -o gpu_dlink.o $(OBJ_CU) -lcudadevrt
	@# Final host link with device-linked object
	$(CC) $(CCFLAGS) -o $@ $(OBJS) gpu_dlink.o $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

# Generic CUDA rule (.cu -> .o) with host flags via -Xcompiler
$(CU_DIR)/%.o: $(CU_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

# Explicit rule
$(CU_DIR)/RCGpuCore.o: $(CU_DIR)/RCGpuCore.cu RCGpuUtils.h Ec.h defs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) gpu_dlink.o $(TARGET)

print-vars:
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "SM=$(SM)"
	@echo "USE_JACOBIAN=$(USE_JACOBIAN)"
	@echo "PROFILE=$(PROFILE)"
	@echo "SRC_CPP=$(SRC_CPP)"
	@echo "CU_DIR=$(CU_DIR)"
	@echo "SRC_CU=$(SRC_CU)"
	@echo "OBJ_CPP=$(OBJ_CPP)"
	@echo "OBJ_CU=$(OBJ_CU)"
	@echo "OBJS=$(OBJS)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "NVCCXCOMP=$(NVCCXCOMP)"
