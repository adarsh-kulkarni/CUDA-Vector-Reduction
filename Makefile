# Add source files here
EXECUTABLE	:= runtimeReduce
# Cuda source files (compiled with cudacc)
CUFILES		:= runtimeReduce.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \

################################################################################
# Rules and targets

include ../../common/common.mk

#The \ at the end of the lines is used for multi line comments in makefile.Starts from NVCC.NVCC has to be uncommented

#NVCC = /usr/local/cuda/bin/nvcc\
CUDAPATH = /usr/local/cuda\
#CUTILPATH = -I$(NVIDIA_SDK)/C/common/inc/cutil_inline.h\
#NVCCFLAGS = -I$(CUDAPATH)/include -I$(NVIDIA_SDK)/C/common/inc -gencode-arch=compute_20,code=sm_20,compute_20\
NVCCFLAGS = -I$(CUDAPATH)/include -I$(NVIDIA_SDK)/C/common/inc -gencode arch=compute_20,code=sm_20\
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm\
\
\
atomicReduce:\
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o atomicReduce atomicReduce.cu\


