CXXSTD := -std=c++17
include ../../../makefiles/common.mk

NCCL_HOME ?= $(abspath ../../../build)
LIB := $(CURDIR)/libep_bench_core.so
SRC := $(CURDIR)/ep_bench_core.cu

INCFLAGS := -I$(CURDIR) -I$(abspath ..) -I$(abspath ../include) -I$(NCCL_HOME)/include -I$(NCCL_HOME)/include/nccl_device -I$(abspath ../../../src/include)
EP_BENCH_NVCUFLAGS := $(NVCUFLAGS) $(INCFLAGS) --compiler-options "-fPIC"
EP_BENCH_NVLDFLAGS := -L$(NCCL_HOME)/lib -lnccl_ep -lnccl -lcupti -lpthread -lrt -ldl -lcuda \
	-Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

.PHONY: all clean

all: $(LIB)

$(LIB): $(SRC) $(CURDIR)/ep_bench_core.h
	@printf "Building   %-35s > %s\n" $(notdir $<) $@
	$(NVCC) -shared -o $@ $(EP_BENCH_NVCUFLAGS) $< $(EP_BENCH_NVLDFLAGS)

clean:
	rm -f $(LIB)
