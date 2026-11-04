# blockproof — CPU spike (cp-147). Flags mirror the host build (gpu/Makefile,
# -ffp-contract=off is required for shared-protocol float semantics).
#
#   make            : g++ build  -> plp_block       (CPU reference, frozen)
#   make cuda       : nvcc build -> plp_block_cuda  (Spark GB10, sm_121)
#
# Float flags are parity-critical (same protocol as gpu/Makefile):
#   --fmad=false + -Xcompiler -ffp-contract=off. NEVER add -use_fast_math.

CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -ffp-contract=off
NVCC ?= /usr/local/cuda-13.0/bin/nvcc
ARCH ?= sm_121

plp_block: plp_block.cpp ../gpu/plp_common.hpp
	$(CXX) $(CXXFLAGS) plp_block.cpp -o plp_block

cuda: plp_block_cuda.cu ../gpu/plp_common.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) --fmad=false \
		-Xcompiler -ffp-contract=off plp_block_cuda.cu -o plp_block_cuda

clean:
	rm -f plp_block plp_block_cuda

.PHONY: cuda clean
