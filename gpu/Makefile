# 3D-PLP decoder builds.
#
#   make host   : g++ build  -> plp_gpu      (parity-checked on macOS)
#   make cuda   : nvcc build -> plp_gpu_cuda (Spark GB10, sm_121)
#
# Float flags are parity-critical and must not be relaxed:
#   host:   -ffp-contract=off   (no FMA contraction in the host part of .cu too)
#   device: --fmad=false        (no FMA contraction in kernel code)
# NEVER add -use_fast_math or any fast-math flag: bit parity against the
# CPython reference requires exact IEEE-754 semantics.

NVCC ?= /usr/local/cuda-13.0/bin/nvcc
ARCH ?= sm_121

.PHONY: host cuda clean

host:
	g++ -O3 -std=c++17 -ffp-contract=off -Wall -Wextra plp_gpu.cpp -o plp_gpu

cuda:
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) --fmad=false \
		-Xcompiler -ffp-contract=off plp_gpu.cu -o plp_gpu_cuda

clean:
	rm -f plp_gpu plp_gpu_cuda
