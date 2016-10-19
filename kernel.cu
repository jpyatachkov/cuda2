#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>

#define PHASE_VEL   1
#define OUTER_FORSE 10

#define STEP_X 0.5
#define STEP_Y 0.5
#define STEP_T 0.1

static double *hostDataX   = nullptr, *hostDataY   = nullptr, *hostDataZ   = nullptr;
static double *devDataX	   = nullptr, *devDataY    = nullptr, *devDataZ    = nullptr;
static double *devDataBufX = nullptr, *devDataBufY = nullptr, *devDataBufZ = nullptr;

static void _cpuFree() {
	if (::hostDataX)
		std::free((void *)::hostDataX);

	if (::hostDataY)
		std::free((void *)::hostDataY);

	if (::hostDataZ)
		std::free((void *)::hostDataZ);
}

#define cudaCheck
static void _gpuFree() {
	if (::devDataX)
		cudaCheck(cudaFree((void *)::devDataX));

	if (::devDataY)
		cudaCheck(cudaFree((void *)::devDataY));

	if (::devDataZ)
		cudaCheck(cudaFree((void *)::devDataZ));

	if (::devDataBufX)
		cudaCheck(cudaFree((void *)::devDataBufX));

	if (::devDataBufY)
		cudaCheck(cudaFree((void *)::devDataBufY));

	if (::devDataBufZ)
		cudaCheck(cudaFree((void *)::devDataBufZ));
}

/*
* CUDA errors catching block
*/

static void _checkCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define cudaCheck(value) _checkCudaErrorAux(__FILE__, __LINE__, #value, value)

static void _checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;

	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;

	system("pause");

	_cpuFree();
	_gpuFree();

	exit(1);
}

/*
 * CUDA kernel block
 */

__global__ void kernel(double * __restrict__ z, double * __restrict__ y, double * __restrict__ x,
					   double * __restrict__ yBuf, double * __restrict__ xBuf,
					   const std::size_t size,
					   const double phaseVelocity, const double outerForse,
					   const double stepX, const double stepY,
					   const double stepT, const double maxTime) {
	for (auto t = 0.0; t < maxTime; t += stepT) {
		auto idx = threadIdx.x + blockIdx.x * blockDim.x;

		if (idx < size) {
			xBuf[idx] = (x[idx + 1] + x[idx - 1] + 2.0 * x[idx]) / (stepX * stepX);
			yBuf[idx] = (y[idx + 1] + y[idx - 1] - 2.0 * y[idx]) / (stepY * stepY);
			z[idx]    = (stepT * stepT) * (phaseVelocity * phaseVelocity * (xBuf[idx] + yBuf[idx]) + outerForse);
			printf("%d\n", z[idx]);

			__syncthreads();

			x[idx] = xBuf[idx];
			y[idx] = yBuf[idx];

			__syncthreads();
		}
	}
}

/*
 * Init
 */

int cpuInit(std::size_t size) {
	::hostDataX = (double *)std::calloc(size, sizeof(double));
	if (!::hostDataX)
		return 1;

	::hostDataY = (double *)std::calloc(size, sizeof(double));
	if (!::hostDataY)
		return 1;

	::hostDataZ = (double *)std::calloc(size, sizeof(double));
	if (!::hostDataZ)
		return 1;

	std::memset(::hostDataX, 0, size);
	std::memset(::hostDataY, 0, size);
	std::memset(::hostDataZ, 0, size);

	return 0;
}

void gpuInit(std::size_t size) {
	auto byteSize = size * sizeof(double);

	cudaCheck(cudaMalloc((void **)&::devDataX, byteSize));
	cudaCheck(cudaMalloc((void **)&::devDataY, byteSize));
	cudaCheck(cudaMalloc((void **)&::devDataZ, byteSize));
	cudaCheck(cudaMalloc((void **)&::devDataBufX, byteSize));
	cudaCheck(cudaMalloc((void **)&::devDataBufY, byteSize));
	cudaCheck(cudaMalloc((void **)&::devDataBufZ, byteSize));

	cudaCheck(cudaMemset(::devDataX, 0, byteSize));
	cudaCheck(cudaMemset(::devDataY, 0, byteSize));
	cudaCheck(cudaMemset(::devDataZ, 0, byteSize));
	cudaCheck(cudaMemset(::devDataBufX, 0, byteSize));
	cudaCheck(cudaMemset(::devDataBufY, 0, byteSize));
	cudaCheck(cudaMemset(::devDataBufZ, 0, byteSize));
}

/*
 * Main
 */

int main() {
	const std::size_t size = 100;
	const std::size_t time = 10;

	const auto maxTime = time / STEP_T;

	if (cpuInit(size)) {
		_cpuFree();
		return 1;
	}

	gpuInit(size);

	dim3 nBlocks(1);
	dim3 nThreads(256);

	kernel <<<nBlocks, nThreads>>> (devDataZ, devDataY, devDataX, devDataBufY, devDataBufX, size,
									PHASE_VEL, OUTER_FORSE, STEP_X, STEP_Y, STEP_T, maxTime);

	cudaCheck(cudaMemcpy(hostDataZ, devDataZ, size * sizeof(double), cudaMemcpyDeviceToHost));

	for (auto i = 0; i < size; i++)
		std::cout << hostDataZ[i] << " ";
	std::cout << std::endl;

	_gpuFree();
	_cpuFree();

	system("pause");

	return 0;
}