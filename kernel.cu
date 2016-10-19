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

static double *hostData = nullptr;
static double *devData = nullptr, *devBuffer = nullptr;

static void _cpuFree() {
	if (::hostData)
		std::free((void *)::hostData);
}

#define cudaCheck
static void _gpuFree() {
	if (::devData)
		cudaCheck(cudaFree((void *)::devData));

	if (::devBuffer)
		cudaCheck(cudaFree((void *)::devBuffer));
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

__global__ void kernel(double * __restrict__ data, double * __restrict__ buffer,
					   const std::size_t size,
					   const double phaseVelocity, const double outerForse,
					   const double stepX, const double stepY,
					   const double stepT, const double maxTime) {
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (auto t = 0.0; t < maxTime; t += stepT) {
		if (idx < size) {
			buffer[idx] = (stepT * stepT) * (phaseVelocity * phaseVelocity * ((data[idx + 1] + data[idx - 1] - 2.0 * data[idx]) / (stepX * stepX) + 
				                                                              (data[idx + 1] + data[idx - 1] - 2.0 * data[idx]) / (stepY * stepY) + outerForse)) + data[idx];

			__syncthreads();

			data[idx] = buffer[idx];

			__syncthreads();
		}
	}
}

/*
 * Init
 */

int cpuInit(std::size_t size) {
	::hostData = (double *)std::calloc(size, sizeof(double));
	if (!::hostData)
		return 1;

	std::memset(::hostData, 0, size);

	return 0;
}

void gpuInit(std::size_t size) {
	auto byteSize = size * sizeof(double);

	cudaCheck(cudaMalloc((void **)&::devData, byteSize));
	cudaCheck(cudaMalloc((void **)&::devBuffer, byteSize));

	cudaCheck(cudaMemset(::devData, 0, byteSize));
	cudaCheck(cudaMemset(::devBuffer, 0, byteSize));
}

/*
 * Helpers
 */

int printResultToGnuplotFile(const char *filename, const double *result, std::size_t size, double stepX) {
	std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

	if (!ofs.is_open())
		return 1;

	ofs << "plot '-'" << std::endl;

	auto x = 0.0;
	for (auto i = 0; i < size; i++) {
		ofs << x << "\t" << result[i] << std::endl;
		x += stepX;
	}

	ofs << "e" << std::endl;

	ofs.close();

	return 0;
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

	kernel <<<nBlocks, nThreads>>> (devData, devBuffer, size,
									PHASE_VEL, OUTER_FORSE, STEP_X, STEP_Y, STEP_T, maxTime);

	cudaCheck(cudaMemcpy(hostData, devData, size * sizeof(double), cudaMemcpyDeviceToHost));

	for (auto i = 0; i < size; i++)
		std::cout << hostData[i] << " ";
	std::cout << std::endl;

	if (printResultToGnuplotFile("result.txt", hostData, size, STEP_X)) {
		std::cout << "Unable to print to file" << std::endl;
		_cpuFree();
		_gpuFree();
		return 1;
	}

	_gpuFree();
	_cpuFree();

	system("pause");

	return 0;
}