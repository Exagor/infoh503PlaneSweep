#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <cuda_fp16.h> // for half data type
#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)
#define INDEX_2D(y, x, width) ((y)*(width) + (x)) //Macro to access the 2D array that's stored as a 1D in memory (r = row, c = column)
#define INDEX_3D(z, y, x, dimY, dimX) (((z) * (dimY) * (dimX)) + ((y) * (dimX)) + (x))

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();

void wrap_plane_sweep(cam const ref, std::vector<cam> const& cam_vector, int z_planes, int window, __half* h_cost_volume);