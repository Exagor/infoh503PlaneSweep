#include "main.cuh"

#include <cstdio>

//#define DEBUG
#define BLOCKSIZE 16 //8 or 16 or 32
__constant__ float d_param_ref[21]; //21 parameters for the reference camera
__constant__ float d_param_cam[54]; //18 parameters * 3 cameras = 54
__constant__ unsigned int d_width;
__constant__ unsigned int d_height;

//Functions to time the kernel
cudaEvent_t start_cuda_timer()
{
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	return start;
}

void end_cuda_timer(cudaEvent_t start, const char* name)
{
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float millisec;
	cudaEventElapsedTime(&millisec, start, stop);

	//printf("%s:\n", name);
	printf("Time for kernel %s: %f ms\n",name, millisec);
}

float* get_params_cam(const cam camera, const int is_ref, std::vector<cam> const& cam_vector) {
	if (is_ref) {
		// Copy the camera parameters into the array
		float* params = (float*)malloc(sizeof(float) * 21);
		int index = 0;
		// K_inv
		for (int i = 0; i < 9; ++i) {
			params[index++] = (float)camera.p.K_inv[i];
		}
		// R_inv
		for (int i = 0; i < 9; ++i) {
			params[index++] = (float)camera.p.R_inv[i];
		}
		// t_inv
		for (int i = 0; i < 3; ++i) {
			params[index++] = (float)camera.p.t_inv[i];
		}
		return params;
	}
	else {
		float* params = (float*)malloc(sizeof(float) * 18 * 3);
		int index = 0;
		for (auto& cam : cam_vector)
		{
			if (cam.name == camera.name)
				continue;
			// R
			for (int i = 0; i < 9; ++i) {
				params[index++] = (float)cam.p.R[i];
			}
			// t
			for (int i = 0; i < 3; ++i) {
				params[index++] = (float)cam.p.t[i];
			}
			// K
			for (int i = 0; i < 6; ++i) {
				params[index++] = (float)cam.p.K[i];
			}
		}
		return params;
	}
}

__global__ void naive_sweeping_plane_kernel(
	const uint8_t* im_ref,
	const uint8_t* im_cam,
	const double* param_ref,
	const double* param_cam,
	__half* cost_volume, //was float
	const unsigned int width, const unsigned int height,
	int z_planes,
	int window)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int zi = blockIdx.z;

	if (x >= width || y >= height || zi >= z_planes) //handle threads/blocks out of bounds
		return;
	// (1) compute the projection index
	//double x_proj, y_proj; //At the end see if it optimize the code by putting float instead of double
	double z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); //Defined in constants.hpp

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	double X_ref = (param_ref[0] * x + param_ref[1] * y + param_ref[2]) * z; //Was float type
	double Y_ref = (param_ref[3] * x + param_ref[4] * y + param_ref[5]) * z;
	double Z_ref = (param_ref[6] * x + param_ref[7] * y + param_ref[8]) * z;

	// 3D in ref camera coordinates to 3D world
	double X = param_ref[9] * X_ref + param_ref[10] * Y_ref + param_ref[11] * Z_ref - param_ref[18];
	double Y = param_ref[12] * X_ref + param_ref[13] * Y_ref + param_ref[14] * Z_ref - param_ref[19];
	double Z = param_ref[15] * X_ref + param_ref[16] * Y_ref + param_ref[17] * Z_ref - param_ref[20];

	// 3D world to projected camera 3D coordinates
	double X_proj = param_cam[21] * X + param_cam[22] * Y + param_cam[23] * Z - param_cam[30];
	double Y_proj = param_cam[24] * X + param_cam[25] * Y + param_cam[26] * Z - param_cam[31];
	double Z_proj = param_cam[27] * X + param_cam[28] * Y + param_cam[29] * Z - param_cam[32];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	double x_proj = (param_cam[33] * X_proj / Z_proj + param_cam[34] * Y_proj / Z_proj + param_cam[35]);
	double y_proj = (param_cam[36] * X_proj / Z_proj + param_cam[37] * Y_proj / Z_proj + param_cam[38]);
	//float z_proj = Z_proj;

	// Verification it's not out of bounds
	x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

	// (2) Compute the SAD between the windows of ref and cam
	//float cost = compute_cost(im_ref, im_cam, width, height, x_proj, y_proj, x, y, window);
	float cost = 0.0f;
	int half = window / 2;
	float count = 0;

	for (int dy = -half; dy <= half; dy++) {
		for (int dx = -half; dx <= half; dx++) {
			int rx = x + dx;
			int ry = y + dy; //Could be outside this for loop, but it would be less readable
			int px = (int)roundf(x_proj) + dx;
			int py = (int)roundf(y_proj) + dy;

			if (rx >= 0 && ry >= 0 && rx < width && ry < height &&
				px >= 0 && py >= 0 && px < width && py < height) {

				int ref_idx = INDEX_2D(ry, rx, width);//ry * width + rx;
				int cam_idx = INDEX_2D(py, px, width);//py * height + px; why was it height here?
				cost += fabsf((float)(im_ref[ref_idx]) - (float)(im_cam[cam_idx])); //was float
				count += 1.0f; //Was 1.0f
			}
		}
	}
	if (count > 0) {
		cost = cost / count;
	}
	else {
		cost = 255.0f; //If no pixels were counted, return a high cost
	}
	// (3) Store the min cost in the cost volume
	cost_volume[INDEX_3D(zi, y, x, height, width)] = fminf(cost_volume[INDEX_3D(zi, y, x, height, width)],__float2half(cost));
}

__global__ void params_sweeping_plane_kernel(
	const uint8_t* im_ref,
	const uint8_t* im_cam,
	__half* cost_volume,
	const int cam_nb,
	const unsigned int width, const unsigned int height,
	int z_planes,
	int window)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int zi = blockIdx.z;

	if (x >= width || y >= height || zi >= z_planes) //handle threads/blocks out of bounds
		return;
	// (1) compute the projection index
	int cam_index = (cam_nb * 18); //18*3=54
	float z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); //Defined in constants.hpp

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (d_param_ref[0] * x + d_param_ref[1] * y + d_param_ref[2]) * z;
	float Y_ref = (d_param_ref[3] * x + d_param_ref[4] * y + d_param_ref[5]) * z;
	float Z_ref = (d_param_ref[6] * x + d_param_ref[7] * y + d_param_ref[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = d_param_ref[9] * X_ref + d_param_ref[10] * Y_ref + d_param_ref[11] * Z_ref - d_param_ref[18];
	float Y = d_param_ref[12] * X_ref + d_param_ref[13] * Y_ref + d_param_ref[14] * Z_ref - d_param_ref[19];
	float Z = d_param_ref[15] * X_ref + d_param_ref[16] * Y_ref + d_param_ref[17] * Z_ref - d_param_ref[20];

	// 3D world to projected camera 3D coordinates
	float X_proj = d_param_cam[0 + cam_index] * X + d_param_cam[1 + cam_index] * Y + d_param_cam[2 + cam_index] * Z - d_param_cam[9 + cam_index];
	float Y_proj = d_param_cam[3 + cam_index] * X + d_param_cam[4 + cam_index] * Y + d_param_cam[5 + cam_index] * Z - d_param_cam[10 + cam_index];
	float Z_proj = d_param_cam[6 + cam_index] * X + d_param_cam[7 + cam_index] * Y + d_param_cam[8 + cam_index] * Z - d_param_cam[11 + cam_index];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (d_param_cam[12 + cam_index] * X_proj / Z_proj + d_param_cam[13 + cam_index] * Y_proj / Z_proj + d_param_cam[14 + cam_index]);
	float y_proj = (d_param_cam[15 + cam_index] * X_proj / Z_proj + d_param_cam[16 + cam_index] * Y_proj / Z_proj + d_param_cam[17 + cam_index]);
	//float z_proj = Z_proj;

	// Verification it's not out of bounds
	x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

	// (2) Compute the SAD between the windows of ref and cam
	float cost = 0.0f;
	const int half = window / 2;
	float count = 0;

	int px_base = (int)x_proj; //already cast outside the loop
	int py_base = (int)y_proj;
	for (int dy = -half; dy <= half; dy++) {
		for (int dx = -half; dx <= half; dx++) {
			int rx = x + dx;
			int ry = y + dy;
			int px = px_base + dx;
			int py = py_base + dy;

			if (rx < 0 || ry < 0 || rx >= width || ry >= height) continue;
			if (px < 0 || py < 0 || px >= width || py >= height) continue;

			int ref_idx = INDEX_2D(ry, rx, width);
			int cam_idx = INDEX_2D(py, px, width);
			cost += fabsf((float)(im_ref[ref_idx]) - (float)(im_cam[cam_idx]));
			count += 1.0f;
		}
	}
	if (count > 0) {
		cost = cost / count;
	}
	else {
		cost = 255.0f; //If no pixels were counted, return a high cost
	}
	// (3) Store the min cost in the cost volume
	int idx = INDEX_3D(zi, y, x, height, width);
	cost_volume[idx] = fminf(cost_volume[idx], __float2half(cost));
}

__global__ void shared_sweeping_plane_kernel(
	const uint8_t* im_ref,
	const uint8_t* im_cam,
	__half* cost_volume,
	const int cam_nb,
	const int z_planes,
	const int window)
{
	extern __shared__ uint8_t shared_ref[];

	const int pad = window / 2; //For the borders of the block
	const int shared_width = blockDim.x + 2 * pad;
	const int shared_height = blockDim.y + 2 * pad;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int x = blockIdx.x * blockDim.x + tx;
	const int y = blockIdx.y * blockDim.y + ty;
	const int zi = blockIdx.z;

	if (x >= d_width || y >= d_height || zi >= z_planes) return;

	// Load im_ref into shared memory (with borders)
	const int lx = tx + pad;
	const int ly = ty + pad;

	// Fill center
	shared_ref[INDEX_2D(ly, lx, shared_width)] = im_ref[INDEX_2D(y, x, d_width)];

	// Borders - left/right
	if (tx < pad) {
		int left_x = x - pad;
		shared_ref[INDEX_2D(ly, tx, shared_width)] = (left_x >= 0) ? im_ref[INDEX_2D(y, left_x, d_width)] : 0; //fill border with 0

		int right_x = x + blockDim.x;
		shared_ref[INDEX_2D(ly, tx + blockDim.x + pad, shared_width)] = (right_x < d_width) ? im_ref[INDEX_2D(y,right_x, d_width)] : 0;
	}

	// Borders - top/bottom
	if (ty < pad) {
		int top_y = y - pad;
		shared_ref[INDEX_2D(ty, lx, shared_width)] = (top_y >= 0) ? im_ref[INDEX_2D(top_y, x, d_width)] : 0;

		int bottom_y = y + blockDim.y;
		shared_ref[INDEX_2D(ty + blockDim.y + pad, lx, shared_width)] = (bottom_y < d_height) ? im_ref[INDEX_2D(bottom_y, x, d_width)] : 0;
	}

	// Corners
	if (tx < pad && ty < pad) {
		int tl_x = x - pad, tl_y = y - pad;
		int tr_x = x + blockDim.x, tr_y = y - pad;
		int bl_x = x - pad, bl_y = y + blockDim.y;
		int br_x = x + blockDim.x, br_y = y + blockDim.y;

		shared_ref[INDEX_2D(ty, tx, shared_width)] =
			(tl_x >= 0 && tl_y >= 0) ? im_ref[INDEX_2D(tl_y, tl_x, d_width)] : 0;

		shared_ref[INDEX_2D(ty, tx + blockDim.x + pad, shared_width)] =
			(tr_x < d_width && tr_y >= 0) ? im_ref[INDEX_2D(tr_y, tr_x, d_width)] : 0;

		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx, shared_width)] =
			(bl_x >= 0 && bl_y < d_height) ? im_ref[INDEX_2D(bl_y, bl_x, d_width)] : 0;

		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx + blockDim.x + pad, shared_width)] =
			(br_x < d_width && br_y < d_height) ? im_ref[INDEX_2D(br_y, br_x, d_width)] : 0;
	}

	__syncthreads();

	// (1) compute the projection index
	int cam_index = (cam_nb * 18); //18*3=54
	float z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); //Defined in constants.hpp

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (d_param_ref[0] * x + d_param_ref[1] * y + d_param_ref[2]) * z;
	float Y_ref = (d_param_ref[3] * x + d_param_ref[4] * y + d_param_ref[5]) * z;
	float Z_ref = (d_param_ref[6] * x + d_param_ref[7] * y + d_param_ref[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = d_param_ref[9] * X_ref + d_param_ref[10] * Y_ref + d_param_ref[11] * Z_ref - d_param_ref[18];
	float Y = d_param_ref[12] * X_ref + d_param_ref[13] * Y_ref + d_param_ref[14] * Z_ref - d_param_ref[19];
	float Z = d_param_ref[15] * X_ref + d_param_ref[16] * Y_ref + d_param_ref[17] * Z_ref - d_param_ref[20];

	// 3D world to projected camera 3D coordinates
	float X_proj = d_param_cam[0 + cam_index] * X + d_param_cam[1 + cam_index] * Y + d_param_cam[2 + cam_index] * Z - d_param_cam[9 + cam_index];
	float Y_proj = d_param_cam[3 + cam_index] * X + d_param_cam[4 + cam_index] * Y + d_param_cam[5 + cam_index] * Z - d_param_cam[10 + cam_index];
	float Z_proj = d_param_cam[6 + cam_index] * X + d_param_cam[7 + cam_index] * Y + d_param_cam[8 + cam_index] * Z - d_param_cam[11 + cam_index];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (d_param_cam[12 + cam_index] * X_proj / Z_proj + d_param_cam[13 + cam_index] * Y_proj / Z_proj + d_param_cam[14 + cam_index]);
	float y_proj = (d_param_cam[15 + cam_index] * X_proj / Z_proj + d_param_cam[16 + cam_index] * Y_proj / Z_proj + d_param_cam[17 + cam_index]);

	// Verification it's not out of bounds
	x_proj = x_proj < 0 || x_proj >= d_width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= d_height ? 0 : roundf(y_proj);

	// (2) Compute the SAD between the windows of ref and cam
	float cost = 0.0f;
	float count = 0;

	int px_base = (int)x_proj;
	int py_base = (int)y_proj;
	for (int dy = -pad; dy <= pad; dy++) { //pad is half the window size
		for (int dx = -pad; dx <= pad; dx++) {
			int rx = lx + dx;
			int ry = ly + dy;

			int px = px_base + dx;
			int py = py_base + dy;

			//if (rx < 0 || ry < 0 || rx >= width || ry >= height) continue; //Don't need to verify because lx = tx + pad
			if (px < 0 || py < 0 || px >= d_width || py >= d_height) continue;

			int cam_idx = INDEX_2D(py, px, d_width);
			float ref_val = (float)shared_ref[INDEX_2D(ry, rx, shared_width)];
			float cam_val = (float)im_cam[cam_idx];
			cost += fabsf(ref_val - cam_val);
			count += 1.0f;
		}
	}
	cost = (count > 0) ? cost / count : 255.0f;

	// (3) Store the min cost in the cost volume
	int idx = INDEX_3D(zi, y, x, d_height, d_width);
	cost_volume[idx] = fminf(cost_volume[idx], __float2half(cost));
}

__global__ void shared_9blocks_sweeping_plane_kernel(
	const uint8_t* im_ref,
	const uint8_t* im_cam,
	__half* cost_volume,
	const int cam_nb,
	const unsigned int width, const unsigned int height,
	const int z_planes,
	const int window)
{
	extern __shared__ uint8_t shared_ref[];

	const int pad = window / 2;
	const int blockW = blockDim.x;
	const int blockH = blockDim.y;

	const int shared_width = blockW * 3;
	const int shared_height = blockH * 3;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int x = blockIdx.x * blockW + tx;
	const int y = blockIdx.y * blockH + ty;
	const int zi = blockIdx.z;

	if (x >= width || y >= height || zi >= z_planes) return;

	// Index for shared memory
	for (int dy = -blockH; dy <= blockH; dy += blockH) {
		for (int dx = -blockW; dx <= blockW; dx += blockW) {
			int global_x = x + dx;
			int global_y = y + dy;

			int shared_x = tx + dx + blockW;
			int shared_y = ty + dy + blockH;

			uint8_t val = 0;
			if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
				val = im_ref[INDEX_2D(global_y, global_x, width)];
			}

			shared_ref[INDEX_2D(shared_y, shared_x, shared_width)] = val;
		}
	}

	__syncthreads();

	// (1) compute the projection index
	int cam_index = (cam_nb * 18);
	float z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); // Depth calculation

	// 2D ref to 3D in ref camera
	float X_ref = (d_param_ref[0] * x + d_param_ref[1] * y + d_param_ref[2]) * z;
	float Y_ref = (d_param_ref[3] * x + d_param_ref[4] * y + d_param_ref[5]) * z;
	float Z_ref = (d_param_ref[6] * x + d_param_ref[7] * y + d_param_ref[8]) * z;

	// 3D ref to world
	float X = d_param_ref[9] * X_ref + d_param_ref[10] * Y_ref + d_param_ref[11] * Z_ref - d_param_ref[18];
	float Y = d_param_ref[12] * X_ref + d_param_ref[13] * Y_ref + d_param_ref[14] * Z_ref - d_param_ref[19];
	float Z = d_param_ref[15] * X_ref + d_param_ref[16] * Y_ref + d_param_ref[17] * Z_ref - d_param_ref[20];

	// World to camera 3D
	float X_proj = d_param_cam[0 + cam_index] * X + d_param_cam[1 + cam_index] * Y + d_param_cam[2 + cam_index] * Z - d_param_cam[9 + cam_index];
	float Y_proj = d_param_cam[3 + cam_index] * X + d_param_cam[4 + cam_index] * Y + d_param_cam[5 + cam_index] * Z - d_param_cam[10 + cam_index];
	float Z_proj = d_param_cam[6 + cam_index] * X + d_param_cam[7 + cam_index] * Y + d_param_cam[8 + cam_index] * Z - d_param_cam[11 + cam_index];

	// Camera projection to 2D
	float x_proj = (d_param_cam[12 + cam_index] * X_proj / Z_proj + d_param_cam[13 + cam_index] * Y_proj / Z_proj + d_param_cam[14 + cam_index]);
	float y_proj = (d_param_cam[15 + cam_index] * X_proj / Z_proj + d_param_cam[16 + cam_index] * Y_proj / Z_proj + d_param_cam[17 + cam_index]);

	// Clamp projected coordinates
	x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

	// (2) Compute SAD cost
	float cost = 0.0f;
	float count = 0.0f;

	int px_base = (int)x_proj;
	int py_base = (int)y_proj;

	int lx = tx + blockW; // offset into shared memory
	int ly = ty + blockH;

	for (int dy = -pad; dy <= pad; dy++) {
		for (int dx = -pad; dx <= pad; dx++) {
			int ref_rx = lx + dx;
			int ref_ry = ly + dy;

			int px = px_base + dx;
			int py = py_base + dy;

			if (px < 0 || py < 0 || px >= width || py >= height) continue;

			float ref_val = (float)shared_ref[INDEX_2D(ref_ry, ref_rx, shared_width)];
			float cam_val = (float)im_cam[INDEX_2D(py, px, width)];
			cost += fabsf(ref_val - cam_val);
			count += 1.0f;
		}
	}

	cost = (count > 0) ? cost / count : 255.0f;

	// (3) Write min cost
	int idx = INDEX_3D(zi, y, x, height, width);
	cost_volume[idx] = fminf(cost_volume[idx], __float2half(cost));
}

__global__ void shared_all_sweeping_plane_kernel(
	const uint8_t* im_ref,
	const uint8_t* im_cam,
	__half* cost_volume,
	const int cam_nb,
	const int z_planes,
	const int window)
{
	const int pad = window / 2; //For the borders of the block
	const int shared_width = blockDim.x + 2 * pad;
	const int shared_height = blockDim.y + 2 * pad;

	extern __shared__ uint8_t shared_mem[]; //Have to do that because we can't have 2 shared memory arrays
	uint8_t* shared_ref = shared_mem;
	uint8_t* shared_cam = shared_mem + shared_width * shared_height;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int x = blockIdx.x * blockDim.x + tx;
	const int y = blockIdx.y * blockDim.y + ty;
	const int zi = blockIdx.z;

	if (x >= d_width || y >= d_height || zi >= z_planes) return;

	// (1) compute the projection index
	int cam_index = (cam_nb * 18); //18*3=54
	float z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); //Defined in constants.hpp

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (d_param_ref[0] * x + d_param_ref[1] * y + d_param_ref[2]) * z;
	float Y_ref = (d_param_ref[3] * x + d_param_ref[4] * y + d_param_ref[5]) * z;
	float Z_ref = (d_param_ref[6] * x + d_param_ref[7] * y + d_param_ref[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = d_param_ref[9] * X_ref + d_param_ref[10] * Y_ref + d_param_ref[11] * Z_ref - d_param_ref[18];
	float Y = d_param_ref[12] * X_ref + d_param_ref[13] * Y_ref + d_param_ref[14] * Z_ref - d_param_ref[19];
	float Z = d_param_ref[15] * X_ref + d_param_ref[16] * Y_ref + d_param_ref[17] * Z_ref - d_param_ref[20];

	// 3D world to projected camera 3D coordinates
	float X_proj = d_param_cam[0 + cam_index] * X + d_param_cam[1 + cam_index] * Y + d_param_cam[2 + cam_index] * Z - d_param_cam[9 + cam_index];
	float Y_proj = d_param_cam[3 + cam_index] * X + d_param_cam[4 + cam_index] * Y + d_param_cam[5 + cam_index] * Z - d_param_cam[10 + cam_index];
	float Z_proj = d_param_cam[6 + cam_index] * X + d_param_cam[7 + cam_index] * Y + d_param_cam[8 + cam_index] * Z - d_param_cam[11 + cam_index];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (d_param_cam[12 + cam_index] * X_proj / Z_proj + d_param_cam[13 + cam_index] * Y_proj / Z_proj + d_param_cam[14 + cam_index]);
	float y_proj = (d_param_cam[15 + cam_index] * X_proj / Z_proj + d_param_cam[16 + cam_index] * Y_proj / Z_proj + d_param_cam[17 + cam_index]);

	// Verification it's not out of bounds
	x_proj = x_proj < 0 || x_proj >= d_width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= d_height ? 0 : roundf(y_proj);

	int px_base = (int)x_proj;
	int py_base = (int)y_proj;

	// Load im_ref and im_cam into shared memory (with borders)
	const int lx = tx + pad;
	const int ly = ty + pad;

	// Fill center
	shared_ref[INDEX_2D(ly, lx, shared_width)] = im_ref[INDEX_2D(y, x, d_width)];
	shared_cam[INDEX_2D(ly, lx, shared_width)] = im_cam[INDEX_2D(py_base, px_base, d_width)];

	// Borders - left/right
	if (tx < pad) {
		int left_x = x - pad;
		int left_x_cam = px_base - pad;
		shared_ref[INDEX_2D(ly, tx, shared_width)] = (left_x >= 0) ? im_ref[INDEX_2D(y, left_x, d_width)] : 0; //fill border with 0
		shared_cam[INDEX_2D(ly, tx, shared_width)] = (left_x_cam >= 0) ? im_cam[INDEX_2D(py_base, left_x_cam, d_width)] : 0; //fill border with 0

		int right_x = x + blockDim.x;
		int right_x_cam = px_base + blockDim.x;
		shared_ref[INDEX_2D(ly, tx + blockDim.x + pad, shared_width)] = (right_x < d_width) ? im_ref[INDEX_2D(y, right_x, d_width)] : 0;
		shared_cam[INDEX_2D(ly, tx + blockDim.x + pad, shared_width)] = (right_x_cam < d_width) ? im_cam[INDEX_2D(py_base, right_x_cam, d_width)] : 0;
	}

	// Borders - top/bottom
	if (ty < pad) {
		int top_y = y - pad;
		int top_y_cam = py_base - pad;
		shared_ref[INDEX_2D(ty, lx, shared_width)] = (top_y >= 0) ? im_ref[INDEX_2D(top_y, x, d_width)] : 0;
		shared_cam[INDEX_2D(ty, lx, shared_width)] = (top_y_cam >= 0) ? im_cam[INDEX_2D(top_y_cam, px_base, d_width)] : 0;

		int bottom_y = y + blockDim.y;
		int bottom_y_cam = py_base + blockDim.y;
		shared_ref[INDEX_2D(ty + blockDim.y + pad, lx, shared_width)] = (bottom_y < d_height) ? im_ref[INDEX_2D(bottom_y, x, d_width)] : 0;
		shared_cam[INDEX_2D(ty + blockDim.y + pad, lx, shared_width)] = (bottom_y_cam < d_height) ? im_cam[INDEX_2D(bottom_y_cam, px_base, d_width)] : 0;
	}

	// Corners
	if (tx < pad && ty < pad) {
		//for ref
		int tl_x = x - pad, tl_y = y - pad;
		int tr_x = x + blockDim.x, tr_y = y - pad;
		int bl_x = x - pad, bl_y = y + blockDim.y;
		int br_x = x + blockDim.x, br_y = y + blockDim.y;
		//for cam
		int tl_x_cam = px_base - pad, tl_y_cam = py_base - pad;
		int tr_x_cam = px_base + blockDim.x, tr_y_cam = py_base - pad;
		int bl_x_cam = px_base - pad, bl_y_cam = py_base + blockDim.y;
		int br_x_cam = px_base + blockDim.x, br_y_cam = py_base + blockDim.y;
		//for ref
		shared_ref[INDEX_2D(ty, tx, shared_width)] =
			(tl_x >= 0 && tl_y >= 0) ? im_ref[INDEX_2D(tl_y, tl_x, d_width)] : 0;
		shared_ref[INDEX_2D(ty, tx + blockDim.x + pad, shared_width)] =
			(tr_x < d_width && tr_y >= 0) ? im_ref[INDEX_2D(tr_y, tr_x, d_width)] : 0;
		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx, shared_width)] =
			(bl_x >= 0 && bl_y < d_height) ? im_ref[INDEX_2D(bl_y, bl_x, d_width)] : 0;
		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx + blockDim.x + pad, shared_width)] =
			(br_x < d_width && br_y < d_height) ? im_ref[INDEX_2D(br_y, br_x, d_width)] : 0;
		//for cam
		shared_cam[INDEX_2D(ty, tx, shared_width)] =
			(tl_x_cam >= 0 && tl_y_cam >= 0) ? im_cam[INDEX_2D(tl_y_cam, tl_x_cam, d_width)] : 0;
		shared_cam[INDEX_2D(ty, tx + blockDim.x + pad, shared_width)] =
			(tr_x_cam < d_width && tr_y_cam >= 0) ? im_cam[INDEX_2D(tr_y_cam, tr_x_cam, d_width)] : 0;
		shared_cam[INDEX_2D(ty + blockDim.y + pad, tx, shared_width)] =
			(bl_x_cam >= 0 && bl_y_cam < d_height) ? im_cam[INDEX_2D(bl_y_cam, bl_x_cam, d_width)] : 0;
		shared_cam[INDEX_2D(ty + blockDim.y + pad, tx + blockDim.x + pad, shared_width)] =
			(br_x_cam < d_width && br_y_cam < d_height) ? im_cam[INDEX_2D(br_y_cam, br_x_cam, d_width)] : 0;
	}

	__syncthreads();

	// (2) Compute the SAD between the windows of ref and cam
	float cost = 0.0f;
	float count = 0;

	for (int dy = -pad; dy <= pad; dy++) { //pad is half the window size
		for (int dx = -pad; dx <= pad; dx++) {
			int rx = lx + dx;
			int ry = ly + dy;

			//if (rx < 0 || ry < 0 || rx >= width || ry >= height) continue; //Don't need to verify because lx = tx + pad
			//if (px < 0 || py < 0 || px >= d_width || py >= d_height) continue; //don't need to verify because lx = tx + pad

			cost += fabsf(shared_ref[INDEX_2D(ry, rx, shared_width)] - shared_cam[INDEX_2D(ry, rx, shared_width)]);
			count += 1.0f;
		}
	}
	cost = (count > 0) ? cost / count : 255.0f;

	// (3) Store the min cost in the cost volume
	int idx = INDEX_3D(zi, y, x, d_height, d_width);
	cost_volume[idx] = fminf(cost_volume[idx], __float2half(cost));
}


void wrap_plane_sweep(cam const ref, std::vector<cam> const &cam_vector, int z_planes, int window, __half* h_cost_volume)
{
	cudaEvent_t start;
	unsigned int height = ref.height;
	unsigned int width = ref.width;
	const unsigned int img_size = width * height;
	const unsigned int volume_size = img_size * z_planes;

	uint8_t* d_im_ref = 0;
	uint8_t* d_im_cam1 = 0;
	uint8_t* d_im_cam2 = 0;
	uint8_t* d_im_cam3 = 0;
	__half* d_cost_volume = 0; //previously float

	CHK(cudaSetDevice(0));

	CHK(cudaMalloc((void**) &d_im_ref, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam1, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam2, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam3, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_cost_volume, volume_size * sizeof(__half)));//previously float
	CHK(cudaMemcpy(d_im_ref, ref.YUV[0].data, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam1, cam_vector.at(1).YUV[0].data, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam2, cam_vector.at(2).YUV[0].data, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam3, cam_vector.at(3).YUV[0].data, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemset(d_cost_volume, 255.0, volume_size * sizeof(__half)));//previously float

	// Copy the params into the gpu
	float* h_params_ref = get_params_cam(ref, 1, cam_vector);
	float* h_params_cam = get_params_cam(ref, 0, cam_vector);

	//Copy parameters in the constant memory
	CHK(cudaMemcpyToSymbol(d_param_ref, h_params_ref, sizeof(float) * 21));
	CHK(cudaMemcpyToSymbol(d_param_cam, h_params_cam, sizeof(float) * 54));
	CHK(cudaMemcpyToSymbol(d_width, (const void*) &width, sizeof(unsigned int)));
	CHK(cudaMemcpyToSymbol(d_height, (const void*) &height, sizeof(unsigned int)));

	// Define the kernel launch parameters
	dim3 block_size(BLOCKSIZE, BLOCKSIZE); //Number of threads per block (size of blocks) 16*16=256 or 32*32=1024
	dim3 grid_size((width + BLOCKSIZE-1) / BLOCKSIZE, (height + BLOCKSIZE-1) / BLOCKSIZE, z_planes); //Assure that all pixels are covered with (width+15)/16 and (height+15)/16
	printf("Launching kernel with grid size: %d %d %d\n", grid_size.x, grid_size.y, grid_size.z);
	printf("Launching kernel with block size: %d %d\n", block_size.x, block_size.y);

	// Parameters for shared memory
	int pad = window / 2;
	//int shared_mem_size = (BLOCKSIZE + 2 * pad) * (BLOCKSIZE + 2 * pad) * sizeof(uint8_t); //for shared kernel
	//int shared_mem_size = (3 * BLOCKSIZE) * (3 * BLOCKSIZE) * sizeof(uint8_t); //for shared 9 blocks
	int shared_mem_size = (BLOCKSIZE + 2 * pad) * (BLOCKSIZE + 2 * pad) * sizeof(uint8_t)*2; //for shared all

	// launch 1 kernel per camera
	start = start_cuda_timer();
	shared_all_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam1, d_cost_volume, 0, z_planes, window);
	shared_all_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam2, d_cost_volume, 1, z_planes, window);
	shared_all_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam3, d_cost_volume, 2, z_planes, window);
	end_cuda_timer(start, "Shared all GPU");
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
	CHK(cudaMemcpy(h_cost_volume, d_cost_volume, sizeof(__half) * volume_size, cudaMemcpyDeviceToHost));

Error:
	CHK(cudaFree(d_im_ref));
	CHK(cudaFree(d_im_cam1));
	CHK(cudaFree(d_im_cam2));
	CHK(cudaFree(d_im_cam3));
	CHK(cudaFree(d_cost_volume));

	// Needed for profiling
	CHK(cudaDeviceReset());
}