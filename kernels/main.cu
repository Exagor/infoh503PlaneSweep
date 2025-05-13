#include "main.cuh"

#include <cstdio>

//#define DEBUG
#define BLOCKSIZE 16 //16 or 32
__constant__ float d_param_ref[21];
__constant__ float d_param_cam[54]; //18*3=54

// Those functions are an example on how to call cuda functions from the main.cpp
__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

void wrap_test_vectorAdd() {
	printf("Testing GPU with vector addition:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

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
	float z = 0.3f * 1.1f / (0.3f + ((float)zi / z_planes) * (1.1f - 0.3f)); //Defined in constants.hpp

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (param_ref[0] * x + param_ref[1] * y + param_ref[2]) * z; //Was float type
	float Y_ref = (param_ref[3] * x + param_ref[4] * y + param_ref[5]) * z;
	float Z_ref = (param_ref[6] * x + param_ref[7] * y + param_ref[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = param_ref[9] * X_ref + param_ref[10] * Y_ref + param_ref[11] * Z_ref - param_ref[18];
	float Y = param_ref[12] * X_ref + param_ref[13] * Y_ref + param_ref[14] * Z_ref - param_ref[19];
	float Z = param_ref[15] * X_ref + param_ref[16] * Y_ref + param_ref[17] * Z_ref - param_ref[20];

	// 3D world to projected camera 3D coordinates
	float X_proj = param_cam[21] * X + param_cam[22] * Y + param_cam[23] * Z - param_cam[30];
	float Y_proj = param_cam[24] * X + param_cam[25] * Y + param_cam[26] * Z - param_cam[31];
	float Z_proj = param_cam[27] * X + param_cam[28] * Y + param_cam[29] * Z - param_cam[32];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (param_cam[33] * X_proj / Z_proj + param_cam[34] * Y_proj / Z_proj + param_cam[35]);
	float y_proj = (param_cam[36] * X_proj / Z_proj + param_cam[37] * Y_proj / Z_proj + param_cam[38]);
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
	const unsigned int width, const unsigned int height,
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

	if (x >= width || y >= height || zi >= z_planes) return;

	// Load im_ref into shared memory (with borders)
	const int lx = tx + pad;
	const int ly = ty + pad;

	// Fill center
	shared_ref[INDEX_2D(ly, lx, shared_width)] = im_ref[INDEX_2D(y, x, width)];

	// Borders - left/right
	if (tx < pad) {
		int left_x = x - pad;
		shared_ref[INDEX_2D(ly, tx, shared_width)] = (left_x >= 0) ? im_ref[INDEX_2D(y, left_x, width)] : 0; //fill border with 0

		int right_x = x + blockDim.x;
		shared_ref[INDEX_2D(ly, tx + blockDim.x + pad, shared_width)] = (right_x < width) ? im_ref[INDEX_2D(y,right_x, width)] : 0;
	}

	// Borders - top/bottom
	if (ty < pad) {
		int top_y = y - pad;
		shared_ref[INDEX_2D(ty, lx, shared_width)] = (top_y >= 0) ? im_ref[INDEX_2D(top_y, x, width)] : 0;

		int bottom_y = y + blockDim.y;
		shared_ref[INDEX_2D(ty + blockDim.y + pad, lx, shared_width)] = (bottom_y < height) ? im_ref[INDEX_2D(bottom_y, x, width)] : 0;
	}

	// Corners
	if (tx < pad && ty < pad) {
		int tl_x = x - pad, tl_y = y - pad;
		int tr_x = x + blockDim.x, tr_y = y - pad;
		int bl_x = x - pad, bl_y = y + blockDim.y;
		int br_x = x + blockDim.x, br_y = y + blockDim.y;

		shared_ref[INDEX_2D(ty, tx, shared_width)] =
			(tl_x >= 0 && tl_y >= 0) ? im_ref[INDEX_2D(tl_y, tl_x, width)] : 0;

		shared_ref[INDEX_2D(ty, tx + blockDim.x + pad, shared_width)] =
			(tr_x < width && tr_y >= 0) ? im_ref[INDEX_2D(tr_y, tr_x, width)] : 0;

		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx, shared_width)] =
			(bl_x >= 0 && bl_y < height) ? im_ref[INDEX_2D(bl_y, bl_x, width)] : 0;

		shared_ref[INDEX_2D(ty + blockDim.y + pad, tx + blockDim.x + pad, shared_width)] =
			(br_x < width && br_y < height) ? im_ref[INDEX_2D(br_y, br_x, width)] : 0;
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
	x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

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
			if (px < 0 || py < 0 || px >= width || py >= height) continue;

			int cam_idx = INDEX_2D(py, px, width);
			float ref_val = (float)shared_ref[INDEX_2D(ry, rx, shared_width)];
			float cam_val = (float)im_cam[cam_idx];
			cost += fabsf(ref_val - cam_val);
			count += 1.0f;
		}
	}
	cost = (count > 0) ? cost / count : 255.0f;

	// (3) Store the min cost in the cost volume
	int idx = INDEX_3D(zi, y, x, height, width);
	cost_volume[idx] = fminf(cost_volume[idx], __float2half(cost));
}

void wrap_plane_sweep(cam const ref, std::vector<cam> const &cam_vector, int z_planes, int window, __half* h_cost_volume)
{
	cudaEvent_t start;
	const unsigned int height = ref.height;
	const unsigned int width = ref.width;
	const unsigned int img_size = width * height;
	const unsigned int volume_size = img_size * z_planes;

	uint8_t* d_im_ref = 0;
	uint8_t* d_im_cam1 = 0;
	uint8_t* d_im_cam2 = 0;
	uint8_t* d_im_cam3 = 0;
	__half* d_cost_volume = 0; //previously float
	//For the params used in naive
	//double* d_param_ref = 0;
	//double* d_param_cam1 = 0;
	//double* d_param_cam2 = 0;
	//double* d_param_cam3 = 0;

	CHK(cudaSetDevice(0));

	// init variables to contain image
	uint8_t* ref_flattened = new uint8_t[img_size];
	uint8_t* cam1_flattened = new uint8_t[img_size];
	uint8_t* cam2_flattened = new uint8_t[img_size];
	uint8_t* cam3_flattened = new uint8_t[img_size];
	// flatten the matrix
	for (int y = 0; y < ref.height; ++y) {
		for (int x = 0; x < ref.width; ++x) {
			ref_flattened[INDEX_2D(y,x,ref.width)] = ref.YUV[0].at<uint8_t>(y, x);
		}
	}
	for (int y = 0; y < cam_vector.at(1).height; ++y) {
		for (int x = 0; x < cam_vector.at(1).width; ++x) {
			cam1_flattened[INDEX_2D(y, x, cam_vector.at(1).width)] = cam_vector.at(1).YUV[0].at<uint8_t>(y, x);
		}
	}
	for (int y = 0; y < cam_vector.at(2).height; ++y) {
		for (int x = 0; x < cam_vector.at(2).width; ++x) {
			cam2_flattened[INDEX_2D(y, x, cam_vector.at(2).width)] = cam_vector.at(2).YUV[0].at<uint8_t>(y, x);
		}
	}
	for (int y = 0; y < cam_vector.at(3).height; ++y) {
		for (int x = 0; x < cam_vector.at(3).width; ++x) {
			cam3_flattened[INDEX_2D(y, x, cam_vector.at(3).width)] = cam_vector.at(3).YUV[0].at<uint8_t>(y, x);
		}
	}
	
#ifdef DEBUG // print the flattened matrix
	for (int y = 0; y < ref.height/10; ++y) {
		for (int x = 0; x < ref.width/10; ++x) {
			printf("%d ", ref_flattened[INDEX_2D(y, x, ref.width)]);
		}
		printf("\n");
	}
	for (int y = 0; y < cam_vector.at(1).height/10; ++y) {
		for (int x = 0; x < cam_vector.at(1).width/10; ++x) {
			printf("%d ", cam_flattened[INDEX_2D(y, x, cam_vector.at(1).width)]);
		}
		printf("\n");
	}
#endif

	CHK(cudaMalloc((void**) &d_im_ref, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam1, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam2, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_im_cam3, img_size * sizeof(uint8_t)));
	CHK(cudaMalloc((void**) &d_cost_volume, volume_size * sizeof(__half)));//previously float
	CHK(cudaMemcpy(d_im_ref, ref_flattened, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam1, cam1_flattened, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam2, cam2_flattened, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(d_im_cam3, cam3_flattened, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemset(d_cost_volume, 255.0, volume_size * sizeof(__half)));//previously float

	// Copy the params into the gpu
	float* h_params_ref = get_params_cam(ref, 1, cam_vector);
	float* h_params_cam = get_params_cam(ref, 0, cam_vector);

#ifdef DEBUG // print the params
	printf("Print params :\n");
	for (int i = 0; i < 39; ++i) {
		printf("%f ", h_params_ref[i]);
	}
	printf("\n");
	for (int i = 0; i < 39; ++i) {
		printf("%f ", h_params_cam1[i]);
	}
	printf("\n");
#endif
	//Used in naive
	//CHK(cudaMalloc((void**) &d_param_ref, sizeof(double) * 39));
	//CHK(cudaMalloc((void**) &d_param_cam1, sizeof(double) * 39));
	//CHK(cudaMalloc((void**) &d_param_cam2, sizeof(double) * 39));
	//CHK(cudaMalloc((void**) &d_param_cam3, sizeof(double) * 39));
	//CHK(cudaMemcpy(d_param_ref, h_params_ref, sizeof(double) * 39, cudaMemcpyHostToDevice)); 
	//CHK(cudaMemcpy(d_param_cam1, h_params_cam1, sizeof(double) * 39, cudaMemcpyHostToDevice));
	//CHK(cudaMemcpy(d_param_cam2, h_params_cam2, sizeof(double) * 39, cudaMemcpyHostToDevice));
	//CHK(cudaMemcpy(d_param_cam3, h_params_cam3, sizeof(double) * 39, cudaMemcpyHostToDevice));
	//Used in coalesced
	CHK(cudaMemcpyToSymbol(d_param_ref, h_params_ref, sizeof(float) * 21));
	CHK(cudaMemcpyToSymbol(d_param_cam, h_params_cam, sizeof(float) * 54));

	// Define the kernel launch parameters
	dim3 block_size(BLOCKSIZE, BLOCKSIZE); //Number of threads per block (size of blocks) 16*16=256 or 32*32=1024
	dim3 grid_size((width + BLOCKSIZE-1) / BLOCKSIZE, (height + BLOCKSIZE-1) / BLOCKSIZE, z_planes); //Assure that all pixels are covered with (width+15)/16 and (height+15)/16
	printf("Launching kernel with grid size: %d %d %d\n", grid_size.x, grid_size.y, grid_size.z);
	printf("Launching kernel with block size: %d %d\n", block_size.x, block_size.y);
	// Parameters for shared memory
	int pad = window / 2;
	int shared_mem_size = (BLOCKSIZE + 2 * pad) * (BLOCKSIZE + 2 * pad) * sizeof(uint8_t);

	// launch 1 kernel per camera
	start = start_cuda_timer();
	/*naive_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam1, d_param_ref, d_param_cam1, d_cost_volume, width, height, z_planes, window);
	naive_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam2, d_param_ref, d_param_cam2, d_cost_volume, width, height, z_planes, window);
	naive_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam3, d_param_ref, d_param_cam3, d_cost_volume, width, height, z_planes, window);*/
	//end_cuda_timer(start, "Naive GPU");
	/*params_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam1, d_cost_volume,0, width, height, z_planes, window);
	params_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam2, d_cost_volume,1, width, height, z_planes, window);
	params_sweeping_plane_kernel <<<grid_size, block_size>>> (
		d_im_ref, d_im_cam3, d_cost_volume,2, width, height, z_planes, window);
	end_cuda_timer(start, "Params optimized GPU");*/
	shared_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam1, d_cost_volume, 0, width, height, z_planes, window);
	shared_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam2, d_cost_volume, 1, width, height, z_planes, window);
	shared_sweeping_plane_kernel <<<grid_size, block_size, shared_mem_size>>> (
		d_im_ref, d_im_cam3, d_cost_volume, 2, width, height, z_planes, window);
	end_cuda_timer(start, "Shared GPU");
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
	CHK(cudaMemcpy(h_cost_volume, d_cost_volume, sizeof(__half) * volume_size, cudaMemcpyDeviceToHost));

Error:
	CHK(cudaFree(d_im_ref));
	CHK(cudaFree(d_im_cam1));
	CHK(cudaFree(d_im_cam2));
	CHK(cudaFree(d_im_cam3));
	CHK(cudaFree(d_cost_volume));
	//Used in naive
	//CHK(cudaFree(d_param_ref));
	//CHK(cudaFree(d_param_cam));
	//CHK(cudaFree(d_param_cam1));
	//CHK(cudaFree(d_param_cam2));
	//CHK(cudaFree(d_param_cam3));
	delete[] ref_flattened;
	delete[] cam1_flattened;
	delete[] cam2_flattened;
	delete[] cam3_flattened;

	// Needed for profiling
	CHK(cudaDeviceReset());
}