

// Vector Reduction
//

// Includes
#include <stdio.h>
#include <cutil_inline.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



// Input Array Variables
float* h_In = NULL;
float* d_In = NULL;

// Output Array
float* h_Out = NULL;
float* d_Out = NULL;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;

//Timer Variables

unsigned int timer_total=0;



// Functions
void Cleanup(void);
void RandomInit(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);



// Device code
__global__ void VecReduce(float* g_idata, float* g_odata, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[]; 

  unsigned int tid = threadIdx.x; 
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x; 

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] = g_idata[globalid]; 
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) { 
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)  {
    // g_odata[blockIdx.x] = sdata[0];
    atomicAdd(&g_odata[blockIdx.x],sdata[0]);
  }

  
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int N = GlobalSize;
    printf("Vector reduction: Input size %d\n", N);
    printf("Vector reduction: Block size %d\n", BlockSize);
	
    size_t in_size = N * sizeof(float);
    float CPU_result = 0.0;// GPU_result = 0.0;

    // Allocate input vectors h_In and h_B in host memory
    h_In = (float*)malloc(in_size);
    if (h_In == 0) 
      Cleanup();

    // Initialize input vectors
    RandomInit(h_In, N);

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  
    size_t out_size = blocksPerGrid * sizeof(float);
  

    // Allocate host output
    h_Out = (float*)malloc(out_size);
    if (h_Out == 0) 
      Cleanup();

    // STUDENT: CPU computation - time this routine for base comparison
    CPU_result = CPUReduce(h_In, N);

    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_In, in_size) );
    cutilSafeCall( cudaMalloc((void**)&d_Out, out_size) );

    //Initialize timers to zero
    cutilCheckError(cutCreateTimer(&timer_total));  
   
    cutilCheckError(cutStartTimer(timer_total));

    // STUDENT: Copy h_In from host memory to device memory
    cudaMemcpy(d_In, h_In, in_size, cudaMemcpyHostToDevice);

    
    // Invoke kernel
    VecReduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_In, d_Out, N);
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
  
    
    // STUDENT: copy results back from GPU to the h_Out
    cudaMemcpy(h_Out, d_Out, out_size, cudaMemcpyDeviceToHost); 
    
    //Stop the second timer 
 
    cutilCheckError(cutStopTimer(timer_total));

    // STUDENT Check results to make sure they are the same
    printf("CPU results : %f\n", CPU_result);
    printf("GPU results : %f\n", h_Out);

    //Print the Timer values
   
  
 
    printf("Overall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(timer_total));
 
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_In)
        cudaFree(d_In);
    if (d_Out)
        cudaFree(d_Out);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);




    cutilCheckError(cutDeleteTimer(timer_total));

    
        
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

float CPUReduce(float* data, int n)
{
  float sum = 0;
    for (int i = 0; i < n; i++)
        sum = sum + data[i];

  return sum;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}
