#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
// #include <pnglite.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>
#include <curand_kernel.h>

#define NUM_THREADS 128
#define TICKS 10
#define HEIGHT 8
#define WIDTH 8
#define NUM_GENERATORS 1000

struct Generator {
	float* seed;  // HEIGHT by WIDTH // [0, 1) representing percent chance of the space being black
	float* rand;  // HEIGHT by WIDTH // [0, 1) representing percent chance used to see if a space is black
	bool* values; // HEIGHT by WIDTH // bool where true is black, false is white
	float fitness;  // int?
};


global bool * gen_compare = 0000000000011000001001000100001001111110010000100010010000011000; //letter to compare

__device__ void generation(Generator* gen_data, int idx)
{
	// for loop, decides whether or not a space is black
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; i < WIDTH; j++) { 
			gen_data[idx].values[i + j * WIDTH] = (gen_data[idx].rand[i + j * WIDTH] < gen_data[idx].seed[i + j * WIDTH]);
		}
	}
}

__device__ void fitness(Generator * gen_data) 
{
	// TODO: compare values to pictures of greek letters
	// is the ratio of white to black correct?
	// are the positions correct?, if the position is wrong, is there still a high chance for it to be correct?
	
	
	for (inmt i = 0; i < HEIGHT; i++){
	
		for (int j = 0; k < WIDTH; j++) {
			
			if (gen_data[idx].values[i + j * WIDTH] == gen_compare[i + j * WIDTH]){
				gen_data.fitness++;
			}
			
			else{
				gen_data.fitness--;
			}
		}
	
	}
	
	
	
}

__global__ void kernel(Generator * gen_data) 
{
	// TODO: run generation to make a new values
	// TODO: run fitness for the new values
	
	
	int idx = blockIdx.x + threadIdx.x; 
	
	generation (gen_data, idx);	
	
	fitness(gen_data);
	
		
}

unsigned long utime(void) 
{
	struct timeval tv;
	unsigned long result = 0;
	
	gettimeofday(&tv, NULL);
	result += (tv.tv_sec * 1000000);
	result += tv.tv_usec;
	
	return result;
}

float randPercent() 
{
	return ((rand() % 1000)/(1000.0));
}

void sort(Generator* gen_data, int p, int q)
{
	// IDK use ... uh... quicksort? 
	// ...
	// BOGOSORT
}

void snap(Generator* gen_data)
{
	// sort generators by fitness
	sort(gen_data, 0, NUM_GENERATORS);
	
	// TODO: kill 50% of them // SNAP
}

void genRandomNumbers(Generator* gen_data, int idx) 
{
	// for loop, gen the numbers
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; i < WIDTH; j++) { 
			gen_data[idx].rand[i + j * WIDTH] = randPercent();
		}
	}
}

void tick(Generator* gen_data, Generator* gen_data_dev)
{
	// generate new random numbers
	for(int i = 0; i < NUM_GENERATORS; i++) {
		genRandomNumbers(gen_data, i);
	}

	// copy gen_data to gen_data_dev
	cudaMemcpy( gen_data, gen_data_dev, sizeof(Generator) * NUM_GENERATORS, cudaMemcpyHostToDevice );
	
	// TODO: call kernal fuction
	
	dim3 grid ((NUM_GENERATIONS + NUM_THREADS - 1) / NUM_THREADS)
	
	kernel<<<grid, NUM_THREADS>>(gen_data_dev);
	
	// copy gen_data_dev to gen_data
	cudaMemcpy( gen_data_dev, gen_data, sizeof(Generator) * NUM_GENERATORS, cudaMemcpyDeviceToHost );
	
	// SNAP
	snap(gen_data);
	
	// TODO: use the remaining 50% of generators to "breed the next generation"
	// make a new gen_data
	// populate new_gen_data
	// exchange new_gen_data <=> gen_data
	// free(new_gen_data)
}

void init_generator(Generator* gen_data, int index)
{
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			gen_data[index].seed[i + j * WIDTH] = randPercent();
			gen_data[index].values[i + j * WIDTH] = false;
		}
	}
	gen_data[index].fitness = 0;
}

int main(int argc, char **argv)
{
	// init generator* gen_data for the initial random seed
	Generator* gen_data = (Generator*)malloc(sizeof(Generator) * NUM_GENERATORS);
	srand(time(NULL));
	for(int i = 0; i < NUM_GENERATORS; i++) {
		init_generator(gen_data, i);
	}
	
	// init gen_data_dev	
	Generator* gen_data_dev;
	cudaMalloc(&gen_data_dev, sizeof(Generator) * NUM_GENERATORS);
	
	for(int i = 0; i < TICKS; i++) {
		// run tick
		tick(gen_data, gen_data_dev);
	}
	
	// TODO: print out the best one so far

	return 0;
}
