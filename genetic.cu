#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
// #include <pnglite.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>

#define NUM_THREADS 128
#define TICKS 10
#define HEIGHT 8
#define WIDTH 8
#define NUM_GENERATORS 1000

struct Generator {
	float* seed;  // HEIGHT by WIDTH // [0, 1) representing percent chance of the space being black
	bool* values; // HEIGHT by WIDTH // bool where true is black, false is white
	int fitness;  // int?
};

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
	return ((rand() % 101)/(100.0));
}

/*
Generator Gen_Create()
{
	// TODO: Create generator	
	
}
*/

__device__ void generation()
{
	// TODO: generator new letter
	// for loop, decides whether or not a space is black
}

__device__ void fitness() 
{
	// TODO: compare values to pictures of greek letters
	// is the ratio of white to black correct?
	// are the positions correct?, if the position is wrong, is there still a high chance for it to be correct?
}

__global__ void kernel() 
{
	// TODO: run generation to make a new values
	// TODO: run fitness for the new values
}

void sort(Generator* gen_data)
{
	// IDK use ... uh... quicksort? 
	// ...
	// BOGOSORT
}

void snap(Generator* gen_data)
{
	// sort generators by fitness
	sort(gen_data);
	
	// TODO: kill 50% of them // SNAP
}

void tick(Generator* gen_data, Generator* gen_data_dev)
{
	// copy gen_data to gen_data_dev
	cudaMemcpy( gen_data, gen_data_dev, sizeof(Generator) * NUM_GENERATORS, cudaMemcpyHostToDevice );
	
	// TODO: call kernal fuction
	
	
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
