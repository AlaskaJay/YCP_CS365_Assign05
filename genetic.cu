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

struct Generator {
	float* seed;  // HEIGHT by WIDTH // [0, 1) representing percent chance of the space being black
	bool* values; // HEIGHT by WIDTH // bool where true is black, false is white
	int fitness;  // int?
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

Generator Gen_Create()
{
	// TODO: Create generator	
	
}


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

void tick()
{
	// TODO: copy gen_data to gen_data_dev
	// TODO: call kernal fuction
	// TODO: copy gen_data_dev to gen_data
}

int main(int argc, char **argv)
{
	// TODO: init generator* gen_data for the initial random seed
	// TODO: init gen_data_dev
	
	for(int i = 0; i < TICKS; i++) {
		// TODO: run tick
		// TODO: sort generators by fitness
		// TODO: kill 50% of them // SNAP
		// TODO: use the remaining 50% of generators to "breed the next generation"
	}
	
	// TODO: print out the best one so far

	return 0;
}
