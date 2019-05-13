#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
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
	float* rand;  // HEIGHT by WIDTH // [0, 1) representing percent chance used to see if a space is black
	bool* values; // HEIGHT by WIDTH // bool where true is black, false is white
	float fitness;  // int?
};


//global bool * gen_compare = 0000000000011000001001000100001001111110010000100010010000011000; //letter to compare

__device__ void generation(Generator* gen_data, int idx)
{
	// for loop, decides whether or not a space is black
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; i < WIDTH; j++) { 
			gen_data[idx].values[i + j * WIDTH] = (gen_data[idx].rand[i + j * WIDTH] < gen_data[idx].seed[i + j * WIDTH]);
		}
	}
}

__device__ void fitness(Generator * gen_data, int idx, bool* gen_comp_dev) 
{
	// TODO: compare values to pictures of greek letters
	// is the ratio of white to black correct?
	// are the positions correct?, if the position is wrong, is there still a high chance for it to be correct?
	
	
	for (int i = 0; i < HEIGHT; i++){
	
		for (int j = 0; j < WIDTH; j++) {
			
			if (gen_data[idx].values[i + j * WIDTH] == gen_comp_dev[i + j * WIDTH]){
				gen_data[idx].fitness++;
			}
			
			else{
				gen_data[idx].fitness--;
			}
		}
	
	}
	
	
	
}

__global__ void kernel(Generator * gen_data, bool* gen_comp_dev) 
{
	// TODO: run generation to make a new values
	// TODO: run fitness for the new values
	
	
	int idx = ((blockIdx.x * NUM_THREADS) + threadIdx.x); 
	
	generation (gen_data, idx);	
	
	fitness(gen_data, idx, gen_comp_dev);
	
		
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

void exchange(Generator* gen_data, int a, int b) 
{
	Generator hold = gen_data[a];
	gen_data[a] = gen_data[b];
	gen_data[b] = hold;
}

int partition(Generator* gen_data, int p, int r) 
{
	int pivot = gen_data[r].fitness;
	int i = p - 1;
	for(int j = p; j < r - 1; j++) {
		if(gen_data[j].fitness > pivot) {
			i++;
			exchange(gen_data, i, j);
		}
	}
	exchange(gen_data, i + 1, r);
	
	return i + 1;
}

void sort(Generator* gen_data, int p, int r)
{
	if(p < r) {
		int q = partition(gen_data, p, r);
		sort(gen_data, p, q-1);
		sort(gen_data, q+1, r);
	}
}

void genRandomNumbers(Generator* gen_data, int idx) 
{
	// for loop, gen the numbers

	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) { 
			gen_data[idx].rand[i + j * WIDTH] = randPercent();
		}
	}
	
}

Generator mutate(Generator * gen_data, int idx)
{

	//Generator child = (Generator) malloc(sizeof(Generator) * NUM_GENERATORS);
	Generator child;
	child.seed = (float*) malloc (sizeof(float) *  (WIDTH * HEIGHT));
	child.values = (bool*) malloc (sizeof(bool) * (WIDTH * HEIGHT));
	for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++){
			child.seed[i + j * WIDTH] = gen_data[idx].seed[i + j * WIDTH] + (0.5 - randPercent()) / 50;
			child.values[i + j * WIDTH] = gen_data[idx].values[i + j * WIDTH]; 
		}
	}
	
	return child;
}

void tick(Generator* gen_data, Generator* gen_data_dev, bool* gen_comp_dev)
{
	// generate new random numbers
	
	
	for(int i = 0; i < NUM_GENERATORS; i++) {
		genRandomNumbers(gen_data, i);
	}


	// copy gen_data to gen_data_dev
	cudaMemcpy( gen_data_dev, gen_data, sizeof(Generator) * NUM_GENERATORS, cudaMemcpyHostToDevice );
	
	// TODO: call kernal fuction
	
	dim3 grid ((NUM_GENERATORS + NUM_THREADS - 1) / NUM_THREADS);

	
	kernel<<<grid, NUM_THREADS>>>(gen_data_dev, gen_comp_dev);
	
	// copy gen_data_dev to gen_data
	cudaMemcpy( gen_data, gen_data_dev, sizeof(Generator) * NUM_GENERATORS, cudaMemcpyDeviceToHost );
	
	// sort the gen_data by fitness
	sort(gen_data, 1, NUM_GENERATORS-1);

	// use the remaining 50% of generators to "breed the next generation"
	// make a new gen_data
	Generator* new_gen_data = (Generator*)malloc(sizeof(Generator) * NUM_GENERATORS);
	
	//printf("FADS\n");
	
	// populate new_gen_data
	for(int i = 0; i < NUM_GENERATORS/2; i++) {
		// mutate gen_data[i] and put the child in new_gen_data[i*2 + 0]
		new_gen_data[i*2 + 0] = mutate(gen_data, i);
		// mutate gen_data[i] and put the child in new_gen_data[i*2 + 1]
		new_gen_data[i*2 + 1] = mutate(gen_data, i);
	}
	

	// exchange new_gen_data <=> gen_data
	Generator* hold = gen_data;
	gen_data = new_gen_data;
	new_gen_data = hold;//gen_data;
	//free(new_gen_data);
	
}



void init_generator(Generator* gen_data, int idx){

	for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++){
			
			//gen_data[idx].seed[i + j * WIDTH] = (float) malloc (sizeof(float) * NUM_GENERATIONS);
			
			gen_data[idx].seed[i + j * WIDTH] = randPercent();
			gen_data[idx].values[i + j * WIDTH] = false;
		}
	}
	
	gen_data[idx].fitness = 0;

}

int main(int argc, char **argv)
{

	bool* gen_compare = (bool*)malloc(sizeof(bool) * WIDTH*HEIGHT);	
	char* hold = (char*)malloc(sizeof(char) * WIDTH*HEIGHT);
	sscanf("0000000000011000001001000100001001111110010000100010010000011000", "%s", hold);
	for(int i = 0; i < WIDTH*HEIGHT; i++) {
		gen_compare[i] = hold[i] == '1';
	}
	bool* gen_comp_dev;


	
	cudaMalloc( &gen_comp_dev, sizeof(bool) * WIDTH*HEIGHT );
	cudaMemcpy( gen_comp_dev, gen_compare, sizeof(bool) * WIDTH*HEIGHT, cudaMemcpyHostToDevice );

	// init generator* gen_data for the initial random seed 
	//printf("%i\n", sizeof(Generator) NUM_GENERATORS);
	Generator* gen_data = (Generator*) malloc(sizeof(Generator) * NUM_GENERATORS);
	
	//gen_data->seed = (float*) malloc (sizeof(float) *  NUM_GENERATORS);
	//gen_data->values = (bool*) malloc (sizeof(bool) * NUM_GENERATORS);
	
	srand(time(NULL));
	
	for(int i = 0; i < NUM_GENERATORS; i++) {
		gen_data[i].seed = (float*) malloc (sizeof(float) *  (WIDTH * HEIGHT));
		gen_data[i].values = (bool*) malloc (sizeof(bool) * (WIDTH * HEIGHT));
		gen_data[i].rand = (float*) malloc (sizeof(float) *  (WIDTH * HEIGHT));
		init_generator(gen_data, i);
	}
	
	
	// init gen_data_dev	
	Generator* gen_data_dev;
	cudaMalloc(&gen_data_dev, sizeof(Generator) * NUM_GENERATORS);
	
	
	cudaMalloc (&gen_data->seed, sizeof(float) * (WIDTH * HEIGHT));
	cudaMalloc (&gen_data->values, sizeof(bool) * (WIDTH * HEIGHT));
	cudaMalloc (&gen_data->rand, sizeof(float) * (WIDTH * HEIGHT));
	cudaMalloc (&gen_data->fitness, sizeof(float) * (WIDTH * HEIGHT));
	
	
	for(int i = 0; i < TICKS; i++) {
		// run tick
		tick(gen_data, gen_data_dev, gen_comp_dev);
	}
	
	// TODO: print out the best one so far
	
	/*for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++){
			printf ("%c", gen_data[0].values[i + j * WIDTH] ? '1' : '0');
		}
	}*/
	// print
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			if(gen_data->values[i * WIDTH  + j]) {
				printf("1.");
			} else {
				printf("0.");
			}
		}
		printf("\n");
}

	return 0;
}
