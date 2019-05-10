#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define TICKS 50
#define HEIGHT 8
#define WIDTH 8
#define NUM_GENERATORS 100

typedef struct {
	float* seed;
	bool* image;
	float* fitness;
} GenData;

GenData* alloc_gen_data() {
	GenData* gen_data = (GenData*)malloc(sizeof(GenData));	
	gen_data->seed = (float*)malloc(sizeof(float) * HEIGHT*WIDTH*NUM_GENERATORS);
	gen_data->image = (bool*)malloc(sizeof(bool) * HEIGHT*WIDTH*NUM_GENERATORS);
	gen_data->fitness = (float*)malloc(sizeof(float) * NUM_GENERATORS);
}

float randPercent() {
	return ((rand() % 1000)/(1000.0));
}

void init_generator(GenData* gen_data, int idx) {
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			gen_data->seed[(idx * HEIGHT * WIDTH) + (i * WIDTH) + j] = randPercent();
			gen_data->image[(idx * HEIGHT * WIDTH) + (i * WIDTH) + j] = false;
		}
	}
	gen_data->fitness[idx] = 0;
}

bool* init_letter() {
	bool* gen_compare = (bool*)malloc(sizeof(bool) * WIDTH*HEIGHT);	
	char* hold = (char*)malloc(sizeof(char) * WIDTH*HEIGHT);
	sscanf("0000000000011000001001000100001001111110010000100010010000011000", "%s", hold);
	for(int i = 0; i < WIDTH*HEIGHT; i++) {
		gen_compare[i] = hold[i] == '1';
	}
	return gen_compare;
}

void generation(GenData* gen_data, int idx) {
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			int pos = (idx * HEIGHT * WIDTH) + (i * WIDTH) + j;
			if(randPercent() < gen_data->seed[pos])
				gen_data->image[pos] = true;
			else
				gen_data->image[pos] = false;
		}
	}
}

void fitness(GenData* gen_data, bool* gen_compare, int idx) {
	// printf("fitness start\n");
	gen_data->fitness[idx] = 0.0;
	float count = 0;
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			if(gen_compare[i * WIDTH + j]) {
				count++;
				if(gen_data->image[(idx * HEIGHT * WIDTH) + (i * WIDTH) + j] == gen_compare[i * WIDTH + j]) {
					gen_data->fitness[idx] += .02;
				} else {
					gen_data->fitness[idx] -= .01;
				}
			} else {
				if(gen_compare[i * WIDTH + j]) {
					if(gen_data->image[(idx * HEIGHT * WIDTH) + (i * WIDTH) + j] == gen_compare[i * WIDTH + j]) {
						gen_data->fitness[idx] += .01;
					} else {
						gen_data->fitness[idx] -= .005;
					}
				}
			}
		}
	}
	gen_data->fitness[idx] += (count-18)/64;
	// printf("fitness end, %i, %f\n", idx, gen_data->fitness[idx]);
}

void exchange_float(float* arr, int a, int b) {
	float temp = arr[a];
	arr[a] = arr[b];
	arr[b] = temp;
}

void exchange_bool(bool* arr, int a, int b) {
	bool temp = arr[a];
	arr[a] = arr[b];
	arr[b] = temp;
}

void exchange_generators(GenData* gen_data, int a, int b) {
	exchange_float(gen_data->fitness, a, b);
	int aPrime = a * HEIGHT * WIDTH;
	int bPrime = b * HEIGHT * WIDTH;
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			int ij = i * WIDTH  + j;
			exchange_float(gen_data->seed, aPrime+ij, bPrime+ij);
			exchange_bool(gen_data->image, aPrime+ij, bPrime+ij);
		}
	}
}

int partition(GenData* gen_data, int p, int r) 
{
	int pivot = gen_data->fitness[r];
	int i = p - 1;
	for(int j = p; j < r - 1; j++) {
		if(gen_data->fitness[j] > pivot) {
			i++;
			exchange_generators(gen_data, i, j);
		}
	}
	exchange_generators(gen_data, i + 1, r);
	return i + 1;
}

void sort(GenData* gen_data, int p, int r) {
	if(p < r) {
		int q = partition(gen_data, p, r);
		sort(gen_data, p, q-1);
		sort(gen_data, q+1, r);
	}
}

void mutate(GenData* gen_data, GenData* new_gen_data, int idx) {
	int l = (idx*2 + 0)*HEIGHT*WIDTH;
	int r = (idx*2 + 1)*HEIGHT*WIDTH;
	int o = idx*HEIGHT*WIDTH;
	
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			int ij = i * WIDTH  + j;
			
			// image
			new_gen_data->image[l + ij] = gen_data->image[o + ij];
			new_gen_data->image[r + ij] = gen_data->image[o + ij];
			
			// seed
			if(new_gen_data->image[l + ij]) {
				new_gen_data->seed[l + ij] = gen_data->seed[o + ij] - randPercent()/100;
			} else {
				new_gen_data->seed[l + ij] = gen_data->seed[o + ij] + randPercent()/100;
			}
			
			if(new_gen_data->image[r + ij]) {
				new_gen_data->seed[r + ij] = gen_data->seed[o + ij] - randPercent()/100;
			} else {
				new_gen_data->seed[r + ij] = gen_data->seed[o + ij] + randPercent()/100;
			}
			/*
			new_gen_data->seed[l + ij] = gen_data->seed[o + ij] + (0.5 - randPercent()) / 50;
			new_gen_data->seed[r + ij] = gen_data->seed[o + ij] + (0.5 - randPercent()) / 50;
			*/
		}
	}
}

void next_gen(GenData* gen_data) {
	// printf("next_gen start\n");
	// printf("fist fitness of this tick is: %f\n", gen_data->fitness[0]);
	sort(gen_data, 0, NUM_GENERATORS-1);
	printf("best fitness of this tick is: %f\n", gen_data->fitness[0]);
	GenData* new_gen_data = alloc_gen_data();
	for(int i = 0; i < NUM_GENERATORS/2; i++) {
		mutate(gen_data, new_gen_data, i);
	}
	gen_data = new_gen_data;
	// printf("next_gen end\n");
}

void tick(GenData* gen_data, bool* gen_compare) {
	for(int i = 0; i < NUM_GENERATORS; i++) {
		generation(gen_data, i);
		fitness(gen_data, gen_compare, i);
	}
	next_gen(gen_data);
}

int main(int arc, char **argv) {
	// allocate
	GenData* gen_data = alloc_gen_data();
	
	// init
	srand(time(NULL));
	for(int i = 0; i < NUM_GENERATORS; i++) {
		init_generator(gen_data, i);
	}
	bool* gen_compare = init_letter();
	
	// ticks
	for(int i = 0; i < TICKS; i++) {
		// printf("Tick! %i\n", i);
		tick(gen_data, gen_compare);
	}
	
	// print
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			if(gen_data->image[i * WIDTH  + j]) {
				printf("1.");
			} else {
				printf("0.");
			}
		}
		printf("\n");
	}
	
	// destroy
	free(gen_data);
}
