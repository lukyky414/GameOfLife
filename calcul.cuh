#pragma once
#include "main.cuh"


__global__ void data_cuda(unsigned char* in, unsigned char* out, unsigned long rule, unsigned char portee, unsigned long texture_width);
__global__ void texture_cuda(unsigned char* data, uchar4* texture, uint y, unsigned long texture_width);
void random_data();
void initial_data();
void new_rule();