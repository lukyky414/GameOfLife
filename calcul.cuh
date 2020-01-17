#pragma once
#include "main.cuh"


__global__ void data_cuda(unsigned char* in, unsigned char* out, unsigned long rule);
__global__ void texture_cuda(unsigned char* data, uchar4* texture, uint y);
void random_data();
void initial_data();
void new_rule();