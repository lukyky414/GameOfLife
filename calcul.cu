#include "calcul.cuh"


//Calcul d'une génération
__global__ void data_cuda(unsigned char* in, unsigned char* out, unsigned long rule, unsigned char portee, unsigned long texture_width){
    uint x = threadIdx.x + blockIdx.x * NB_THREAD;
    if(x >= texture_width)
        return;

    int i;
    //L'état d'une cellule est défini par son voisinage
    unsigned long state = 0;

    for(i=-portee; i <= portee; i++){
        state = state << 1;
        state += in[(x+texture_width+i)%texture_width];
    }

    out[x] = (rule&(1l<<state))>>state;
}

//Calcul de la texture
__global__ void texture_cuda(unsigned char* data, uchar4* texture, uint y, unsigned long texture_width){
    uint x = threadIdx.x + blockIdx.x*NB_THREAD;
    if(x >= texture_width)
        return;
    uint pos = x + y*texture_width;

    if(data[x]){
        texture[pos].x = 255;
        texture[pos].y = 255;
        texture[pos].z = 255;
    }
    else{
        texture[pos].x = 0;
        texture[pos].y = 0;
        texture[pos].z = 0;
    }
}

extern unsigned char* host_data;
extern unsigned char* data1;
extern unsigned long texture_width;
//Générer un ruban aléatoire
void random_data(){
    uint i;

    for(i=0; i<texture_width; i++)
        host_data[i] = rand()%2;

    cudaMemcpy(data1, host_data, texture_width /* *sizeof(char)*/, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
}

//Générer un ruban avec une seule cellule active
void initial_data(){
    uint i;

    for(i=0; i<texture_width; i++)
        host_data[i] = 0;
    
    host_data[texture_width/2] = 1;

    cudaMemcpy(data1, host_data, texture_width /* *sizeof(char)*/, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
}