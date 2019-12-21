#include <stdio.h>

//les fonctions avec __global__ seront execute sur la CG (device)
__global__ void add(int* a, int* b, int* c){
    *c = *a + *b;
}

//Le reste est compile avec le compilateur de base genre gcc
int main(void) {
    int size = sizeof(int);

    //Variables présente sur le processeur (host)
    int h_a, h_b, h_c;

    //Pointeurs pour la device memory
    int *d_a, *d_b, *d_c;
    
    //Alloue la mémoire device
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    //Valeurs de base
    h_a = 1687;
    h_b = 35148;

    //Copie les valeurs dans la device memory
    cudaMemcpy(d_a, &h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, size, cudaMemcpyHostToDevice);

    //Calcul de somme en parallèle
    add<<<1,1>>>(d_a, d_b, d_c);

    //Récupération des donnes du device vers le host
    cudaMemcpy(&h_c, d_c, size, cudaMemcpyDeviceToHost);

    //Désallocation de la device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    printf("Calcul en parallele:\n%d + %d = %d\n", h_a, h_b, h_c);

    return 0;
}