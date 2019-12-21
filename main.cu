#include <stdio.h>
#include <time.h>

//Nombre de block device
#define BLOCK 4096
//Nombre de thread device
#define THREAD 4096

//les fonctions avec __global__ seront execute sur la CG (device)
__global__ void mult(int* a, int* b, int* c){
    //Tableau des index pour deux blocks et trois threads:
    // [t1,t2,t3][t1,t2,t3]
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    c[index] = a[index] * b[index];
}

void random_ints(int* tab, int n){
    int i;
    srand (time (NULL));

    for(i=0; i<n; i++)
        tab[i] = rand();
}

//Le reste est compile avec le compilateur de base genre gcc
int main(void) {
    int N = BLOCK * THREAD;
    int size = N * sizeof(int);
    int i;
    clock_t time1, time2;

    //Variables présente sur le processeur (host)
    int *h_a, *h_b, *h_c;

    //Pointeurs pour le device memory
    int *d_a, *d_b, *d_c;

    
    //Alloue la mémoire device
    printf("Allocation Device\n");
    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    //Alloue la mémoire host
    printf("Allocation Host\n");
    h_a = (int*) malloc (size); random_ints(h_a,N);
    h_b = (int*) malloc (size); random_ints(h_b,N);
    h_c = (int*) malloc (size);

    //Copie les valeurs dans la device memory
    printf("Copie:\n");
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //Calcul de multiplication en parallèle
    printf("Calcul en parallele:");
    time1 = clock();
    mult<<<BLOCK,THREAD>>>(d_a, d_b, d_c);
    time2 = clock();
    printf("%d\n", time2-time1);

    printf("Calcul en local:");
    time1 = clock();
    for(i=0; i<N; i++)
        h_c[i] = h_a[i] * h_b[i];
    time2 = clock();
    printf("%d\n", time2-time1);

    //Récupération des donnes du device vers le host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //Désallocation du device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    //Désallocation du host memory
    free(h_a); free(h_b); free(h_c);
    return 0;
}