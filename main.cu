#include <stdio.h>
#include <time.h>

#define T_LIGNE 50
#define T_COLON 5

//Epoque calcule sur le device. Un appel a cette fonction par pixel
__global__ void epoque(char* in, char* out){
    int x = threadIdx.x;
    int y = blockIdx.x;

    //Pour éviter de refaire les multiplication
    int t_2ligne = T_LIGNE * 2;
    int out_pos = x + y * T_LIGNE;

    int somme = 0;

    __shared__ char ligne[T_LIGNE * 3];

    //Recopier la ligne du dessus, la ligne actuel, et celle du dessous pour les calculs.
    ligne[x] = ( y>=0 ? in[x + (y-1)*T_LIGNE] : (char)0);
    ligne[x + T_LIGNE] = in[x + y*T_LIGNE];
    ligne[x + t_2ligne] = (y<T_COLON? in[x + (y+1)*T_LIGNE] : (char)0);

    //Attendre que le block ait finis de recopier les lignes
    __syncthreads();

    if(x > 1)
        somme += ligne[x-1] + ligne[x-1 + T_LIGNE] + ligne[x-1 + t_2ligne];
    somme += ligne[x] + ligne[x + t_2ligne];
    if(x < T_LIGNE-1)
        somme += ligne[x+1] + ligne[x+1 + T_LIGNE] + ligne[x+1 + t_2ligne];
    
    //case vivante
    if(ligne[x + T_LIGNE] == 1){
        if(somme == 3 || somme == 2)
            out[out_pos] = (char)1;
        else
            out[out_pos] = (char)0;
    }
    //case morte
    else{
        if(somme == 3)
            out[out_pos] = (char)1;
        else
            out[out_pos] = (char)0;
    }
}

//Générer une carte de départ
void random_ints(char* tab, int n){
    int i;
    srand (time (NULL));

    for(i=0; i<n; i++)
        tab[i] = rand()%2;
}

void affichage(char* map){
    int i, j;
    for(j=0; j<T_COLON; j++){
        for(i=0; i<T_LIGNE; i++)
            printf((map[i + j*T_LIGNE]?"#":" "));
        printf("\n");
    }
}

//Le reste est compile avec le compilateur de base genre gcc
int main(void) {
    int N = T_LIGNE * T_COLON;
    int size = N * sizeof(int);
    int k, state = 0;
    clock_t t;

    //Variables présente sur le processeur (host)
    char *map;

    //Pointeurs pour le device memory
    char *map1, *map2;
    
    //Alloue la mémoire device
    printf("Allocation Device\n");
    cudaMalloc((void**) &map1, size);
    cudaMalloc((void**) &map2, size);

    //Alloue la mémoire host
    printf("Allocation Host\n");
    map = (char*) malloc (size); random_ints(map,N);
    affichage(map);

    //Copie les valeurs dans la device memory
    printf("Copie:\n");
    cudaMemcpy(map1, map, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    printf("Epoques:\n");
    t = clock();
    for(k=0; k<64; k++){
        if(state){
            state = 0;
            epoque<<<T_COLON,T_LIGNE>>>(map2, map1);
        }
        else{
            state = 1;
            epoque<<<T_COLON,T_LIGNE>>>(map1, map2);
        }

        if(state)
            cudaMemcpy(map, map2, size, cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(map, map1, size, cudaMemcpyDeviceToHost);
        
        cudaDeviceSynchronize();
        affichage(map);
        printf("-----\n");
    }
    t = clock()-t;
    printf("%.3f\n", (double)t/CLOCKS_PER_SEC);

    cudaDeviceSynchronize();

    //Récupération des donnes du device vers le host
    if(state)
        cudaMemcpy(map, map2, size, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(map, map1, size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    affichage(map);

    //Désallocation du device memory
    cudaFree(map1); cudaFree(map2);

    //Désallocation du host memory
    free(map);

    return 0;
}