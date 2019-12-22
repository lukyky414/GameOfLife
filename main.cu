#include <stdio.h>
#include <time.h>

#define NB_COLONNE 256
#define NB_LIGNE 256
#define NB_THREAD 1024

//Epoque calcule sur le device. Un appel a cette fonction par pixel
__global__ void epoque(char* in, char* out){
    int first_x = threadIdx.x;
    int x;
    int y = blockIdx.x;

    //Pour éviter de refaire les multiplication
    int t_2ligne = NB_COLONNE * 2;
    int out_line = y * NB_COLONNE;

    int somme = 0;

    __shared__ char ligne[NB_COLONNE * 3];

    //Recopier la ligne du dessus, la ligne actuel, et celle du dessous pour les calculs.
    x = first_x;
    while(x < NB_COLONNE){
        ligne[x] = ( y>=0 ? in[x + (y-1)*NB_COLONNE] : (char)0);
        ligne[x + NB_COLONNE] = in[x + y*NB_COLONNE];
        ligne[x + t_2ligne] = (y<NB_LIGNE? in[x + (y+1)*NB_COLONNE] : (char)0);

        x += NB_THREAD;
    }

    //Attendre que le block ait finis de recopier les lignes
    __syncthreads();

    x = first_x;
    while(x < NB_COLONNE){
        if(x > 1)
            somme += ligne[x-1] + ligne[x-1 + NB_COLONNE] + ligne[x-1 + t_2ligne];
        somme += ligne[x] + ligne[x + t_2ligne];
        if(x < NB_COLONNE-1)
            somme += ligne[x+1] + ligne[x+1 + NB_COLONNE] + ligne[x+1 + t_2ligne];

        //case vivante
        if(ligne[x + NB_COLONNE] == 1){
            if(somme == 3 || somme == 2)
                out[x + out_line] = (char)1;
            else
                out[x + out_line] = (char)0;
        }
        //case morte
        else{
            if(somme == 3)
                out[x + out_line] = (char)1;
            else
                out[x + out_line] = (char)0;
        }

        x += NB_THREAD;
    }
}

//Générer une carte de départ
void random_map(char* map, int n){
    int i;
    srand (time (NULL));

    for(i=0; i<n; i++)
        map[i] = rand()%2;
}

void affichage(char* map){
    int i, j;
    for(j=0; j<NB_LIGNE; j++){
        for(i=0; i<NB_COLONNE; i++)
            printf((map[i + j*NB_COLONNE]?"#":" "));
        printf("\n");
    }
}

//Le reste est compile avec le compilateur de base genre gcc
int main(void) {
    //Informations sur la map
    int N = NB_COLONNE * NB_LIGNE;
    int size = N * sizeof(int);

    //Variables de boucles
    int k=0, state = 0;
    clock_t t, t_1;

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
    map = (char*) malloc (size); random_map(map,N);
    //affichage(map);

    //Copie les valeurs dans la device memory
    printf("Copie:\n");
    cudaMemcpy(map1, map, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    printf("Epoques:");
    t_1 = clock();
    t = (clock()-t_1)/CLOCKS_PER_SEC;
    while(t < 1){
        if(state){
            state = 0;
            epoque<<<NB_LIGNE,NB_THREAD>>>(map2, map1);
        }
        else{
            state = 1;
            epoque<<<NB_LIGNE,NB_THREAD>>>(map1, map2);
        }
        t = (clock()-t_1)/CLOCKS_PER_SEC;
        k++;
    }
    printf("%d in %.3fs\n", k, (double)t);

    cudaDeviceSynchronize();

    //Récupération des donnes du device vers le host
    if(state)
        cudaMemcpy(map, map2, size, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(map, map1, size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    //affichage(map);

    //Désallocation du device memory
    cudaFree(map1); cudaFree(map2);

    //Désallocation du host memory
    free(map);

    return 0;
}