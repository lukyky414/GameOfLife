#include "calcul.cuh"

//Calcul d'une génération
__global__ void data_cuda(unsigned char* in, unsigned char* out, unsigned char* rule){
    uint x = threadIdx.x + blockIdx.x * NB_THREAD;

    int i;
    //L'état d'une cellule est défini par son voisinage
    uint state = 0;

    for(i=-VOISINAGE; i <= VOISINAGE; i++){
        state = state << 1;
        state += in[(x+i)%TEXTUR_COL];
    }

    out[x] = rule[state];
}

//Calcul de la texture
__global__ void texture_cuda(unsigned char* data, uchar4* texture, uint y){
    uint x = threadIdx.x + blockIdx.x*NB_THREAD;
    uint pos = x + y*TEXTUR_COL;

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
//Générer un ruban aléatoire
void random_data(){
    uint i;

    for(i=0; i<TEXTUR_COL; i++)
        host_data[i] = rand()%2;
}

//Générer un ruban avec une seule cellule active
void initial_data(){
    uint i;

    for(i=0; i<TEXTUR_COL; i++)
        host_data[i] = 0;
    
    host_data[TEXTUR_COL/2] = 1;
}


extern uint rule_id;
extern unsigned char *rule, *host_rule;
//Passer d'un ID à une règle
void new_rule(){
    uint i;
    uint nb_state = pow(2, VOISINAGE*2+1);
    uint s = 1<<nb_state;

    for(i=0; i < nb_state; i++){
        //Etat de sortie 0 ou 1 en fonction du bit de l'id
        s = s >> 1;
        host_rule[i] = ((rule_id & s) > 0?1:0);
    }
    
    cudaMemcpy(rule, host_rule, nb_state, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
}