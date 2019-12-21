#include <stdio.h>

//les fonctions avec __global__ seront execute sur la CG (device)
__global__ void mykernel(void){

}

//Le reste est compile avec le compilateur de base genre gcc
int main(void) {
    //Appel d'une fonction vers le device, ne pas oublier les <<< et >>> avant les parenthèses.
    mykernel<<<1,1>>>();

    
    printf("Hello World!\n");
    return 0;
}