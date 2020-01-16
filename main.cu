#include "main.cuh"

//Pointeurs pour la memoire
unsigned char *data1, *data2, *host_data;
unsigned char *rule, *host_rule;
uint rule_id;

//Pour l'affichage
GLuint gl_pixelBufferObject;
GLuint gl_texturePtr;
cudaGraphicsResource* cudaPboResource;
uchar4* d_textureBufferData;


//Le reste est compile avec le compilateur de base genre gcc
int main(int argc, char** argv) {
    initialisation();

    glutMainLoop();
}


//Permet de désallouer toutes les variables avant de quitter
void exit_function(){
    printf("Exiting...\n");
    cudaDeviceSynchronize();
    cudaGraphicsUnregisterResource(cudaPboResource);

    cudaFree(data1);
    cudaFree(data2);
    cudaFree(rule);

    free(host_data);
    free(host_rule);

    exit(0);
}


//Gère les commandes clavier
void keyboardHandler(unsigned char key, int x, int y){
    //Permet de quitter le programme
    if(key==27){
        exit_function();
    }
    //Random reset
    if(key=='r'){
        random_data();
        cudaMemcpy(data1, host_data, TEXTUR_COL, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
        renderScene();
    }
    //Initial seed reset
    if(key=='i'){
        initial_data();
        cudaMemcpy(data1, host_data, TEXTUR_COL, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
        renderScene();
    }

    if(key=='n'){
        rule_id++;
        initial_data();
        new_rule();
        renderScene();
    }
    if(key=='b'){
        rule_id--;
        initial_data();
        new_rule();
        renderScene();
    }
    //TODO right = regle_number++, new_regle
    //lest = regle_number--, new_regle
    //TODO deplacement de la camera avec haut et bas
}