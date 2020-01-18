#include "main.cuh"

//Pointeurs pour la memoire
unsigned char *data1, *data2, *host_data;
unsigned long rule_id;
char* rule;
unsigned char voisinage, portee;
unsigned long texture_width, texture_height;
unsigned long screen_width, screen_height;
bool is_random;

//Pour l'affichage
GLuint gl_pixelBufferObject;
GLuint gl_texturePtr;
cudaGraphicsResource* cudaPboResource;
uchar4* d_textureBufferData;


//Le reste est compile avec le compilateur de base genre gcc
int main(int argc, char** argv) {
    initialisation(argc, argv);


    glutMainLoop();
}


//Permet de désallouer toutes les variables avant de quitter
void exit_function(){
    printf("Exiting...\n");
    cudaDeviceSynchronize();
    cudaGraphicsUnregisterResource(cudaPboResource);

    cudaFree(data1);
    cudaFree(data2);

    free(rule);
    free(host_data);

    exit(0);
}


//Gère les commandes clavier
void keyboardHandler(unsigned char key, int x, int y){
    //printf("%c\n", key);
    //Permet de quitter le programme
    if(key==27){
        exit_function();
    }
    //Random reset
    if(key=='r'){
        is_random = true;
        random_data();
        renderScene();
    }
    //Initial seed reset
    if(key=='i'){
        is_random = false;
        initial_data();
        renderScene();
    }
    if(key=='d' || key=='q'){
        if(key=='d'){
            rule_id++;
            if(rule_id == pow(2,pow(2,voisinage)))
                rule_id = 0;
        }
        else{
            if(rule_id == 0)
                rule_id=pow(2,pow(2,voisinage));
            rule_id--;
        }

        if(is_random)
            random_data();
        else
            initial_data();
        
        sprintf(rule, "%d", rule_id);

        renderScene();

    }
    //TODO deplacement de la camera avec haut et bas
}