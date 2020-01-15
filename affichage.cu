#include "affichage.cuh"



extern GLuint gl_pixelBufferObject;
extern GLuint gl_texturePtr;
extern cudaGraphicsResource* cudaPboResource;
extern uchar4* d_textureBufferData;

extern unsigned char *data1, *data2, *host_data;
extern unsigned char *rule, *host_rule;

extern uint need_refresh;

//Fonction de boucle principale
void renderScene(void){
    //Empêche d'afficher s'il n'y a pas besoin
    if(need_refresh == 0)
        return;

    uint i;
    bool state = 1;
    static size_t texture_size = TEXTUR_COL * TEXTUR_ROW * sizeof(uchar4);
    static uint NB_BLOCK = TEXTUR_COL / NB_THREAD;
    
    glClear(GL_COLOR_BUFFER_BIT); //Effacer l'écran
    glEnable(GL_TEXTURE_2D); //Active server-side
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr); //Bind de la texture pour l'utiliser
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject); //Bind du PBO pour l'utiliser
    cudaGraphicsMapResources(1, &cudaPboResource, 0); //Bind de la texture cuda pour l'utiliser
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &texture_size, cudaPboResource); //Récupération du pointeur
    
    //Calcul de la 1e ligne de la texture
    texture_cuda<<<NB_BLOCK,NB_THREAD>>>(data1, d_textureBufferData, 0);
    cudaMemcpy(host_data, data1, TEXTUR_COL, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    if(need_refresh == 1)print_data();

    //Boucle sur le reste des lignes de la texture
    for(i=1; i < TEXTUR_ROW; i++){
        if(state){
            //Effectuer une époque
            data_cuda<<<NB_BLOCK,NB_THREAD>>>(data1, data2, rule); cudaDeviceSynchronize();
            //Calcul de la ligne de la texture
            texture_cuda<<<NB_BLOCK,NB_THREAD>>>(data2, d_textureBufferData, i); cudaDeviceSynchronize();
            cudaMemcpy(host_data, data2, TEXTUR_COL, cudaMemcpyDeviceToHost);cudaDeviceSynchronize();
        }
        else{
            data_cuda<<<NB_BLOCK,NB_THREAD>>>(data2, data1, rule); cudaDeviceSynchronize();
            texture_cuda<<<NB_BLOCK,NB_THREAD>>>(data1, d_textureBufferData, i); cudaDeviceSynchronize();
            cudaMemcpy(host_data, data1, TEXTUR_COL, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
        }
        if(need_refresh == 1)print_data();
        state = 1-state;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTUR_COL, TEXTUR_ROW, GL_RGBA, GL_UNSIGNED_BYTE, 0); //Copier les pixels du PBO vers la texture gl
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0); //Désallouer la texture cuda
    
    //Afficher la texture à l'écran

    glBegin(GL_QUADS); //On dessine une texture dans un quadrilatère (rectangle de l'écran)
    

    //coordonnée texture (pixel)                      -                      coordonnées écran (pixel)
    glTexCoord2f( 0.0f, 1.0f);                                              glVertex2f(0.0f, -20.0f);
    glTexCoord2f( 1.0f, 1.0f);                                              glVertex2f(float(SCREEN_COL), -20.0f);
    glTexCoord2f( 1.0f, 0.0f);                                              glVertex2f(float(SCREEN_COL), float(SCREEN_ROW));
    glTexCoord2f( 0.0f, 0.0f);                                              glVertex2f(0.0f, float(SCREEN_ROW));

    glEnd(); //Fin du quadrilatère
   
    //Libérer les buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    if(need_refresh == 2){
        initial_data();
        cudaMemcpy(data1, host_data, TEXTUR_COL, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
    }
    need_refresh--;
}

//Affichage d'une ligne dans le terminal
void print_data(){
    uint i;
    
    for(i=0; i<TEXTUR_COL; i++)
        printf("%c",(host_data[i]?'#':' '));

    printf("\n");
}


extern uint rule_id;
//Affichage de la règle dans le terminal
void print_rule(){
    printf("Rule number: %d\n", rule_id);
    uint i;
    uint nb_state = pow(2, VOISINAGE*2+1);

    printf(":   :  #: # : ##:#  :# #:## :###:\n");

    for(i=0; i<nb_state; i++)
        printf("%s", (host_rule[i]==1?": # ":":   "));
    
    printf(":\n\n");
}