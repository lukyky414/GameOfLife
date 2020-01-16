#include "affichage.cuh"



extern GLuint gl_pixelBufferObject;
extern GLuint gl_texturePtr;
extern cudaGraphicsResource* cudaPboResource;
extern uchar4* d_textureBufferData;

extern unsigned char *data1, *data2, *host_data;
extern unsigned char *rule, *host_rule;

//Permet de calculer la proportion de l'image à afficher, car les coordonnées ne sont pas en pixel.
float l, r, u, d;
struct static_block{
    static_block(){
        //Affiche le centre horizontal de l'image
        l = (float(TEXTUR_COL)/2.0f - float(SCREEN_COL)/2.0f)/float(TEXTUR_COL);
        r = (float(TEXTUR_COL)/2.0f + float(SCREEN_COL)/2.0f)/float(TEXTUR_COL);
        //Affiche le haut de l'immage
        u = 0.0f;
        d = (float(SCREEN_ROW))/float(TEXTUR_ROW);
        //Affiche le bas de l'image
        //u = (float(TEXTUR_ROW) - float(SCREEN_ROW))/float(TEXTUR_ROW);
        //d = 1.0f;
    
        if(l < 0.0f) l = 0.0f;
        if(r > 1.0f) r = 1.0f;
        if(u < 0.0f) u = 0.0f;
        if(d > 1.0f) d = 1.0f;

        //Inverser le haut et le bas. Je ne sais pas pourquoi la texture est inversée
        float tmp;
        tmp = u;
        u = d;
        d = tmp;
    }
};
static static_block my_static_block;

//Fonction de boucle principale
void renderScene(void){
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
    print_data();

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
        print_data();
        state = 1-state;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTUR_COL, TEXTUR_ROW, GL_RGBA, GL_UNSIGNED_BYTE, 0); //Copier les pixels du PBO vers la texture gl
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0); //Désallouer la texture cuda
    
    //Afficher la texture à l'écran

    glBegin(GL_QUADS); //On dessine une texture dans un quadrilatère (rectangle de l'écran)
    

    //coordonnée texture (pixel)   -   coordonnées écran (pixel)
    glTexCoord2f( l, u);              glVertex2f(0.0f, 0.0f);
    glTexCoord2f( r, u);              glVertex2f(float(SCREEN_COL), 0.0f);
    glTexCoord2f( r, d);              glVertex2f(float(SCREEN_COL), float(SCREEN_ROW));
    glTexCoord2f( l, d);              glVertex2f(0.0f, float(SCREEN_ROW));

    glEnd(); //Fin du quadrilatère
   
    //Libérer les buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    initial_data();
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