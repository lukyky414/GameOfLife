#include "affichage.cuh"
#include "enregistrement.cuh"


extern GLuint gl_pixelBufferObject;
extern GLuint gl_texturePtr;
extern cudaGraphicsResource* cudaPboResource;
extern uchar4* d_textureBufferData;

extern unsigned char *data1, *data2, *host_data;
extern unsigned long rule_id;
extern char* rule;
extern unsigned char voisinage, portee;
extern unsigned long texture_width, texture_height;
extern bool is_random;

//Permet de calculer la proportion de l'image à afficher, car les coordonnées ne sont pas en pixel.
float l, r, u, d;
uint nb_block;
struct static_block{
    static_block(){
        //Affiche le centre horizontal de l'image
        l = (float(texture_width)/2.0f - float(SCREEN_WIDTH)/2.0f)/float(texture_width);
        r = (float(texture_width)/2.0f + float(SCREEN_WIDTH)/2.0f)/float(texture_width);
        //Affiche le haut de l'immage
        u = 0.0f;
        d = (float(SCREEN_HEIGHT))/float(texture_height);
        //Affiche le bas de l'image
        //u = (float(texture_height) - float(SCREEN_HEIGHT))/float(texture_height);
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

        nb_block = texture_width / NB_THREAD;
        if(texture_width%NB_THREAD > 0)
            nb_block++;
    }
};

//Fonction de boucle principale
void renderScene(void){
    uint i, len;
    bool state = 1;
    static static_block my_static_block;
    static size_t texture_size = texture_width * texture_height * sizeof(uchar4);
    
    glClear(GL_COLOR_BUFFER_BIT); //Effacer l'écran
    glEnable(GL_TEXTURE_2D); //Active server-side
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr); //Bind de la texture pour l'utiliser
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject); //Bind du PBO pour l'utiliser
    cudaGraphicsMapResources(1, &cudaPboResource, 0); //Bind de la texture cuda pour l'utiliser
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &texture_size, cudaPboResource); //Récupération du pointeur
    
    //Calcul de la 1e ligne de la texture
    texture_cuda<<<nb_block,NB_THREAD>>>(data1, d_textureBufferData, 0, texture_width);
    //cudaMemcpy(host_data, data1, TEXTUR_COL, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    //print_data();

    //Boucle sur le reste des lignes de la texture
    for(i=1; i < texture_height; i++){
        if(state){
            //Effectuer une époque
            data_cuda<<<nb_block,NB_THREAD>>>(data1, data2, rule_id, portee, texture_width); cudaDeviceSynchronize();
            //Calcul de la ligne de la texture
            texture_cuda<<<nb_block,NB_THREAD>>>(data2, d_textureBufferData, i, texture_width); cudaDeviceSynchronize();
            //cudaMemcpy(host_data, data2, TEXTUR_COL, cudaMemcpyDeviceToHost);cudaDeviceSynchronize();
        }
        else{
            data_cuda<<<nb_block,NB_THREAD>>>(data2, data1, rule_id, portee, texture_width); cudaDeviceSynchronize();
            texture_cuda<<<nb_block,NB_THREAD>>>(data1, d_textureBufferData, i, texture_width); cudaDeviceSynchronize();
            //cudaMemcpy(host_data, data1, TEXTUR_COL, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
        }
        //print_data();
        state = 1-state;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_width, texture_height, GL_RGBA, GL_UNSIGNED_BYTE, 0); //Copier les pixels du PBO vers la texture gl
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0); //Désallouer la texture cuda
    
    //Afficher la texture à l'écran

    glBegin(GL_QUADS); //On dessine une texture dans un quadrilatère (rectangle de l'écran)

    //coordonnée texture (pixel)   -   coordonnées écran (pixel)
    glTexCoord2f( l, u);              glVertex2f(-1.0f, -1.0f);
    glTexCoord2f( r, u);              glVertex2f( 1.0f, -1.0f);
    glTexCoord2f( r, d);              glVertex2f( 1.0f,  1.0f);
    glTexCoord2f( l, d);              glVertex2f(-1.0f,  1.0f);

    glEnd(); //Fin du quadrilatère

    enregistrer(gl_texturePtr);
   
    //Libérer les buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    //Dessiner le numéro de la règle
    glColor3f(1.0f, 0.0f, 0.0f);//En rouge
    glRasterPos2f(-0.99f, 0.97f);//En haut à gauche de l'écran
    len = strlen(rule);
    for(i=0; i<len; i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, rule[i]);
    glColor3f(1.0f, 1.0f, 1.0f);

    glutSwapBuffers();
    is_random = !is_random;
    if(is_random){
        random_data();
    }
    else{
        initial_data();
        rule_id++;
        if(rule_id == pow(2,voisinage)){
            if(portee==3)
                exit(0);
            portee++;
            voisinage+=2;
            rule_id=0;
        }
        sprintf(rule, "%d", rule_id);
    }
}

//Affichage d'une ligne dans le terminal
void print_data(){
    uint i;
    
    for(i=0; i<texture_width; i++)
        printf("%c",(host_data[i]?'#':' '));

    printf("\n");
}


//Affichage de la règle dans le terminal
void print_rule(){
    printf("Rule number: %d\n", rule_id);
    uint i, j;
    uint nb_state = pow(2, voisinage);

    for(i=0; i<nb_state; i++){
        printf(":");
        for(j=voisinage; j>0; j--)
            printf("%c", ( (i& (1<<(j-1)) )?'#':' ' ));
    }
    printf(":\n");


    for(i=0; i<nb_state; i++){
        printf(":");
        for(j=0; j<voisinage; j++){
            if(j==portee && ( rule_id& (1<<i) ) )
                printf("#");
            else
                printf(" ");
        }
    }
    
    printf(":\n");
}