#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include <unistd.h>
#include <stdio.h>
#include <time.h>

#define SCREEN_COL 1920
#define SCREEN_ROW 1080

#define TEXTUR_COL 4096
#define TEXTUR_ROW 2048

#define NB_THREAD 1024
#define VOISINAGE 3

#define DEVICE 0 //see the temp.cu for the device number

//Pour les boucles
bool state;
int my_window;
int size;
int cell_per_thread;

//Pointeurs pour la memoire
char *row1, *row2, *host_row;

//Pour l'affichage
GLuint gl_pixelBufferObject;
GLuint gl_texturePtr;
cudaGraphicsResource* cudaPboResource;
uchar4* d_textureBufferData = nullptr;


//Epoque calcule sur le device. Un appel a cette fonction par pixel
__global__ void epoque(char* in, char* out, char*){
    uint x = threadIdx.x * (blockIdx.x*NB_THREAD) / 16;

    __shared__ char ligne[TEXTUR_COL];

    uint first=0;
    uint last=TEXTUR_COL-1;

    //Recopier la ligne pour les calculs.
    ligne[x] = in[x];

    //Attendre que le block ait finis de recopier les lignes
    __syncthreads();


}

//Calcule la texture à afficher
__global__ void affichageCuda(char* map, uchar4* texture){
    uint first_x = threadIdx.x;
    uint x;
    uint y = blockIdx.x;

    //Pour éviter de refaire les multiplication
    uint out_line = y * NB_COLONNE;
    uint pos;

    x = first_x;
    while(x < NB_COLONNE){
        pos = x + out_line;
        if(map[pos] == (char)1){
            texture[pos].x = 255;
            texture[pos].y = 255;
            texture[pos].z = 255;
        }
        else{
            texture[pos].x = 0;
            texture[pos].y = 0;
            texture[pos].z = 0;
        }

        x+=NB_THREAD;
    }
}

//Générer une carte de départ aléatoire
void random_map(char* map, int n){
    int i;

    for(i=0; i<n; i++){
        if(rand()%100 > 50)
            map[i] = 1;
        else
            map[i] = 0;
    }
}

void reset(){
    state = false;
    random_map(host_map,N);
    cudaMemcpy(map1, host_map, size, cudaMemcpyHostToDevice);
}

//Afficher la carte dans la console. Un # représente une case vivante, un < > une case morte.
//Un saut de ligne sépare chaque ligne. Si le lecteur (console) affiche automatiquement un saut de ligne cet affichage est inutile quand l'image est trop grande.
void affichageConsole(char* map){
    int i, j;
    for(j=0; j<NB_LIGNE; j++){
        for(i=0; i<NB_COLONNE; i++)
            printf((map[i + j*NB_COLONNE]?"#":" "));
        printf("\n");
    }
}

clock_t t_1 = clock();
//Mis à jours de la fenêtre.
//Déclenche des époques et le dessin de l'image.
void renderScene(void){
    static clock_t t_1 = clock();
    clock_t t, t_e, t_a;
    int k = 0;
    size_t num_bytes;

    //Temps pour les Epoques
    //t_e = clock();

    //Si FAST_SPEED est défini, on effectue un maximum d'époques entre deux frames, sinon une seul époque par frame
#ifdef FAST_SPEED
    do{
#endif
        state = !state;

        if(state)
            epoque<<<NB_LIGNE,NB_THREAD>>>(map1, map2);
        else
            epoque<<<NB_LIGNE,NB_THREAD>>>(map2, map1);
        
        k++;

#ifdef FAST_SPEED
        t = clock()-t_1;
    }while(t < max_time);
#endif
    //printf("  Epoques en %.5fs (%d)\n", (double)(clock()-t_e)/CLOCKS_PER_SEC, k);
    //Reset du timer ici. On prend en compte l'affichage pour le calcul du temps.
    t_1 = clock();

    //Temps pour l'Affichage
    //t_a = clock();

    //Affichage

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    //Bind la texture
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    //Bind le PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    //Calcul de l'image
    //On réserve le PBO
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource);

    if(state)
        affichageCuda<<<NB_LIGNE,NB_THREAD>>>(map2, d_textureBufferData);
    else
        affichageCuda<<<NB_LIGNE,NB_THREAD>>>(map1, d_textureBufferData);

    //Copier les pixels du PBO
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NB_COLONNE, NB_LIGNE, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
   
    //On dessine la texture à l'écran
    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f);    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);    glVertex2f(float(TAILLE_LARGEUR), 0.0f);
    glTexCoord2f(1.0f, 1.0f);    glVertex2f(float(TAILLE_LARGEUR), float(TAILLE_HAUTEUR));
    glTexCoord2f(0.0f, 1.0f);    glVertex2f(0.0f, float(TAILLE_HAUTEUR));

    glEnd();
   
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    //printf("Affichage en %.5fs\n", (double)(clock()-t_a)/CLOCKS_PER_SEC);
}

void exit_function(){
    printf("Exiting...\n");
    cudaDeviceSynchronize();
    cudaGraphicsUnregisterResource(cudaPboResource);

    cudaFree(map1);
    cudaFree(map2);

    free(host_map);

    exit(0);
}

//Gère les commandes clavier
void keyboardHandler(unsigned char key, int x, int y){
    //Permet de quitter le programme
    if(key==27){
        exit_function();
    }
    if(key=='r'){
        reset();
    }
}

bool initialisation_opengl(int& argc, char** argv){
    //init glut
    glutInit(&argc, argv);
    //Init windows
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(1920,1080);
    my_window = glutCreateWindow("Game Of Life");
    glutFullScreen();

    //event callbacks
    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutKeyboardFunc(keyboardHandler);


    //Préparation de la texture
    
    glewInit();
    //Enable server side capabilities
    glEnable(GL_TEXTURE_2D);

    
    //On génère une texture dans le pointeur
    glGenTextures(1, &gl_texturePtr);
    //Bind le type de texture
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    //Quelques paramètres
        //Permet une texture cyclique
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        
        //Si on zoom sur la texture, on utilise le nearest. (pas de flou, gros pixel)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //Défini la texture. Une GL_TEXTURE_2D, level de base, RGB avec Alpha sur 8bit, taille, pas de bord, pixel format rgba, pixel type, pointeur data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NB_COLONNE, NB_LIGNE, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);


    //Génère les buffers. Il y en as 1.
    glGenBuffers(1, &gl_pixelBufferObject);

    //Permet de bind le buffer et travailler dessus ensuite
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    //Créer et initialise le buffer. On copye h_textureBufferData dans le buffer d'openGL
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, NB_COLONNE * NB_LIGNE * sizeof(uchar4), 0, GL_STREAM_COPY);

    //Créer le Pixel Buffer Object. Cuda va écrire dedans, OpenGL va l'afficher. Rien ne passe par le CPU.
    cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
    if (result != cudaSuccess) return false;

    //On un-bind tous les buffer & textures.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    //Change les coordonnees pour l'affichage
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, NB_COLONNE, 0, NB_LIGNE, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    return true;
}

//Le reste est compile avec le compilateur de base genre gcc
int main(int argc, char** argv) {

    srand (time (NULL));
    //Informations sur la map
    N = NB_COLONNE * NB_LIGNE;
    size = N * sizeof(char);

    //Variables de boucles
    max_time = CLOCKS_PER_SEC / FPS;
    state = false;

    //Alloue la mémoire device
    printf("Allocation Device\n");
    cudaMalloc((void**) &map1, size);
    cudaMalloc((void**) &map2, size);
    

    //Alloue la mémoire host
    printf("Allocation Host\n");
    host_map = (char*) malloc (size);
    
    //Attendre que la copie se termine
    cudaDeviceSynchronize();

    //Désallocation du host memory

    //printf("Initialisation de la fenêtre\n");
    if(!initialisation_opengl(argc, argv))
        exit_function();

    //windows process
    printf("Execution\n");
    reset();
    glutMainLoop();
    
    //Pas de désallocation ici, le programme quitte dans le keyboard Handler.
    return 1;
}