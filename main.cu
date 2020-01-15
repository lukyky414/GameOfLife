#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include <unistd.h>
#include <stdio.h>

#define SCREEN_COL 1920
#define SCREEN_ROW 1080

#define TEXTUR_COL 1024
#define TEXTUR_ROW 1024

#define NB_THREAD 1024
//Voisinage 1 -> 3 cases prises en comptes
#define VOISINAGE 1

#define DEVICE 0 //see the temp.cu for the device number

//Pour les boucles
bool state;
int my_window;
bool has_new_regle;
uint regle_number;

//Constantes à calculer
//Nombre de blocks
uint NB_BLOCK;
//Nombre d'octet de donnée
uint NB_OCTET;
//nombre d'octet de donnée par block
uint NB_OCTET_BLOCK;
//nombre de long
uint NB_LONG;
//nombre de long par block
uint NB_LONG_BLOCK;
//Nombre de regle possible avec le voisinage
uint NB_REGLE;
//Nombre de voisinage possible
uint NB_VOISINAGE;

__device__ uint d_NB_BLOCK;
__device__ uint d_NB_OCTET;
__device__ uint d_NB_OCTET_BLOCK;
__device__ uint d_NB_LONG;
__device__ uint d_NB_LONG_BLOCK;
__device__ uint d_NB_REGLE;
__device__ uint d_NB_VOISINAGE;



//Pointeurs pour la memoire
unsigned long *row1, *row2, *host_row;
unsigned char *regle, *host_regle;

//Pour l'affichage
GLuint gl_pixelBufferObject;
GLuint gl_texturePtr;
cudaGraphicsResource* cudaPboResource;
uchar4* d_textureBufferData = nullptr;


//Passe une epoque.
__global__ void epoque(unsigned long* in, unsigned long* out, unsigned char* regle){
    //Le long analysé
    uint l = threadIdx.x / d_NB_LONG_BLOCK + blockIdx.x * d_NB_LONG_BLOCK;
    //le numéro du bit étudié
    uint n = threadIdx.x % d_NB_LONG_BLOCK;
    //un uint avec son ne bit à 1
    uint x = 1 << n;

    //Variables pour les boucles
    int i; //Le ie bit est étudié
    uint fin; //La fin de la boucle
    uint _l, _n, _x; //Les variables du bit recherché (pour le voisinage)

    //L'état d'une cellule est définie par son état et les états de son voisinage
    uint etat;

    //Voisinage gauche
    fin = n;
    for(i = n-VOISINAGE; i < fin; i++){
        if(i < 0){
            if(l == 0)
                _l = d_NB_LONG;
            else
                _l = l-1;
            
            _n = i+64;
        }
        else{
            _l = l;
            _n = i;
        }

        _x = 1 << _n;

        if(in[_l]&&_x)
            etat++;
        etat = etat << 1;
    }

    if(in[l]&&x)
        etat++;
    etat = etat << 1;

    //Voisinage droit
    fin = n+VOISINAGE+1;
    for(i = n+1; i < fin; i++){
        if(i > 63){
            if(l==d_NB_LONG)
                _l = 0;
            else
                _l = l+1;
            
            _n = i - 64;
        }
        else{
            _l = l;
            _n = i;
        }

        _x = 1 << _n;

        if(in[_l]&&_x)
            etat++;
        etat = etat << 1;
    }

    etat = etat >> 1;

    //Choix du out selon l'etat
    if(regle[etat])
        out[l] += (out[l] + x) && x;
    else
        out[l] -= out[l] && (x);
}

//Calcule la texture à afficher
__global__ void affichageCuda(unsigned long* row, uchar4* texture, uint y){
    //Le long analysé
    uint l = threadIdx.x / d_NB_LONG_BLOCK + blockIdx.x * d_NB_LONG_BLOCK;
    //le numéro du bit étudié
    uint n = threadIdx.x % d_NB_LONG_BLOCK;
    //un uint avec son ne bit à 1
    uint x = 1 << n;

    uint o_x = threadIdx.x + blockIdx.x * NB_THREAD;

    uint pos = o_x + y*TEXTUR_COL;

    if(row[l]&&x){
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

//Générer une carte de départ aléatoire
void random_row(unsigned long* row, uint n){
    uint i;

    for(i=0; i<n; i++)
        row[i] = rand();
}

void init_row(unsigned long* row, uint n){
    uint i;

    for(i=0; i<n; i++)
        row[i] = 0;
    
    row[0] += 1;
}

void reset_alea(){
    state = false;
    random_row(host_row,NB_LONG);
    cudaMemcpy(row1, host_row, NB_OCTET, cudaMemcpyHostToDevice);
    has_new_regle = true;
}

void print_row(){
    uint i, j, n;
    
    for(i=0; i<NB_LONG; i++){
        for(j=0; j <64; j++){
            n = 1<<j;
            printf("%c",((host_row[i]&&n)?'#':' '));
        }
    }

    printf("\n");
}

void reset(){
    state=false;
    init_row(host_row,NB_LONG);
    cudaMemcpy(row1, host_row, NB_OCTET, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    has_new_regle = true;
    print_row();
}

void exit_function(){
    printf("Exiting...\n");
    cudaDeviceSynchronize();
    cudaGraphicsUnregisterResource(cudaPboResource);

    cudaFree(row1);
    cudaFree(row2);
    cudaFree(regle);

    free(host_row);
    free(host_regle);

    exit(0);
}

//Mis à jours de la fenêtre.
//Déclenche des époques et le dessin de l'image.
void renderScene(void){
    //if(!has_new_regle)
    //    return;
    reset();
    printf("EPOQUES:\n");
    has_new_regle = false;
    
    size_t num_bytes;
    uint i;

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
    
    affichageCuda<<<NB_BLOCK,NB_THREAD>>>(row1, d_textureBufferData, 0);
    cudaMemcpy(host_row, row1, NB_OCTET, cudaMemcpyDeviceToHost);print_row();

    for(i=1; i < TEXTUR_ROW; i++){
        cudaDeviceSynchronize();
        state = !state;

        if(state){
            epoque<<<NB_BLOCK,NB_THREAD>>>(row1, row2, regle);
            affichageCuda<<<NB_BLOCK,NB_THREAD>>>(row2, d_textureBufferData, i);
            cudaMemcpy(host_row, row2, NB_OCTET, cudaMemcpyDeviceToHost);print_row();
        }
        else{
            epoque<<<NB_BLOCK,NB_THREAD>>>(row2, row1, regle);
            affichageCuda<<<NB_BLOCK,NB_THREAD>>>(row1, d_textureBufferData, i);
            cudaMemcpy(host_row, row1, NB_OCTET, cudaMemcpyDeviceToHost);print_row();
        }
    }

    //Copier les pixels du PBO
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTUR_COL, TEXTUR_ROW, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
   
    //On dessine la texture à l'écran
    glBegin(GL_QUADS);

    //Coordonnée texture proportion  -   coordonnées écran pixel

    glTexCoord2f(0.0f, 0.0f);          glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);          glVertex2f(float(SCREEN_COL), 0.0f);
    glTexCoord2f(1.0f, 1.0f);          glVertex2f(float(SCREEN_COL), float(SCREEN_ROW));
    glTexCoord2f(0.0f, 1.0f);          glVertex2f(0.0f, float(SCREEN_ROW));//*/

    /*
    glTexCoord2f(0.25f, 0.0f);          glVertex2f(0.0f, 0.0f);
    glTexCoord2f(0.75f, 0.0f);          glVertex2f(float(SCREEN_COL), 0.0f);
    glTexCoord2f(0.75f, 0.5f);          glVertex2f(float(SCREEN_COL), float(SCREEN_ROW));
    glTexCoord2f(0.25f, 0.5f);          glVertex2f(0.0f, float(SCREEN_ROW));//*/

    glEnd();
   
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    exit_function();
}

//Gère les commandes clavier
void keyboardHandler(unsigned char key, int x, int y){
    //Permet de quitter le programme
    if(key==27){
        exit_function();
    }
    if(key=='r'){
        reset_alea();
    }
    if(key=='i'){
        reset();
    }
    //TODO right = regle_number++, new_regle
    //lest = regle_number--, new_regle
    //TODO deplacement de la camera avec haut et bas
}

bool initialisation_opengl(int& argc, char** argv){
    //init glut
    glutInit(&argc, argv);
    //Init windows
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(SCREEN_COL,SCREEN_ROW);
    my_window = glutCreateWindow("Automate Cellulaire");
    glutFullScreen();

    //event callbacks
    glutDisplayFunc(renderScene);
    //glutIdleFunc(renderScene);
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEXTUR_COL, TEXTUR_ROW, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);


    //Génère les buffers. Il y en as 1.
    glGenBuffers(1, &gl_pixelBufferObject);

    //Permet de bind le buffer et travailler dessus ensuite
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    //Créer et initialise le buffer. On copye h_textureBufferData dans le buffer d'openGL
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, TEXTUR_COL * TEXTUR_ROW * sizeof(uchar4), 0, GL_STREAM_COPY);

    //Créer le Pixel Buffer Object. Cuda va écrire dedans, OpenGL va l'afficher. Rien ne passe par le CPU.
    cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
    if (result != cudaSuccess) return false;

    //On un-bind tous les buffer & textures.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    //Change les coordonnees pour l'affichage
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, SCREEN_COL, 0, SCREEN_ROW, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    return true;
}

void print_regle(){
    printf("%d\n", regle_number);
    uint i;

    for(i=0; i<NB_VOISINAGE; i++){
        printf("%c", (host_regle[i]?'#':' '));
    }
    printf("\nEPOQUES:\n");
}

void new_regle(){
    int i;
    uint n = regle_number;
    for(i=NB_VOISINAGE-1; i >= 0; i--){
        host_regle[i] = n && 1;
        n = n >> 1;
    }
    reset();
    cudaMemcpy(regle, host_regle, NB_VOISINAGE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    has_new_regle = true;
    print_regle();
}


__global__ void copy_const(uint nb_octet, uint nb_block, uint nb_long, uint nb_octet_block, uint nb_long_block, uint nb_voisinage, uint nb_regle){
    d_NB_OCTET = nb_octet;

    d_NB_BLOCK = nb_block;
    d_NB_LONG = nb_long;

    d_NB_OCTET_BLOCK = nb_octet_block;
    d_NB_LONG_BLOCK = nb_long_block;

    d_NB_VOISINAGE = nb_voisinage;
    d_NB_REGLE = nb_regle;
}

//Le reste est compile avec le compilateur de base genre gcc
int main(int argc, char** argv) {
    NB_OCTET = TEXTUR_COL / 8;

    NB_BLOCK = TEXTUR_COL / NB_THREAD;
    NB_LONG = NB_OCTET / 8;

    NB_OCTET_BLOCK = NB_THREAD / 8;
    NB_LONG_BLOCK = NB_OCTET_BLOCK / 8;

    NB_VOISINAGE = pow(2,VOISINAGE*2 + 1);
    NB_REGLE = pow(2,NB_VOISINAGE);

    copy_const<<<1,1>>>(NB_OCTET, NB_BLOCK, NB_LONG, NB_OCTET_BLOCK, NB_LONG_BLOCK, NB_VOISINAGE, NB_REGLE);


    //Variables de boucles
    state = false;
    has_new_regle = true;

    //Alloue la mémoire device
    printf("Allocation Device\n");
    cudaMalloc((void**) &row1, NB_OCTET);
    cudaMalloc((void**) &row2, NB_OCTET);
    cudaMalloc((void**) &regle, NB_VOISINAGE);
    
    //Alloue la mémoire host
    printf("Allocation Host\n");
    host_row = (unsigned long*) malloc (NB_OCTET);
    host_regle = (unsigned char*) malloc (NB_VOISINAGE);
    
    regle_number = 126;
    new_regle();

    //Attendre que la copie se termine
    cudaDeviceSynchronize();

    //printf("Initialisation de la fenêtre\n");
    if(!initialisation_opengl(argc, argv))
        exit_function();

    //windows process
    printf("Execution\n");
    glutMainLoop();
    
    //Pas de désallocation ici, le programme quitte dans le keyboard Handler.
    return 1;
}