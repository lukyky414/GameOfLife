#include <GL/glut.h>
#include <stdio.h>
#include <time.h>

#define NB_COLONNE 1920
#define NB_LIGNE 1080
#define NB_THREAD 1024
#define FPS 60
#define DEVICE 0 //see the temp.cu for the device number

bool state;
int my_window;
int N;
int size;
double max_time;

//Pointeurs pour le device memory
char *map1, *map2;


//Epoque calcule sur le device. Un appel a cette fonction par pixel
__global__ void epoque(char* in, char* out){
    int first_x = threadIdx.x;
    int x;
    int y = blockIdx.x;

    //Pour éviter de refaire les multiplication
    int t_2ligne = NB_COLONNE * 2;
    int out_line = y * NB_COLONNE;

    char somme;

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
        somme = 0;
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

void affichageConsole(char* map){
    int i, j;
    for(j=0; j<NB_LIGNE; j++){
        for(i=0; i<NB_COLONNE; i++)
            printf((map[i + j*NB_COLONNE]?"#":" "));
        printf("\n");
    }
}

//Affichage de la fenetre.
void renderScene(void){
    clock_t t, t_1;
    int k = 0;

    t_1 = clock();
    do{
        state = !state;

        if(state)
            epoque<<<NB_LIGNE,NB_THREAD>>>(map1, map2);
        else
            epoque<<<NB_LIGNE,NB_THREAD>>>(map2, map1);
        cudaDeviceSynchronize();
        
        t = clock()-t_1;
        k++;
    }while(t < max_time);
    printf("%d époques en %.3fs\n", k, (double)t/CLOCKS_PER_SEC);

    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
        glVertex3f(-0.5,-0.5,0.0);
        glVertex3f(0.5,0.0,0.0);
        glVertex3f(0.0,0.5,0.0);
    glEnd();

    glutSwapBuffers();
}

void keyboardHandler(unsigned char key, int x, int y){
    if(key==27){
        cudaDeviceSynchronize();

        cudaFree(map1);
        cudaFree(map2);

        exit(0);
    }
}

//Le reste est compile avec le compilateur de base genre gcc
int main(int argc, char** argv) {
    //Informations sur la map
    N = NB_COLONNE * NB_LIGNE;
    size = N * sizeof(char);

    //Variables de boucles
    max_time = CLOCKS_PER_SEC / FPS;
    state = false;

    //Variables présente sur le processeur (host)
    char *map;

    //Alloue la mémoire device
    printf("Allocation Device\n");
    cudaMalloc((void**) &map1, size);
    cudaMalloc((void**) &map2, size);

    //Alloue la mémoire host
    printf("Allocation Host\n");
    map = (char*) malloc (size); random_map(map,N);

    //Copie les valeurs dans la device memory
    printf("Copie sur device\n");
    cudaMemcpy(map1, map, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //Désallocation du host memory
    free(map);

    printf("Execution\n");
    
    //init glut
    glutInit(&argc, argv);
    //Init windows
    glutInitWindowPosition(10,10);
    glutInitWindowSize(1920,1080);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    my_window = glutCreateWindow("Game Of Life");
    glutFullScreen();

    //event callbacks
    glutDisplayFunc(renderScene);
    glutKeyboardFunc(keyboardHandler);

    //windows process
    glutMainLoop();

    cudaDeviceSynchronize();
    
    //Désallocation du device memory
    cudaFree(map1); cudaFree(map2);

    return 0;
}