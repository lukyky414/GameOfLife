#include "initialisation.cuh"

extern unsigned char *data1, *data2, *host_data;
extern unsigned char *rule, *host_rule;
extern uint rule_id;

void initialisation(){
    uint size_data = TEXTUR_COL;//*sizeof(char);
    uint size_rule = pow(2,VOISINAGE*2 + 1);//*sizeof(char);

    //Alloue la mémoire device
    cudaMalloc((void**) &data1, size_data);
    cudaMalloc((void**) &data2, size_data);
    cudaMalloc((void**) &rule, size_rule);
    cudaDeviceSynchronize();

    //Alloue la mémoire host
    host_data = (unsigned char*) malloc (size_data);
    host_rule = (unsigned char*) malloc (size_rule);

    rule_id = 126;
    new_rule();

    initial_data();
    cudaMemcpy(data1, host_data, TEXTUR_COL, cudaMemcpyHostToDevice); cudaDeviceSynchronize();

    initialisation_opengl();
}

extern GLuint gl_pixelBufferObject;
extern GLuint gl_texturePtr;
extern cudaGraphicsResource* cudaPboResource;

void initialisation_opengl(){
    int argc;
    char** argv = (char**) malloc(sizeof(char*));
    argv[0] = (char*)malloc(4*sizeof(char));
    argv[0] = "main";

    //initialisation de glut
    glutInit(&argc, argv);
    
    //initialisation de la fenêtre
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE); //Mode RGB
    glutInitWindowSize(SCREEN_COL,SCREEN_ROW); //Taille de la fenêtre
    glutCreateWindow("Automate Cellulaire"); //Création de la fenêtre
    glutFullScreen(); //Plein écran

    //Callback
    glutDisplayFunc(renderScene); //Lors de l'affichage
    //glutIdleFunc(renderScene); //Idle -> quand rien ne se passe
    glutKeyboardFunc(keyboardHandler); // évenements claviers


    //Préparation de la texture
    glewInit();//Initialisation de glew
    glEnable(GL_TEXTURE_2D);//Activer les capacités server-side

    //Création de la texture
    glGenTextures(1, &gl_texturePtr); //initialisation
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr); //Bind une texture pour travailler dessus
        
    //Paramètre: le zoom prend le NEAREST -> pas de flou lors du zoom, on aura des gros pixels
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //Lors du dézoom j'aimerais flouter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //Ne répète pas la texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //Défini la texture. Une GL_TEXTURE_2D, level de base, RGB avec Alpha sur 8bit, taille, pas de bord, pixel format rgba, pixel type, pointeur data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEXTUR_COL, TEXTUR_ROW, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    //Génère les buffers. Il y en as 1.
    glGenBuffers(1, &gl_pixelBufferObject);

    //Permet de bind le buffer et travailler dessus ensuite
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    //Créer et initialise le buffer à 0
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, TEXTUR_COL * TEXTUR_ROW * sizeof(uchar4), 0, GL_STREAM_COPY);

    //Créer le Pixel Buffer Object. Cuda va écrire dedans, OpenGL va l'afficher. Rien ne passe par le CPU.
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

    //On un-bind tous les buffer & textures.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    //Changers les coodronées écrans en pixel
    //glMatrixMode(GL_TEXTURE);
    //glOrtho(-(TEXTUR_COL/2), TEXTUR_COL/2, -(TEXTUR_ROW/2), TEXTUR_ROW, -1, 1);
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, SCREEN_COL, 0, SCREEN_ROW, -1, 1);
    glMatrixMode(GL_MODELVIEW);
}