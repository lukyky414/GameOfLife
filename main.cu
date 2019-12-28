#include <GL/glut.h>

#include <stdio.h>

int my_window;

//Affichage de la fenetre.
void renderScene(void){
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
        glVertex3f(-0.5,-0.5,0.0);
        glVertex3f(0.5,0.0,0.0);
        glVertex3f(0.0,0.5,0.0);
    glEnd();

    glutSwapBuffers();
    printf("Frame.\n");
}

void keyboardHandler(unsigned char key, int x, int y){
    if(key==27) exit(0);
}

int main(int argc, char** argv){
    //init glut
        glutInit(&argc, argv);
        //Init windows
        glutInitWindowPosition(10,10);
        glutInitWindowSize(1920,1080);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE /* | GLUT_DEPTH*/ );
        my_window = glutCreateWindow("Ma première fenêtre!");
        glutFullScreen();

    //event callbacks
    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutKeyboardFunc(keyboardHandler);

    //windows process
    glutMainLoop();

    return 0;
}