#include <GL/glut.h>

//Affichage de la fenetre.
void renderScene(void){
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
        glVertex3f(-0.5,-0.5,0.0);
        glVertex3f(0.5,0.0,0.0);
        glVertex3f(0.0,0.5,0.0);
    glEnd();

    glutSwapBuffers();
}



int main(int argc, char** argv){
    //init glut
        glutInit(&argc, argv);

        //Init windows
        glutInitWindowPosition(10,10);
        glutInitWindowSize(500,400);

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE /* | GLUT_DEPTH*/ );

        glutCreateWindow("Ma première fenêtre!");

    //event callbacks
        glutDisplayFunc(&renderScene);

    //windows process
        glutMainLoop();

    return 0;
}