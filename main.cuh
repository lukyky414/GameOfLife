#pragma once
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include <unistd.h>
#include <stdio.h>

//Taille de l'ecran
#define SCREEN_COL 1920
#define SCREEN_ROW 1080

//Fenetre de calcul 6144 3000
#define TEXTUR_COL 480
#define TEXTUR_ROW 270

//Voisinage de distance 1 (3 cases prises en comptes)
#define VOISINAGE 2

//Voir le retour de device_info.cu
#define NB_THREAD 1024
#define DEVICE 0

#include "calcul.cuh"
#include "affichage.cuh"
#include "initialisation.cuh"


int main(int argc, char** argv);
void keyboardHandler(unsigned char key, int x, int y);