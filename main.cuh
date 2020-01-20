#pragma once
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include <unistd.h>
#include <stdio.h>

//Taille de l'ecran
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

//Voir le retour de device_info.cu
#define NB_THREAD 1024
#define DEVICE 0

#include "calcul.cuh"
#include "affichage.cuh"
#include "initialisation.cuh"
#include "enregistrement.cuh"


int main(int argc, char** argv);
void keyboardHandler(unsigned char key, int x, int y);