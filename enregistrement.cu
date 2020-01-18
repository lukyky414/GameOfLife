#include "enregistrement.cuh"
#include <fstream>

extern unsigned long rule_id;
extern unsigned long texture_width, texture_height;
extern unsigned char voisinage;

using namespace std;

void enregistrer(GLuint texture_ptr){
    ofstream file;
    char* filename = (char*)malloc(40);
    static unsigned long size_data = texture_width * texture_height * sizeof(uchar4);
    uchar4* texture_data = (uchar4*) malloc (size_data);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data);

    //.x .y .z pour r g b
    sprintf(filename, "./output/%s/%dvoisinage/%d_%dx%d.ppm",(random?"random":"one_seed"), voisinage, rule_id,texture_width,texture_height);
    file.open(filename);

    file << "P3 " << texture_width << " " << texture_height << " 255" << endl;

    unsigned long x, y, ligne = 0;
    for(y=0; y<texture_height; y++){
        for(x=0; x<texture_width; x++){
            file << (int)texture_data[x+ligne].x << " ";
            file << (int)texture_data[x+ligne].y << " ";
            file << (int)texture_data[x+ligne].z << "  ";
        }

        file << endl;
        ligne += texture_width;
    }
    file.close();
    free(filename);
}