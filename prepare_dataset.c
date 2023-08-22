#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "matrix/Matrix.h"

Matrix* open_data(char* path, int shape[]) 
{
    FILE *fp;
    fp = fopen(path, "r");
    if (fp == NULL) 
    {
        errx(1, "Erreur lors de l'ouverture du fichier\n");
        return 0;
    }
    else 
    {
        int* data = malloc(sizeof(int)*shape[0]*shape[1]);
        int i, j;
        Matrix* res = Init_Matrix(shape, 2, "null");
        for (i = 0; i < shape[0]; i++) 
        {
            for (j = 0; j < shape[1]; j++) 
            {
                if (fscanf(fp, "%d", data+i*shape[1]+j) != 1) 
                {
                    printf("Erreur lors de la lecture des données\n");
                    exit(1);
                }
                res->array[i*shape[1]+j] = (double) data[i*shape[1]+j];                
            }
        }
        free(data);
        fclose(fp);
        return res;
    }    
}
