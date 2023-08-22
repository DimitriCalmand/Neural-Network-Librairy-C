#ifndef SAVER_H
#define SAVER_H
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Model.h"

#ifndef DOUBLE
#define H5T_NATIVE_VAR H5T_NATIVE_FLOAT
#else
#define H5T_NATIVE_VAR H5T_NATIVE_DOUBLE
#endif
void Save(Model* this, char* path);
Model* Load_Model(char* path);

#endif