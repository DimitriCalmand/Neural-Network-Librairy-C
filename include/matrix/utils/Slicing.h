#ifndef SLICING_H
#define SLICING_H

#include "../Matrix.h"


Matrix* __Slicing_Matrix(Matrix this, int index_shape, int start, int end);
void Slicing_Matrix(Matrix* this, int index_shape, int start, int end);
void Put_Matrix(Matrix* this, Matrix* m2, int axis, int start, int end);

void __Slincing2d_Matrix(Matrix this, int axis, int start, int end, Matrix *res);
void __Slicing3d_Matrix(Matrix this, int axis, int start, int end, Matrix *res);
void __Slicing4d_Matrix(Matrix this, int axis, int start, int end, Matrix *res);

#endif