#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H

#include "../Matrix.h"

Matrix* __Mult_Scalar_Matrix(Matrix this, VAR scalar);
void Mult_Scalar_Matrix(Matrix* this, VAR scalar);

Matrix* __Mult_Matrix(Matrix m1, Matrix m2);
void Mult_Matrix(Matrix* m1, Matrix* m2);

Matrix* __Dot_Matrix(Matrix m1, Matrix m2);
void Dot_Matrix(Matrix* m1, Matrix* m2);

void __Mult_Mat_Place(Matrix m_1, Matrix m_2, Matrix* res);
void __Dot_Matrix_2D_Place(Matrix m1, Matrix m2, Matrix* res);


#endif