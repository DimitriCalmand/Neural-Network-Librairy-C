#include "../../matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Loss.h"
#include "Bce.h"

VAR Loss_Bce (Matrix* y_true, Matrix* y_pred)
{
    VAR res = 0;
    VAR* yt = y_true->array;
    VAR* yp = y_pred->array;
    VAR eps = 1e-7;
    for (int i = 0; i < y_true->size_array; i++)
    {
        res += yt[i] * log(yp[i] + eps) + (1 - yt[i]) * log(1 - yp[i] + eps);
    }
    return -res / y_true->size_array;
}
Matrix* Loss_Prime_Bce(Matrix* y_true, Matrix* y_pred)
{
    // Matrix* res = Init_Matrix(y_true->shape, y_true->size_shape, "null");
    // VAR* yt = y_true->array;
    // VAR* yp = y_pred->array;
    // VAR eps = 1e-7;
    // for (int i = 0; i < y_true->size_array; i++)
    // {
    //     res->array[i] = ((-yt[i] / (yp[i]+eps)) + ((1-yt[i])/(1-yp[i]+eps))) / y_true->size_array;
    // }
    
    // y_pred->Print(y_pred);
    Matrix* res = y_pred->__Sub(*y_pred, *y_true);
    res->Mult_Scalar(res, 1.0/y_true->shape[0]);
    return  res;
}
void Free_Bce(Bce* this){
    free(this);
}
Loss* Init_Bce(){
    Bce* bce = malloc(sizeof(Bce));
    bce->Loss = Loss_Bce;
    bce->Loss_Prime = Loss_Prime_Bce;
    bce->Free = Free_Bce;
    Loss* loss = Init_Loss();
    loss->inheritance.bce = bce;
    loss->name = "bce";
    return loss;
}