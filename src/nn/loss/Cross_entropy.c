    #include "matrix/Matrix.h"
    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <err.h>
    #include "nn/Loss.h"
    #include "nn/loss/Cross_entropy.h"
    VAR Loss_Cross_entropy (Matrix* y_true, Matrix* y_pred)
    {
        VAR res = 0;
        VAR* yt = y_true->array;
        VAR* yp = y_pred->array;
        VAR eps = 1e-7;
        for (int i = 0; i < y_true->size_array; i++)
        {
            res += yt[i]*log(yp[i] + eps);
        }
        return -res / y_true->size_array;
    }
    Matrix* Loss_Prime_Cross_entropy(Matrix* y_true, Matrix* y_pred)
    {
        Matrix* res = Init_Matrix(y_true->shape, y_true->size_shape, "null");
        VAR* yt = y_true->array;
        VAR* yp = y_pred->array;
        VAR eps = 1e-7;
        for (int i = 0; i < y_true->size_array; i++)
        {
            res->array[i] = (-yt[i] / (yp[i]+eps)) / y_true->size_array;
        }
        return res;
    }
    void Free_Cross_entropy(Cross_entropy* this)
    {
        free(this);
    }
    Loss* Init_Cross_entropy()
    {
        printf("\n\nCross_entropy still don't work properly\n\n");
        Cross_entropy* cross_entropy = malloc(sizeof(Cross_entropy));
        cross_entropy->Loss = Loss_Cross_entropy;
        cross_entropy->Loss_Prime = Loss_Prime_Cross_entropy;
        cross_entropy->Free = Free_Cross_entropy;
        Loss* loss = Init_Loss();
        loss->inheritance.cross_entropy = cross_entropy;
        loss->name = "cross_entropy";
        return loss;
    }
