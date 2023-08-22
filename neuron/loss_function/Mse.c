    #include "../../matrix/Matrix.h"
    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <err.h>
    #include "../Loss.h"
    #include "Mse.h"
    VAR Loss_Mse (Matrix* y_true, Matrix* y_pred)
    {
        Matrix* tmp_substract = y_pred->__Sub(*y_pred, *y_true);
        Matrix* tmp_square = tmp_substract->__Power(*tmp_substract, 2);
        tmp_square->Sum(tmp_square, 0);
        double size = y_true->size_array;
        double result = tmp_square->array[0] / size;
        tmp_substract->Free(tmp_substract);
        tmp_square->Free(tmp_square);
        return result;
    }
    Matrix* Loss_Prime_Mse(Matrix* y_true, Matrix* y_pred){
        double size = y_true->size_array;
        Matrix* tmp_substract = y_pred->__Sub(*y_pred, *y_true);
        tmp_substract->Mult_Scalar(tmp_substract, 2 / size);
        return tmp_substract;
    }
    void Free_Mse(Mse* this)
    {
        free(this);
    }
    Loss* Init_Mse(){
        Mse* mse = malloc(sizeof(Mse));
        mse->Loss = Loss_Mse;
        mse->Loss_Prime = Loss_Prime_Mse;
        mse->Free = Free_Mse;
        Loss* loss = Init_Loss();
        loss->inheritance.mse = mse;
        loss->name = "mse";
        return loss;
    }