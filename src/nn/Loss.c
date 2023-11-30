#include "string.h"
#include "matrix/Matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "nn/Loss.h"



VAR Loss_ (Loss* this, Matrix* y_true, Matrix* y_pred)
{
    char* name = this->name;
    if (strcmp(name, "mse") == 0)
    {
        return this->inheritance.mse->Loss(y_true, y_pred);
    }
    else if (strcmp(name, "bce") == 0)
    {
        return this->inheritance.bce->Loss(y_true, y_pred);
    }
    else if (strcmp(name, "cross_entropy") == 0)
    {
        return this->inheritance.cross_entropy->Loss(y_true, y_pred);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name");
    }
}
Matrix* Loss_Prime (Loss* this, Matrix* y_true, Matrix* y_pred){
    char* name = this->name;
    if (strcmp(name, "mse") == 0){
        return this->inheritance.mse->Loss_Prime(y_true, y_pred);
    }
    else if (strcmp(name, "bce") == 0)
    {
        return this->inheritance.bce->Loss_Prime(y_true, y_pred);
    }
    else if (strcmp(name, "cross_entropy") == 0)
    {
        return this->inheritance.cross_entropy->Loss_Prime(y_true, y_pred);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name");
    }
}
void Free_Loss(Loss* this){
    char* name = this->name;
    if (strcmp(name, "mse") == 0)
    {
        this->inheritance.mse->Free(this->inheritance.mse);
    }
    else if (strcmp(name, "bce") == 0)
    {
        this->inheritance.bce->Free(this->inheritance.bce);
    }
    else if (strcmp(name, "cross_entropy") == 0)
    {
        this->inheritance.cross_entropy->Free(this->inheritance.cross_entropy);
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name");
    }
    free(this);
}
Loss* Get_Loss(char* name)
{
    if (strcmp(name, "mse") == 0)
    {
        return Init_Mse();
    }
    else if (strcmp(name, "bce") == 0)
    {
        return Init_Bce();
    }
    else if (strcmp(name, "cross_entropy") == 0)
    {
        return Init_Cross_entropy();
    }
    else 
    {
        errx(EXIT_FAILURE, "Bad Name");
    }

}
Loss* Init_Loss()
{
    Loss* loss = malloc(sizeof(Loss));
    loss->Loss = Loss_;
    loss->Loss_Prime = Loss_Prime;
    loss->Free = Free_Loss;
    return loss;
}
