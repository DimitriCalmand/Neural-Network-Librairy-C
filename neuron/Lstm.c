#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Lstm.h"
#include <string.h>
#include <time.h>
#include <sys/time.h>

void Init_Saver(Lstm* this)
{
    int shape_res[3] = {this->input->shape[0], this->input->shape[1], this->hidden_units};
    this->h_out = Init_Matrix(shape_res, 3, "null");
    this->c_out = Init_Matrix(shape_res, 3, "null");

    int shape_list[3] = {this->input->shape[1], this->input->shape[0], this->hidden_units};

    this->ft_list = Init_Matrix(shape_list, 3, "null");
    this->it_list = Init_Matrix(shape_list, 3, "null");
    this->ct_list = Init_Matrix(shape_list, 3, "null");
    this->ot_list = Init_Matrix(shape_list, 3, "null");

    shape_list[2] += this->embedding;
    this->concat_list = Init_Matrix(shape_list, 3, "null");
}
void Set_Optimizer_lstm(Lstm* this, Optimizer* optimizer)
{
    this->Optimizer_b_c = optimizer->Create_Optimizer(optimizer, this->b_f);
    this->Optimizer_b_i = optimizer->Create_Optimizer(optimizer, this->b_i);
    this->Optimizer_b_o = optimizer->Create_Optimizer(optimizer, this->b_o);
    this->Optimizer_b_f = optimizer->Create_Optimizer(optimizer, this->b_f);
    this->Optimizer_w_c = optimizer->Create_Optimizer(optimizer, this->w_c);
    this->Optimizer_w_i = optimizer->Create_Optimizer(optimizer, this->w_i);
    this->Optimizer_w_o = optimizer->Create_Optimizer(optimizer, this->w_o);
    this->Optimizer_w_f = optimizer->Create_Optimizer(optimizer, this->w_f);  
}
void Initialize_Lstm(Lstm* this, Matrix* input_matrix, Optimizer* optimizer)
{
    this->embedding = input_matrix->shape[input_matrix->size_shape-1];
    this->input->Copy(this->input, input_matrix);

    int shape_w[2] = {this->hidden_units + this->embedding, this->hidden_units};

    char* name = "random";
    Matrix* tmp_w_f = Init_Matrix(shape_w,  2, name);
    this->w_f->Copy(this->w_f, tmp_w_f);
    Matrix* tmp_w_i = Init_Matrix(shape_w,  2, name);
    this->w_i->Copy(this->w_i, tmp_w_i);
    Matrix* tmp_w_c = Init_Matrix(shape_w,  2, name);
    this->w_c->Copy(this->w_c, tmp_w_c);
    Matrix* tmp_w_o = Init_Matrix(shape_w, 2, name);
    this->w_o->Copy(this->w_o, tmp_w_o);

    int shape_b[2] = {1 , this->hidden_units};

    Matrix* tmp_b_f = Init_Matrix(shape_b, 2, "0");
    this->b_f->Copy(this->b_f, tmp_b_f);
    Matrix* tmp_b_i = Init_Matrix(shape_b, 2, "0");
    this->b_i->Copy(this->b_i, tmp_b_i);
    Matrix* tmp_b_c = Init_Matrix(shape_b, 2, "0");
    this->b_c->Copy(this->b_c, tmp_b_c);
    Matrix* tmp_b_o = Init_Matrix(shape_b, 2, "0");
    this->b_o->Copy(this->b_o, tmp_b_o);

    Init_Saver(this);

    tmp_w_f->Free(tmp_w_f);
    tmp_w_i->Free(tmp_w_i);
    tmp_w_c->Free(tmp_w_c);
    tmp_w_o->Free(tmp_w_o);

    tmp_b_f->Free(tmp_b_f);
    tmp_b_i->Free(tmp_b_i);
    tmp_b_c->Free(tmp_b_c);
    tmp_b_o->Free(tmp_b_o); 
    Set_Optimizer_lstm(this, optimizer);

 

}


Matrix* __Sigmoid(Matrix* input)
{
    Matrix* output = Init_Matrix(input->shape, input->size_shape, "null");
    for (int i = 0; i!= output->size_array; i++)
    {
        output->array[i] = 1/(1+exp(-input->array[i]));
    }
    return output;
}
Matrix* __Sigmoid_Prime(VAR* input, int start, int size, int shape[], int size_shape)
{
    Matrix* output = Init_Matrix(shape, size_shape, "null");
    for (int i = 0; i!= size; i++)
    {
        output->array[i] = input[i+start] * (1 - input[i+start]);
    }
    return output;
}
Matrix* __Sigmoid_Primes(VAR* input, int start, int size, Matrix* output)
{
    for (int i = 0; i!= size; i++)
    {
        output->array[i] = input[i+start] * (1 - input[i+start]);
    }
    return output;
}
Matrix* __Tanh(VAR* input, int start, int size, int shape[], int size_shape)
{

    Matrix* output = Init_Matrix(shape, size_shape, "null");
    #if USE_SIMD == 2
    {
        vdTanh(size, input+start, output->array);
    }
    #else
    {
        for (int i = 0; i!= size; i++)
        {
            VAR a = input[i+start];
            output->array[i] = tanh(a);
        }
    }
    #endif
    return output;
}
Matrix* __Tanh_Prime(Matrix* input)
{
    Matrix* output = Init_Matrix(input->shape, input->size_shape, "null");
    for (int i = 0; i!= output->size_array; i++)
    {
        output->array[i] = 1 - pow(tanh(input->array[i]), 2);
    }
    return output;
}
void Normalization(Matrix* gradient)
{
    VAR maxi = gradient->Max(gradient);
    gradient->Mult_Scalar(gradient, 1/maxi);
}
void __Little_Free(Matrix* matrix)
{
    free(matrix->shape);
    free(matrix);
}
void __Extand_batch(Matrix* to_change, Matrix* input, int index)
{
    if (to_change->shape[index] != input->shape[0])
    {
        to_change->shape[index] = input->shape[0];
        int new_size_array = 1;
        for (int i = 0; i!= to_change->size_shape; i++)
        {
            new_size_array *= to_change->shape[i];
        }
        to_change->size_array = new_size_array;
        to_change->array = realloc(to_change->array, to_change->size_array * sizeof(VAR));
    }
}
Matrix* Forward_Lstm(Lstm* this, Matrix* input)
{
    
    __Extand_batch(this->h_out, input, 0);
    __Extand_batch(this->c_out, input, 0);
    __Extand_batch(this->ft_list, input, 1);
    __Extand_batch(this->it_list, input, 1);
    __Extand_batch(this->ct_list, input, 1);
    __Extand_batch(this->ot_list, input, 1);
    __Extand_batch(this->concat_list, input, 1);
    input->Transpose(input, 1, 0);
    this->input->Copy(this->input, input);
    int shape_c_1[] = {this->h_out->shape[0], this->hidden_units};


    Matrix* c_1 = Init_Matrix(shape_c_1, 2, "0");
    Matrix* h_out_t_1 = Init_Matrix((int[]) {this->h_out->shape[0], this->h_out->shape[2]}, 2, "0");
    for (int t = 0; t!= this->h_out->shape[1]; t++)
    {

        //hstack
        Matrix* concat = input->__Slicing(*input, 0,  t, t+1);
        concat->Concat(concat, h_out_t_1, 1);  
        // concat->Print(concat);
        Matrix* tmp_ft = concat->__Dot(*concat, *(this->w_f));
        tmp_ft->Add(tmp_ft, this->b_f);
        Matrix* ft = __Sigmoid(tmp_ft);
        Matrix* tmp_it = concat->__Dot(*concat, *(this->w_i));
        tmp_it->Add(tmp_it, this->b_i);
        Matrix* it = __Sigmoid(tmp_it);

        Matrix* tmp_ct = concat->__Dot(*concat, *(this->w_c));
        tmp_ct->Add(tmp_ct, this->b_c);
        Matrix* ct = __Tanh(tmp_ct->array, 0, tmp_ct->size_array, tmp_ct->shape, tmp_ct->size_shape);
        
        Matrix* tmp_ot = concat->__Dot(*concat, (*this->w_o));
        tmp_ot->Add(tmp_ot, this->b_o);
        Matrix* ot = __Sigmoid(tmp_ot);

        Matrix* tmp_c_out = ft->__Mult(*ft, *c_1);
        Matrix* tmp_c_out2 = it->__Mult(*it, *ct);
        tmp_c_out->Add(tmp_c_out, tmp_c_out2);
        Matrix* tmp_tanh = __Tanh(tmp_c_out->array, 0, tmp_c_out->size_array, tmp_c_out->shape, tmp_c_out->size_shape);
        Matrix* tmp_h_out = ot->__Mult(*ot, *tmp_tanh);
        this->h_out->Put(this->h_out, tmp_h_out, 1, t, t+1);
        this->c_out->Put(this->c_out, tmp_c_out, 1, t, t+1);
        
        this->ft_list->Put(this->ft_list, ft, 0, t, t+1);
        this->it_list->Put(this->it_list, it, 0, t, t+1);
        this->ct_list->Put(this->ct_list, ct, 0, t, t+1);
        this->ot_list->Put(this->ot_list, ot, 0, t, t+1);
        this->concat_list->Put(this->concat_list, concat, 0, t, t+1);


        c_1->Copy(c_1, tmp_c_out);
        h_out_t_1->Copy(h_out_t_1, tmp_h_out);
        tmp_ft->Free(tmp_ft);
        tmp_it->Free(tmp_it);
        tmp_ct->Free(tmp_ct);
        tmp_ot->Free(tmp_ot);
        tmp_c_out2->Free(tmp_c_out2);
        tmp_tanh->Free(tmp_tanh);
        tmp_h_out->Free(tmp_h_out);
        tmp_c_out->Free(tmp_c_out);
        ft->Free(ft);
        it->Free(it);
        ct->Free(ct);
        ot->Free(ot);
        concat->Free(concat);
    }
    c_1->Free(c_1);
    // this->h_out->Transpose(this->h_out, 0, 1);
    h_out_t_1->Free(h_out_t_1);
    if (this->return_sequences == 0)
    {
        Matrix* tmp = this->h_out->__Slicing(*(this->h_out), 1, this->h_out->shape[1]-1, this->h_out->shape[1]);
        return tmp;
    }
    return this->h_out->__Copy(*(this->h_out));
}   

void __Slice_lstm(  Matrix* c_out_tanh, Matrix* c_out_tanh_t, 
                    Matrix* c_out_t, Matrix* concat_t,
                    Matrix* ot_t, Matrix* ct_t,
                    Matrix* it_t, Matrix* c_t_liste_tanh,
                    Matrix* c_t_liste_tanh_t, Lstm* this,
                    int t)
{
    // Tanh[c_out[t]]
    __Slicing3d_Matrix(*c_out_tanh, 1, t, t+1, c_out_tanh_t);
    // Ot[t]
    __Slicing3d_Matrix(*this->c_out, 1, t, t+1, c_out_t);
    // Concat[t]
    __Slicing3d_Matrix(*this->concat_list, 0, t, t+1, concat_t);
    concat_t->Transpose(concat_t, 0, -1);
    __Slicing3d_Matrix(*this->ot_list, 0, t, t+1, ot_t);
    __Slicing3d_Matrix(*this->ct_list, 0, t, t+1, ct_t);
    __Slicing3d_Matrix(*this->it_list, 0, t, t+1, it_t);
    __Slicing3d_Matrix(*c_t_liste_tanh, 0, t, t+1, c_t_liste_tanh_t);
}
Matrix* Backprop_Lstm(Lstm* this, Matrix* output)
{
    
    Matrix* input_gradient = Init_Matrix(this->input->shape, this->input->size_shape, "null");
    int first_index = this->h_out->shape[2] * this->h_out->shape[0];
    // Tanh c_out

    Matrix* c_out_tanh = __Tanh(this->c_out->array, 0, this->c_out->size_array, this->c_out->shape, this->c_out->size_shape);
    Matrix* c_t_liste_tanh = __Tanh(this->ct_list->array, 0, this->ct_list->size_array, this->ct_list->shape, this->ct_list->size_shape);
    c_t_liste_tanh->Power(c_t_liste_tanh, 2);
    c_t_liste_tanh->Mult_Scalar(c_t_liste_tanh, -1);
    c_t_liste_tanh->Add_Scalar(c_t_liste_tanh, 1);

    Matrix* dLoss_dWo = Init_Matrix(this->w_o->shape, this->w_o->size_shape, "0");
    Matrix* dLoss_dWf = Init_Matrix(this->w_f->shape, this->w_f->size_shape, "0");
    Matrix* dLoss_dWi = Init_Matrix(this->w_i->shape, this->w_i->size_shape, "0");
    Matrix* dLoss_dWc = Init_Matrix(this->w_c->shape, this->w_c->size_shape, "0");
    
    Matrix* dLoss_dBo = Init_Matrix(this->b_o->shape, this->b_o->size_shape, "0");
    Matrix* dLoss_dBf = Init_Matrix(this->b_f->shape, this->b_f->size_shape, "0");
    Matrix* dLoss_dBi = Init_Matrix(this->b_i->shape, this->b_i->size_shape, "0");
    Matrix* dLoss_dBc = Init_Matrix(this->b_c->shape, this->b_c->size_shape, "0");
    Matrix* input_gadient_1 = output->__Copy(*output);

    Matrix* c_out_tanh_t = Init_Matrix((int[]){c_out_tanh->shape[0], c_out_tanh->shape[2]}, 2, "null");
    Matrix* c_out_t = Init_Matrix((int[]){this->c_out->shape[0], this->c_out->shape[2]}, 2, "null");
    Matrix* concat_t = Init_Matrix((int[]){this->concat_list->shape[1], this->concat_list->shape[2]}, 2, "null");
    Matrix* ot_t = Init_Matrix((int[]){this->ot_list->shape[1], this->ot_list->shape[2]}, 2, "null");
    Matrix* ct_t = Init_Matrix((int[]){this->ct_list->shape[1], this->ct_list->shape[2]}, 2, "null");
    Matrix* it_t = Init_Matrix((int[]){this->it_list->shape[1], this->it_list->shape[2]}, 2, "null");
    Matrix* c_t_liste_tanh_t = Init_Matrix((int[]){c_t_liste_tanh->shape[1], c_t_liste_tanh->shape[2]}, 2, "null");

    Matrix* tanh_prime = Init_Matrix((int[]){c_out_tanh->shape[0], c_out_tanh->shape[2]}, 2, "1");

    // define the matrix of the sigmoid prime
    int shape_ot[] = {this->ot_list->shape[1], this->ot_list->shape[2]};
    Matrix* tmp_sigma= Init_Matrix(shape_ot, 2, "null");
    Matrix* dLoss_sigma_it = Init_Matrix(shape_ot, 2, "null");
    Matrix* dLoss_sigma_ft = Init_Matrix(shape_ot, 2, "null");

    // Multiplication matrix
    // Shape of c_out_tanh : (batch, time_step, lstm_nb);
    int shape_mult[] = {c_out_tanh->shape[0], c_out_tanh->shape[2]};
    Matrix* dLoss_dOt = Init_Matrix(shape_mult, 2, "null");
    Matrix* tmp_ot_list = Init_Matrix(shape_mult, 2, "null");
    Matrix* dLoss_dtanh = Init_Matrix(shape_mult, 2, "null");
    Matrix* dLoss_dCount = Init_Matrix(shape_mult, 2, "null");
    Matrix* tmp_dCount = Init_Matrix(shape_mult, 2, "null");
    Matrix* dLoss_dIt = Init_Matrix(shape_mult, 2, "null");
    Matrix* dLoss_dFt = Init_Matrix(shape_mult, 2, "null");
    //gradient (Weights and bias)
    int shape_gradients[] = {this->w_c->shape[1], this->w_c->shape[0]}; // (nb_lstm, embedding + hidden)
    Matrix* w_f_t = Init_Matrix(shape_gradients, 2, "null");
    Matrix* w_i_t = Init_Matrix(shape_gradients, 2, "null");
    Matrix* w_c_t = Init_Matrix(shape_gradients, 2, "null");
    Matrix* w_o_t = Init_Matrix(shape_gradients, 2, "null");

    shape_gradients[0] = c_out_tanh->shape[0];
    Matrix* dLoss_dO_dInput = Init_Matrix(shape_gradients, 2, "null");
    Matrix* dLoss_dI_dInput = Init_Matrix(shape_gradients, 2, "null");
    Matrix* dLoss_dF_dInput = Init_Matrix(shape_gradients, 2, "null");
    Matrix* dLoss_dC_dInput = Init_Matrix(shape_gradients, 2, "null");

    shape_gradients[0] = this->w_c->shape[0];
    shape_gradients[1] = this->w_c->shape[1];
    // Matrix* dLoss_dHout_tmp = Init_Matrix(shape_gradients, 2, "null");

    Matrix* tmp_dWc = Init_Matrix(shape_gradients, 2, "null");
    Matrix* tmp_dWo = Init_Matrix(shape_gradients, 2, "null");
    Matrix* tmp_dWi = Init_Matrix(shape_gradients, 2, "null");
    Matrix* tmp_dWf = Init_Matrix(shape_gradients, 2, "null");

    int shape_dBias[] = {1, this->hidden_units};
    Matrix* tmp_dBo = Init_Matrix(shape_dBias, 2, "null");
    Matrix* tmp_dBi = Init_Matrix(shape_dBias, 2, "null");
    Matrix* tmp_dBf = Init_Matrix(shape_dBias, 2, "null");
    Matrix* tmp_dBc = Init_Matrix(shape_dBias, 2, "null");

    for (int t = this->h_out->shape[1]-1; t>-1; t--)
    {
        concat_t->shape[0] = this->concat_list->shape[1];
        concat_t->shape[1] = this->concat_list->shape[2];
        
        __Slice_lstm(c_out_tanh, c_out_tanh_t, 
                    c_out_t, concat_t,
                    ot_t, ct_t,
                    it_t, c_t_liste_tanh,
                    c_t_liste_tanh_t, this,
                    t);
        // c_out_tanh_t->Print_Shape(c_out_tanh_t);
        __Power(c_out_tanh_t, 2, tanh_prime);
        // Matrix* tanh_prime = c_out_tanh_t->__Power(*c_out_tanh_t, 2);
        tanh_prime->Mult_Scalar(tanh_prime, -1);
        tanh_prime->Add_Scalar(tanh_prime, 1);

        Matrix* dLoss_dHout ;
        if (this->return_sequences == 1)
            dLoss_dHout = output->__Slicing(*output, 1, t,t+1);
        else
            dLoss_dHout = input_gadient_1->__Copy(*input_gadient_1);
        //Output Gate

        __Mult_Mat_Place(*dLoss_dHout, *c_out_tanh_t, dLoss_dOt);
        
        __Sigmoid_Primes(this->ot_list->array, t*first_index, first_index, tmp_sigma);
        tmp_sigma->Mult(tmp_sigma, dLoss_dOt);
        __Mult_Mat_Place(*ot_t, *dLoss_dHout, tmp_ot_list);

        __Mult_Mat_Place(*tanh_prime, *tmp_ot_list, dLoss_dtanh);
        __Mult_Mat_Place(*dLoss_dtanh, *tmp_ot_list, dLoss_dCount);
        __Mult_Mat_Place(*dLoss_dCount, *c_t_liste_tanh_t, tmp_dCount);
        //Cell State

        //Input Gate
        __Mult_Mat_Place(*dLoss_dtanh, *ct_t, dLoss_dIt);
        __Sigmoid_Primes(this->it_list->array, t*first_index, first_index, dLoss_sigma_it);
        dLoss_sigma_it->Mult(dLoss_sigma_it, dLoss_dIt);
        //Forget Gate
        __Mult_Mat_Place(*dLoss_dtanh, *c_out_t, dLoss_dFt);

        __Sigmoid_Primes(this->ft_list->array, t*first_index, first_index, dLoss_sigma_ft);
        dLoss_sigma_ft->Mult(dLoss_sigma_ft, dLoss_dFt);
        // transpose the weight matrix        
        __Transpose_Matrix_Place(*(this->w_f), 0, -1, w_f_t);
        __Transpose_Matrix_Place(*(this->w_i), 0, -1, w_i_t);
        __Transpose_Matrix_Place(*(this->w_c), 0, -1, w_c_t);
        __Transpose_Matrix_Place(*(this->w_o), 0, -1, w_o_t);

        //input gradient
        
        __Dot_Matrix_2D_Place(*dLoss_dOt, *w_o_t, dLoss_dO_dInput);
        __Dot_Matrix_2D_Place(*dLoss_dIt, *w_i_t, dLoss_dI_dInput);
        __Dot_Matrix_2D_Place(*dLoss_dFt, *w_f_t, dLoss_dF_dInput);
        __Dot_Matrix_2D_Place(*dLoss_dCount, *w_c_t, dLoss_dC_dInput);

        Matrix* dLoss_dHout_tmp = dLoss_dO_dInput->__Add(*dLoss_dO_dInput, *dLoss_dI_dInput);

        if ( this->return_sequences == 1 || t == this->h_out->shape[1] - 1 )
        {
            __Dot_Matrix_2D_Place(*concat_t, *tmp_dCount, tmp_dWc);
            dLoss_dWc->Add(dLoss_dWc, tmp_dWc);
            __Dot_Matrix_2D_Place(*concat_t, *tmp_sigma, tmp_dWo);
            dLoss_dWo->Add(dLoss_dWo, tmp_dWo);
            __Dot_Matrix_2D_Place(*concat_t, *dLoss_sigma_it, tmp_dWi);
            dLoss_dWi->Add(dLoss_dWi, tmp_dWi);
            __Dot_Matrix_2D_Place(*concat_t, *dLoss_sigma_ft, tmp_dWf);
            dLoss_dWf->Add(dLoss_dWf, tmp_dWf);

            __Sum_Matrix_Place(*dLoss_dOt, 0, tmp_dBo);
            dLoss_dBo->Add(dLoss_dBo, tmp_dBo);
            __Sum_Matrix_Place(*dLoss_dIt, 0, tmp_dBi);
            dLoss_dBi->Add(dLoss_dBi, tmp_dBi);
            __Sum_Matrix_Place(*dLoss_dFt, 0, tmp_dBf);
            dLoss_dBf->Add(dLoss_dBf, tmp_dBf);
            __Sum_Matrix_Place(*dLoss_dCount, 0, tmp_dBc);
            dLoss_dBc->Add(dLoss_dBc, tmp_dBc);
        }
        dLoss_dHout->Free(dLoss_dHout);

        dLoss_dHout_tmp->Add(dLoss_dHout_tmp, dLoss_dF_dInput);
        dLoss_dHout_tmp->Add(dLoss_dHout_tmp, dLoss_dC_dInput);
        Matrix* tmp_gradient_input = dLoss_dHout_tmp->__Slicing(*dLoss_dHout_tmp, 1, 0, this->embedding);
        dLoss_dHout_tmp->Slicing(dLoss_dHout_tmp, 1, this->embedding, dLoss_dHout_tmp->shape[1]);
        Normalization(dLoss_dHout_tmp);
        input_gadient_1->Copy(input_gadient_1, dLoss_dHout_tmp);
        // !!!!!!!!!!!!!!!!!!!!!!!!!!! not sur about the axis 0
        // tmp_gradient_input->Print(tmp_gradient_input);

        input_gradient->Put(input_gradient, tmp_gradient_input, 0, t, t+1);
        //update weight      

        tmp_gradient_input->Free(tmp_gradient_input);
        
        dLoss_dHout_tmp->Free(dLoss_dHout_tmp);

    }

    this->Optimizer_b_c->Optimize(this->Optimizer_b_c, dLoss_dBc);
    this->Optimizer_b_f->Optimize(this->Optimizer_b_f, dLoss_dBf);
    this->Optimizer_b_i->Optimize(this->Optimizer_b_i, dLoss_dBi);
    this->Optimizer_b_o->Optimize(this->Optimizer_b_o, dLoss_dBo);

    this->Optimizer_w_c->Optimize(this->Optimizer_w_c, dLoss_dWc);
    this->Optimizer_w_f->Optimize(this->Optimizer_w_f, dLoss_dWf);
    this->Optimizer_w_i->Optimize(this->Optimizer_w_i, dLoss_dWi);
    this->Optimizer_w_o->Optimize(this->Optimizer_w_o, dLoss_dWo);

    dLoss_dWo->Free(dLoss_dWo);
    dLoss_dWi->Free(dLoss_dWi);
    dLoss_dWf->Free(dLoss_dWf);
    dLoss_dWc->Free(dLoss_dWc);

    dLoss_dBo->Free(dLoss_dBo);
    dLoss_dBi->Free(dLoss_dBi);
    dLoss_dBf->Free(dLoss_dBf);
    dLoss_dBc->Free(dLoss_dBc);
    c_out_tanh_t->Free(c_out_tanh_t);
    c_out_t->Free(c_out_t);
    ot_t->Free(ot_t);
    ct_t->Free(ct_t);
    it_t->Free(it_t);
    c_t_liste_tanh_t->Free(c_t_liste_tanh_t);
    tanh_prime->Free(tanh_prime);

    dLoss_sigma_it->Free(dLoss_sigma_it);
    dLoss_sigma_ft->Free(dLoss_sigma_ft);
    tmp_sigma->Free(tmp_sigma);

    input_gadient_1->Free(input_gadient_1);
    concat_t->Free(concat_t);

    c_out_tanh->Free(c_out_tanh);
    c_t_liste_tanh->Free(c_t_liste_tanh);
    input_gradient->Transpose(input_gradient, 0, 1);
    //mult free
    dLoss_dOt->Free(dLoss_dOt);
    tmp_ot_list->Free(tmp_ot_list);
    dLoss_dtanh->Free(dLoss_dtanh);
    dLoss_dCount->Free(dLoss_dCount);
    tmp_dCount->Free(tmp_dCount);
    dLoss_dIt->Free(dLoss_dIt);
    dLoss_dFt->Free(dLoss_dFt);
    // weigts gradients
    w_i_t->Free(w_i_t);
    w_f_t->Free(w_f_t);
    w_c_t->Free(w_c_t);
    w_o_t->Free(w_o_t);

    dLoss_dO_dInput->Free(dLoss_dO_dInput);
    dLoss_dI_dInput->Free(dLoss_dI_dInput);
    dLoss_dF_dInput->Free(dLoss_dF_dInput);
    dLoss_dC_dInput->Free(dLoss_dC_dInput);

    tmp_dWc->Free(tmp_dWc);
    tmp_dWo->Free(tmp_dWo);
    tmp_dWi->Free(tmp_dWi);
    tmp_dWf->Free(tmp_dWf);

    tmp_dBo->Free(tmp_dBo);
    tmp_dBi->Free(tmp_dBi);
    tmp_dBf->Free(tmp_dBf);
    tmp_dBc->Free(tmp_dBc);
    return input_gradient;
}
void Free_Lstm(Lstm* this)
{
    this->w_f->Free(this->w_f);
    this->w_i->Free(this->w_i);
    this->w_c->Free(this->w_c);
    this->w_o->Free(this->w_o);
    this->b_f->Free(this->b_f);
    this->b_i->Free(this->b_i);
    this->b_c->Free(this->b_c);
    this->b_o->Free(this->b_o);
    this->input->Free(this->input);
    this->ct_list->Free(this->ct_list);
    this->ft_list->Free(this->ft_list);
    this->it_list->Free(this->it_list);
    this->ot_list->Free(this->ot_list);
    this->concat_list->Free(this->concat_list);
    this->h_out->Free(this->h_out);
    this->c_out->Free(this->c_out); 
    this->Optimizer_w_f->Free(this->Optimizer_w_f);
    this->Optimizer_w_i->Free(this->Optimizer_w_i);
    this->Optimizer_w_c->Free(this->Optimizer_w_c);
    this->Optimizer_w_o->Free(this->Optimizer_w_o);
    this->Optimizer_b_f->Free(this->Optimizer_b_f);
    this->Optimizer_b_i->Free(this->Optimizer_b_i);
    this->Optimizer_b_c->Free(this->Optimizer_b_c);
    this->Optimizer_b_o->Free(this->Optimizer_b_o);
    
    free(this);
}

void Save_Lstm(Lstm* this, hid_t file_id)
{
    hid_t lstm_group = H5Gcreate(file_id, "lstm", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // hidden units
    hsize_t dims[1] = {1};
    hid_t dataspace_units = H5Screate_simple(1, dims, NULL);
    hid_t dataset_units = H5Dcreate(lstm_group, "hidden_units", H5T_NATIVE_INT, dataspace_units, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_units, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(this->hidden_units));
    H5Dclose(dataset_units);
    H5Sclose(dataspace_units);

    dims[0] = this->input->size_shape;
    this->input->Transpose(this->input, 1, 0);
    hid_t shape_space = H5Screate_simple(1, dims, NULL);
    hid_t shape_dataset = H5Dcreate(lstm_group, "input_shape", H5T_NATIVE_INT, shape_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->input->shape);
    //Close
    H5Dclose(shape_dataset);
    H5Sclose(shape_space);

    //return_sequences
    dims[0] = 1;
    hid_t dataspace_sequence = H5Screate_simple(1, dims, NULL);
    hid_t dataset_sequence = H5Dcreate(lstm_group, "return_sequence", H5T_NATIVE_INT, dataspace_sequence, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_sequence, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(this->return_sequences));
    H5Dclose(dataset_sequence);
    H5Sclose(dataspace_sequence);



    //Weights
    this->w_c->Save(this->w_c, lstm_group, "w_c");
    this->w_i->Save(this->w_i, lstm_group, "w_i");
    this->w_o->Save(this->w_o, lstm_group, "w_o");
    this->w_f->Save(this->w_f, lstm_group, "w_f");

    this->b_c->Save(this->b_c, lstm_group, "b_c");
    this->b_i->Save(this->b_i, lstm_group, "b_i");
    this->b_o->Save(this->b_o, lstm_group, "b_o");
    this->b_f->Save(this->b_f, lstm_group, "b_f");

    H5Gclose(lstm_group);
}

Layer* Load_Lstm(hid_t file_id, Optimizer* optimizer)
{
    Lstm* lstm = malloc(sizeof(Lstm));

    hid_t lstm_group = H5Gopen(file_id, "lstm", H5P_DEFAULT);
    // hidden units
    hid_t dataset_units = H5Dopen(lstm_group, "hidden_units", H5P_DEFAULT);
    H5Dread(dataset_units, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(lstm->hidden_units));
    H5Dclose(dataset_units);

    //return_sequences
    hid_t dataset_sequence = H5Dopen(lstm_group, "return_sequence", H5P_DEFAULT);
    H5Dread(dataset_sequence, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(lstm->return_sequences));
    H5Dclose(dataset_sequence);

    //Weights
    lstm->w_c = Load_Matrix(lstm_group, "w_c");
    lstm->w_i = Load_Matrix(lstm_group, "w_i");
    lstm->w_o = Load_Matrix(lstm_group, "w_o");
    lstm->w_f = Load_Matrix(lstm_group, "w_f");

    lstm->b_c = Load_Matrix(lstm_group, "b_c");
    lstm->b_i = Load_Matrix(lstm_group, "b_i");
    lstm->b_o = Load_Matrix(lstm_group, "b_o");
    lstm->b_f = Load_Matrix(lstm_group, "b_f");
    Set_Optimizer_lstm(lstm, optimizer);
    //input shape
    hid_t shape_dataset = H5Dopen(lstm_group, "input_shape", H5P_DEFAULT);
    hid_t dataspace_size_shape = H5Dget_space(shape_dataset);
    hsize_t size_shape = H5Sget_simple_extent_npoints(dataspace_size_shape);
    
    int* input_shape = malloc(sizeof(int) * size_shape);
    H5Dread(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_shape);
    lstm->input = Init_Matrix(input_shape, size_shape, "null");
    lstm->embedding = lstm->input->shape[lstm->input->size_shape-1];

    Init_Saver(lstm);
    //Close
    H5Dclose(shape_dataset);

    H5Gclose(lstm_group);
    lstm->Forward = Forward_Lstm;
    lstm->Backprop = Backprop_Lstm;
    lstm->Initialize = Initialize_Lstm;
    lstm->Free = Free_Lstm;
    lstm->Save = Save_Lstm;
    Layer* layer = Init_Layer();
    layer->inheritance.lstm = lstm;
    layer->layer = "lstm";
    free(input_shape);
    return layer;
}


Layer* Init_Lstm(int hidden_units, int return_sequences)
{
    Lstm* lstm = malloc(sizeof(Lstm));
    lstm->Forward = Forward_Lstm;
    lstm->Backprop = Backprop_Lstm;
    lstm->Initialize = Initialize_Lstm;
    lstm->Free = Free_Lstm;
    lstm->Save = Save_Lstm;

    int shape[2] = {1,1};
    lstm->input = Init_Matrix(shape,2,"null");
    lstm->return_sequences = return_sequences;
    lstm->hidden_units = hidden_units;
    lstm->embedding = 0;    

    lstm->w_f = Init_Matrix(shape,2,"null");
    lstm->w_i = Init_Matrix(shape,2,"null");
    lstm->w_c = Init_Matrix(shape,2,"null");
    lstm->w_o = Init_Matrix(shape,2,"null");
    lstm->b_f = Init_Matrix(shape,2,"null");
    lstm->b_i = Init_Matrix(shape,2,"null");
    lstm->b_c = Init_Matrix(shape,2,"null");
    lstm->b_o = Init_Matrix(shape,2,"null");

    Layer* layer = Init_Layer();
    layer->inheritance.lstm = lstm;
    layer->layer = "lstm";
    
    return layer;
}