#ifndef LSTM_H
#define LSTM_H

#include <stddef.h>
#include "../matrix/Matrix.h"
#include "Layer.h"
#include "Optimizer.h"
struct Lstm 
{
    Matrix* (*Forward)(struct Lstm* lstm, Matrix* input);
    Matrix* (*Backprop)(struct Lstm* lstm, Matrix* output);
    void (*Initialize)(struct Lstm* lstm, Matrix* input_matrix, Optimizer* optimizer);  
    void (*Free)(struct Lstm* lstm); 
    void(*Save)(struct Lstm* lstm, hid_t file_id);
    
    Matrix* w_f;
    Matrix* w_i;
    Matrix* w_c;
    Matrix* w_o;

    Matrix* b_f;
    Matrix* b_i;
    Matrix* b_c;
    Matrix* b_o;

    Matrix* h_out;
    Matrix* c_out;

    Matrix* ft_list;
    Matrix* it_list;
    Matrix* ct_list;
    Matrix* ot_list;
    Matrix* concat_list;

    Matrix* input;

    int embedding;
    int hidden_units;
    int return_sequences;

    Optimizer* Optimizer_w_f;
    Optimizer* Optimizer_w_i;
    Optimizer* Optimizer_w_c;
    Optimizer* Optimizer_w_o;

    Optimizer* Optimizer_b_f;
    Optimizer* Optimizer_b_i;
    Optimizer* Optimizer_b_c;
    Optimizer* Optimizer_b_o;
};
struct Layer;
typedef struct Lstm Lstm ;
struct Layer* Init_Lstm(int hidden_units, int return_sequences);
struct Layer* Load_Lstm(hid_t file_id, Optimizer* optimizer);
#endif