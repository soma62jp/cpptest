#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>
#include <cmath>
#include <ctime>        // time
#include <cstdlib>      // srand,rand

class nnet
{
 public:
  nnet(int inputnum,int hiddennum,int outputnum,int patternnum);
  ~nnet();

 private:
  // parameter
  int inputnum;
  int hiddennum;
  int outputnum;
  int patternnum;
  double **X_i;      // X input
  double *X_h;      // X hidden
  double *X_o;      // X output

 // foward propagation
  double **W_itoh;  // weight input to hidden
  double **W_htoo;  // weight hidden to output
  double *bias_h;   // bias hidden layer
  double *bias_o;   // bias output layer


  // back propagation
  double **T_signal;// teach signal

  // const parameter
  const double Eta;
  const double Alpha;
  const double ErrorEv;
  const double Rlow;
  const double Rhigh;

  #define activationFunc(x) (1/(1+exp(-x))))
  #define urand() ((double) rand()/RAND_MAX * (Rhigh - Rlow) + Rlow)
  
};

#endif
