#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>
#include <cmath>
#include <ctime>        // time
#include <cstdlib>      // srand,rand

class nnet
{
 public:
  nnet(int inputnum,int hiddennum,int outputnum);
  ~nnet();

 private:
  double **W_itoh;  // weight input to hidden
  double **W_htoo;  // weight hidden to output
  double *X_i;  // X inpit
  double *X_h;  // X hidden
  double *X_o;  // X output
  double *bias_h;   // bias hidden layer
  double *bias_o;   // bias output layer
  int inputnum;
  int hiddennum;
  int outputnum;

  const double Eta;
  const double Alpha;
  const double ErrorEv;
  const double Rlow;
  const double Rhigh;

  #define activationFunc(x) (1/(1+exp(-x))))
  #define urand() ((double) rand()/RAND_MAX * (Rhigh - Rlow) + Rlow)
  
};

#endif
