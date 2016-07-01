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
  void setInData(const int &pnum,const int &i,const double &value);
  void setTeachData(const int &pnum,const int &i,const double &value);
  void setPredictData(const int &i,const double &value);
  void train();
  void predict();

 private:
  // parameter
  int inputnum;
  int hiddennum;
  int outputnum;
  int patternnum;
  double **X_i;         // X input
  double *X_h;          // X hidden
  double *X_o;          // X output

 // foward propagation
  double **W_itoh;      // weight input to hidden
  double **W_htoo;      // weight hidden to output
  double *bias_h;       // bias hidden layer
  double *bias_o;       // bias output layer


  // back propagation
  double **T_signal;      // teach signal
  double **W_itoh_prev;   // weight input to hidden
  double **W_htoo_prev;   // weight hidden to output
  double *bias_h_prev;    // bias hidden layer
  double *bias_o_prev;    // bias output layer
  double *dwih;           // 隠れ層での学習信号
  double *dwho;           // 出力層での学習信号
  

  // const parameter
  const double Eta;
  const double Alpha;
  const double ErrorEv;
  const double Rlow;
  const double Rhigh;
  const int MaxGen;

  #define activationFunc(x) (1/(1+exp(-(x))))   // -xではNG -(x) とすること
  #define activationFunc_diff(x) (x*(1-x))
  #define urand() ((double) rand()/RAND_MAX * (Rhigh - Rlow) + Rlow)

  // functions
  void foward_propagation(const int &pnum);
  void back_propagation(const int &pnum);
  double random() ;
  
};

#endif
