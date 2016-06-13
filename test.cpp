/* 
 * nnet test
 * 2016.06.06
 * author:soma62jp
 *                  */


#include "test.h"

using namespace std;

nnet::nnet(int inputnum,int hiddennum,int outputnum):
      inputnum(inputnum)
     ,hiddennum(hiddennum)
     ,outputnum(outputnum)
     ,Eta(0.75)
     ,Alpha(0.8)
     ,ErrorEv(0.08)
     ,Rlow(-0.30)
     ,Rhigh(0.30)
{
  //this->inputnum=inputnum;
  //this->hiddennum=hiddennum;
  //this->outputnum=outputnum;

  X_i = new double[inputnum];
  X_h = new double[hiddennum];
  X_o = new double[outputnum];

  bias_h = new double[hiddennum];
  bias_o = new double[outputnum];

  // hidden*input
  W_itoh = new double*[hiddennum];
  for(int i=0;i<hiddennum;i++){
    W_itoh[i] = new double[inputnum];
  }

  // output*hidden
  W_htoo = new double*[outputnum];
  for(int i=0;i<outputnum;i++){
    W_htoo[i] = new double[hiddennum];
  }


  //initialize parameter
  srand((unsigned int)time(NULL));

  for(int i=0;i<hiddennum;i++){
    for(int j=0;j<inputnum;j++){
      W_itoh[i][j]=urand();
    }
  }

  for(int i=0;i<outputnum;i++){
    for(int j=0;j<hiddennum;j++){
      W_htoo[i][j]=urand();
    }
  }



}

nnet::~nnet()
{

  delete [] X_i;
  delete [] X_h;
  delete [] X_o;

  delete [] bias_h;
  delete [] bias_o;
  
  for( int i=0; i<hiddennum; i++ ) {
    delete[] W_itoh[i];
  }
  delete [] W_itoh;

  for( int i=0; i<outputnum; i++ ) {
    delete [] W_htoo[i];
  }
  delete [] W_htoo;
}

int main()
{
  cout << "Hellow World!" << endl;
}
