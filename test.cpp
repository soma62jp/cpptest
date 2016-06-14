/* 
 * nnet test
 * 2016.06.06
 * author:soma62jp
 *                  */


#include "test.h"

using namespace std;

nnet::nnet(int inputnum,int hiddennum,int outputnum,int patternnum):
      inputnum(inputnum)
     ,hiddennum(hiddennum)
     ,outputnum(outputnum)
     ,patternnum(patternnum)
     ,Eta(0.75)
     ,Alpha(0.8)
     ,ErrorEv(0.01)
     ,Rlow(-0.30)
     ,Rhigh(0.30)
     ,MaxGen(3000)
{
  //this->inputnum=inputnum;
  //this->hiddennum=hiddennum;
  //this->outputnum=outputnum;

  X_h = new double[hiddennum];
  X_o = new double[outputnum];

  bias_h = new double[hiddennum];
  bias_o = new double[outputnum];
  bias_h_prev = new double[hiddennum];
  bias_o_prev = new double[outputnum];

  // pattern*input
  X_i = new double*[patternnum];
  for(int i=0;i<patternnum;i++){
    X_i[i] = new double[inputnum];
  }

  // pattern*output
  T_signal = new double*[patternnum];
  for(int i=0;i<patternnum;i++){
    T_signal[i] = new double[outputnum];
  }

  // hidden*input
  W_itoh = new double*[hiddennum];
  for(int i=0;i<hiddennum;i++){
    W_itoh[i] = new double[inputnum];
  }
  W_itoh_prev = new double*[hiddennum];
  for(int i=0;i<hiddennum;i++){
    W_itoh_prev[i] = new double[inputnum];
  }

  // output*hidden
  W_htoo = new double*[outputnum];
  for(int i=0;i<outputnum;i++){
    W_htoo[i] = new double[hiddennum];
  }
  W_htoo_prev = new double*[outputnum];
  for(int i=0;i<outputnum;i++){
    W_htoo_prev[i] = new double[hiddennum];
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

  for(int i=0;i<hiddennum;i++){
    bias_h[i] = 0;
  }

  for(int i=0;i<outputnum;i++){
    bias_o[i] = 0;
  }



}

nnet::~nnet()
{

  delete [] X_h;
  delete [] X_o;

  delete [] bias_h;
  delete [] bias_o;
  delete [] bias_h_prev;
  delete [] bias_o_prev;

  for( int i=0; i<patternnum; i++ ) {
    delete[] X_i[i];
  }
  delete [] X_i;

  for( int i=0; i<patternnum; i++ ) {
    delete[] T_signal[i];
  }
  delete [] T_signal;
  
  for( int i=0; i<hiddennum; i++ ) {
    delete[] W_itoh[i];
  }
  delete [] W_itoh;

  for( int i=0; i<hiddennum; i++ ) {
    delete[] W_itoh_prev[i];
  }
  delete [] W_itoh_prev;

  for( int i=0; i<outputnum; i++ ) {
    delete [] W_htoo[i];
  }
  delete [] W_htoo;

  for( int i=0; i<outputnum; i++ ) {
    delete [] W_htoo_prev[i];
  }
  delete [] W_htoo_prev;



}

void nnet::foward_propagation(const int pnum)
{
  int i,j;
  double sum;

  // 入力層ー＞隠れ層
  for(i=0;i<hiddennum;i++){
    sum=0;
    for(j=0;j<inputnum;j++){
      // 重み×入力値
      sum+=W_itoh[i][j]*X_i[pnum][j];
    }
    // 重み×入力値の総和にバイアス項を足してアクティベーション関数に通したものが中間層入力
    X_h[i] = activationFunc(sum+bias_h[i]);
  }

  // 隠れ層ー＞出力層
  for(i=0;i<outputnum;i++){
    sum=0;
    for(j=0;j<hiddennum;j++){
      // 重み×中間層入力
      sum+=W_htoo[i][j]*X_h[j];
    }
    // 重み×中間層入力値の総和にバイアス項を足してアクティベーション関数に通したものが出力層
    X_o[i]=activationFunc(sum+bias_o[i]);
  }

}

void nnet::back_propagation(const int pnum)
{
  int i,j;
  double sum;
  double *dwih = new double[hiddennum];   // 隠れ層での学習信号
  double *dwho = new double[outputnum];   // 出力層での学習信号

  // 出力層の学習信号から計算
  for(i=0;i<outputnum;i++){
    // 出力層での学習信号=(教師信号-出力）*出力*(1-出力)
    // 出力*(1-出力)はシグモイド関数の微分
    dwho[i]=(T_signal[pnum][i]-X_o[i]) * X_o[i] * (1.0-X_o[i]);
  }

  // 重みの変化量[隠れ層ー＞出力層]を計算
  // 隠れ層の学習信号を計算
  for(i=0;i<hiddennum;i++){
    sum=0;
    for(j=0;j<outputnum;j++){
      // 前回の重みの変化量[隠れ層->出力層] = η * 出力層での学習信号 * 隠れ層出力 * α * 前回の重みの変化量[隠れ層->出力層]
      W_htoo_prev[j][i]=Eta*dwho[j]*X_h[i]+Alpha*W_htoo_prev[j][i];
      // 重みの変化量[隠れ層ー＞出力層] = 重みの変化量[隠れ層ー＞出力層] + 前回の重みの変化量[隠れ層ー＞出力層]
      W_htoo[j][i]+=W_htoo_prev[j][i];
      // 出力層での学習信号 * 重みの変化量[隠れ層ー＞出力層]
      sum += dwho[j]*W_htoo[j][i];
    }
    // 隠れ層での学習信号 = 隠れ層出力 * (1 - 隠れ層出力) * sum
    dwih[i]=X_h[i]*(1-X_h[i])*sum;
  }

  // 出力層のバイアス項を計算
  for(i=0;i<outputnum;i++){
    // 前回のバイアス項[出力層] = η * 出力層での学習信号 + α * 前回のバイアス項[出力層]
    bias_o_prev[i] = Eta*dwho[i]+Alpha*bias_o_prev[i];
    // 出力層バイアス項[出力層] = 出力層バイアス項[出力層] + 前回の出力層バイアス項[出力層]
    bias_o[i]+=bias_o_prev[i];
  }

  // 重みの変化量[入力層ー＞隠れ層]を計算
  for(i=0;i<inputnum;i++){
    for(j=0;j<hiddennum;j++){
      // 前回の重みの変化量[入力層ー＞隠れ層] = η * 隠れ層での学習信号 * 入力層出力 * α * 前回の重みの変化量[入力層ー＞隠れ層]
      W_itoh_prev[j][i]=Eta*dwih[j]*X_i[pnum][i]+Alpha*W_itoh_prev[j][i];
      // 重みの変化量[入力層ー＞隠れ層] = 重みの変化量[入力層ー＞隠れ層] + 前回の重みの変化量[入力層ー＞隠れ層]
      W_itoh[j][i]+=W_itoh_prev[j][i];
    }
  }

  // 隠れ層のバイアス項を計算
  for(i=0;i<hiddennum;i++){
    // 前回のバイアス項[隠れ層] = η * 隠れ層での学習信号 + α * 前回のバイアス項[隠れ層]
    bias_h_prev[i]=Eta*dwih[i]+Alpha*bias_h_prev[i];
    // 出力層バイアス項[隠れ層] = 出力層バイアス項[隠れ層] + 前回の出力層バイアス項[隠れ層]
    bias_h[i]+=bias_h_prev[i];
  }

  delete dwih;
  delete dwho;

}

void  nnet::setInData(const int pnum,const int i,const double value)
{
  if(pnum>=patternnum || i>=inputnum){
    cout << "can't set Indata." << endl;
    exit(1);
  }
  X_i[pnum][i] = value;
}
  
void nnet::setTeachData(const int pnum,const int i,const double value)
{
  if(pnum>=patternnum || i>=outputnum){
    cout << "can't set Indata." << endl;
    exit(1);
  }
  T_signal[pnum][i] = value;
}

void nnet::train()
{
  double verror;
  int gen;
  int i,j,ip;

  gen = 0;
  verror = 20.0;
  while((verror > ErrorEv) && (gen < MaxGen))
  {
    gen++;
    
    // foward and back propagation
    for(i=0;i<patternnum;i++){
      ip = patternnum * random();
      foward_propagation(ip);
      back_propagation(ip);
    }

    // error calc
    verror = 0;
    for(i=0;i<patternnum;i++){
      foward_propagation(i);
      for(j=0;j<outputnum;j++){
          verror+=pow((T_signal[i][j] - X_o[j]) * 0.5 ,2.0);
      }
    }
    verror /= (double)patternnum;

    cout << verror << endl;

  }
}

void nnet::setPredictData(const int i,const double value)
{
  if(i>=inputnum){
    cout << "can't set Predictdata." << endl;
    exit(1);
  }
  X_i[patternnum-1][i] = value;  
}

void nnet::predict()
{
  int i;
  foward_propagation(patternnum-1);
  for(i=0;i<outputnum;i++){
    cout << X_o[i] << endl;
  }
}

double nnet::random() 
{
	double r;
	int i;

	i = rand();
	if (i != 0)
		i--;
	r = (double)i / RAND_MAX;
	return (r);
}

int main()
{
  nnet net(2,2,1,4);

  net.setInData(0,0,0);
  net.setInData(0,1,0);
  net.setTeachData(0,0,0);

  net.setInData(1,0,1);
  net.setInData(1,1,1);
  net.setTeachData(1,0,0);

  net.setInData(2,0,1);
  net.setInData(2,1,0);
  net.setTeachData(2,0,1);

  net.setInData(3,0,0);
  net.setInData(3,1,1);
  net.setTeachData(3,0,1);

  net.train();

  cout << "--  Predict 00 --" << endl;
  net.setPredictData(0,0);
  net.setPredictData(1,0);
  net.predict();

  cout << "--  Predict 01 --" << endl;
  net.setPredictData(0,0);
  net.setPredictData(1,1);
  net.predict();

  cout << "--  Predict 10 --" << endl;
  net.setPredictData(0,1);
  net.setPredictData(1,0);
  net.predict();

  cout << "--  Predict 11 --" << endl;
  net.setPredictData(0,1);
  net.setPredictData(1,1);
  net.predict();

}
