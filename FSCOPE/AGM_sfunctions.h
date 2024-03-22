//---------------------------------------------------------------------------

#ifndef specialfH
#define specialfH

//---------------------------------------------------------------------------
//some basic real functions
double XExp(double X);
double SinH(double X);
double CosH(double X);
double TanH(double X);
double CotH(double X);

//---------------------------------------------------------------------------

//real functions
double BesselJ0(double x);
double BesselY0(double x); //x>0
double BesselJ1(double x);
double BesselY1(double x); //x>0
double BesselJn(double x,int n);
double BesselYn(double x,int n); //x>0

int Bessel(const double x, const double xnu, double &rj, double &ry, double &rjp, double &ryp);

double Digamma(double x);   //psi function (see also CPSI)
double Polygamma(double x,int n); //derivatives of psi/digamma

double GammaLn(double x);  //real log of Gamma function
double Zeta(double x);  //real Zeta function

double Factorial(int n); //n!
double Bernoulli(int n); //Bernoulli numbers

//advanced complex functions
void CPSI(double X, double Y, double &PSR, double &PSI);
void CGAMA(double X, double Y, int KF, double &GR, double &GI);

void dilog(double re,double im,double &R,double &I);



#endif
