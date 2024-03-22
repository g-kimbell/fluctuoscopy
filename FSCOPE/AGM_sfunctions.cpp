//---------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include "AGM_const.h"
#include "AGM_core.h"
#include "AGM_sfunctions.h"

#pragma hdrstop
//---------------------------------------------------------------------------
//#pragma package(smart_init)

static double M_PI = 3.14159265358979323846;

//coefficients for exp
static double  XA[] =  {2.7182818284590452354,
						1.648721270700128,
						1.284025416687742,
						1.133148453066826,
						1.064494458917859,
						1.031743407499103,
						1.015747708586686,
						1.007843097206448,
						1.003913889338348,
						1.001955033591003};
//coefficients for Gamma
static double  XG[] =  {0,
						8.333333333333333e-02,
					   -2.777777777777778e-03,
						7.936507936507937e-04,
					   -5.952380952380952e-04,
						8.417508417508418e-04,
					   -1.917526917526918e-03,
						6.410256410256410e-03,
					   -2.955065359477124e-02,
						1.796443723688307e-01,
					   -1.39243221690590};
//and for real log(gamma)
static double XGL[]={   76.18009172947146,
					   -86.50532032941677,
						24.01409824083091,
					   -1.231739572450155,
						0.1208650973866179e-2,
					   -0.5395239384953e-5};
//coefficients for Psi
static double  XP[] =  {0,
					   -0.83333333333333333e-01,
						0.83333333333333333e-02,
					   -0.39682539682539683e-02,
						0.41666666666666667e-02,
					   -0.75757575757575758e-02,
						0.21092796092796093e-01,
					   -0.83333333333333333e-01,
						0.4432598039215686};
//coefficients for Bernoulli number
static double XBN[]={
 20.718442527395005639466798045864127459653576742641,    //4 pi exp(1/2)
 2.837877066409345483560659472811235287590943673502,     //log(2 pi e)
 0.083333333333333333333333333333333333333333333333,     //1/12
 0.00347222222222222222222222222222222222222222222,      //1/288
-0.00268132716049382716049382716049382716049382716,      //-139/51840
-0.000229472093621399176954732510288065843621399177,     //-571/2488320
 0.000784                                                //49/(2*31250)
};
//the "first" 32 non-zero Bernoulli numbers (B_2 to B_64)  [B_0 & B_1 are not included]
static double  B32[] =  {
 0.16666666666666666666666666666667,
-0.033333333333333333333333333333333,
 0.023809523809523809523809523809524,
-0.033333333333333333333333333333333,
 0.075757575757575757575757575757576,
-0.25311355311355311355311355311355,
 1.1666666666666666666666666666667,
-7.0921568627450980392156862745098,
 54.971177944862155388471177944862,
-529.12424242424242424242424242424,
 6192.1231884057971014492753623188,
-86580.253113553113553113553113553,
 1.4255171666666666666666666666667e6,
-2.7298231067816091954022988505747e7,
 6.0158087390064236838430386817484e8,
-1.5116315767092156862745098039216e10,
 4.2961464306116666666666666666667e11,
-1.3711655205088332772159087948562e13,
 4.8833231897359316666666666666667e14,
-1.9296579341940068148632668144863e16,
 8.4169304757368261500055370985604e17,
-4.0338071854059455413076811594203e19,
 2.1150748638081991605601453900709e21,
-1.2086626522296525934602731193708e23,
 7.5008667460769643668557200757576e24,
-5.0387781014810689141378930305220e26,
 3.6528776484818123335110430842971e28,
-2.8498769302450882226269146432911e30,
 2.3865427499683627644645981919219e32,
-2.1399949257225333665810744765191e34,
 2.0500975723478097569921733095672e36,
-2.0938005911346378409095185290028e38
};
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

// Modified cordic exponential subroutine
double XExp(double X) {
// This subroutine takes an input value and returns Y=EXP(X)
// X may be any positive or negative real value.
  double Y,aa,Z,E;
  static double XW[10];
  int I,K;
  E=XA[0];

// Reduce the range of X
  K=(int) floor(X);
  X=X-K;
// Determine the weighting coeffs, XW(I)
  aa=0.5;
  Z=X;
  for (I=1; I<10; I++) {
	XW[I]=0.0;
	if (Z>aa)  XW[I]=1.0;
	Z -= XW[I]*aa;
	aa /= 2.0;
  }
// Calculate products
  Y=1.0;
  for (I=1; I<10; I++)
	if (XW[I]>0.0)  Y *= XA[I];
// Perform residual multiplication
  Y *= (1.0+Z*(1.0+Z/2.0*(1.0+Z/3.0*(1.0+Z/4.0))));
// Account for factor EXP(K)
  if (K<0)  E=1.0/E;
  if (ABS(K)<1) return Y;
  for (I=1; I<=ABS(K); I++) Y *= E;
// Restore X
  X += K;
  return Y;
}

/*--------------------------------------------*
*          Hyperbolic sine Function           *
* ------------------------------------------- *
* This Procedure uses the definition of the   *
* hyperbolic sine and the modified cordic     *
* exponential Function XExp(X) to approximate *
* SINH(X) over the entire range of real X.    *
* -------------------------------------------*/
double SinH(double X) {
  double Y,Z;
  int I;
// Is X small enough to cause round off error?
  if (fabs(X)<0.35) goto e10;
// Calculate SinH(X) using exponential definition
// Get Y=exp(X)
  Y=XExp(X);
  Y=0.5*(Y-(1.0/Y));
  return Y;
e10:; //series approximation (for X small)
  Z=1.0; Y=1.0;
  for (I=1; I<9; I++) {
    Z *= X*X/((2*I)*(2*I+1));
    Y += Z;
  }
  Y *= X;
  return Y;
}

// hyperbolic cosine Function
double CosH(double X) {
  double Y;
  Y=XExp(X);
  Y=0.5*(Y+(1.0/Y));
  return Y;
}

// *** End utility functions for TANH ***

// hyperbolic tangent Function
// TanH(X]=SinH(X)/CosH(X)
double TanH(double X) {
  double Y;
  Y=XExp(X);
  Y=Y*Y;
  Y=(Y-1.0)/(Y+1.0);
  return Y;
}

// hyperbolic cot Function
double CotH(double X) {
    double Y;
    Y=XExp(X);
    Y=Y*Y;
    Y=(Y+1.0)/(Y-1.0);
    return Y;
}

//---------------------------------------------------------------------------

double Sqr(double x) {return x*x;}


void CGAMA(double X, double Y, int KF, double &GR, double &GI) {
/*===========================================================
		Purpose: Compute the gamma function G(z) or Ln[G(z)]
                 for a complex argument
        Input :  x  --- Real part of z
                 y  --- Imaginary part of z
                 KF --- Function code
                        KF=0 for Ln[G(z)]
                        KF=1 for G(z)
        Output:  GR --- Real part of Ln[G(z)] or G(z)
                 GI --- Imaginary part of Ln[G(z)] or G(z)
  ===========================================================*/
	double G0,GR1,GI1,SR,SI,T,TH,TH1,TH2,X0,X1,Y1,Z1,Z2;
    int J,K,NA;

    if (Y == 0.0 && X == floor(X) && X <= 0.0) {
	  GR=1e+300;  //arbitrary big number
	  GI=0.0;
      return;
    }
    else if (X < 0.0) {
      X1=X;
	  Y1=Y;
      X=-X;
      Y=-Y;
    }
    X0=X;
    if (X <= 7.0) {
      NA=(int) floor(7.0-X);
      X0=X + 1.0*NA;
    }
    Z1=sqrt(X0*X0+Y*Y);
    TH=atan(Y/X0);
	GR=(X0-0.5)*log(Z1)-TH*Y-X0+0.5*log(2.0*PI);
	GI=TH*(X0-0.5)+Y*log(Z1)-Y;
    for (K=1; K<=10; K++) {
      T=pow(Z1,(1-2*K));
	  GR=GR+XG[K]*T*cos((2.0*K-1.0)*TH);
	  GI=GI-XG[K]*T*sin((2.0*K-1.0)*TH);
    }
    if (X <= 7.0) {
      GR1=0.0;
      GI1=0.0;
      for (J=0; J<NA; J++) {
        GR1 += 0.5*log((X+J)*(X+J)+Y*Y);
        GI1 += atan(Y/(X+J));
      }
	  GR = GR - GR1;
	  GI = GI - GI1;
    }
    if (X1 < 0.0) {
      Z1=sqrt(X*X+Y*Y);
      TH1=atan(Y/X);
	  SR=-sin(PI*X)*CosH(PI*Y);
	  SI=-cos(PI*X)*SinH(PI*Y);
      Z2=sqrt(SR*SR+SI*SI);
      TH2=atan(SI/SR);
      if (SR < 0.0)  TH2 += PI;
	  GR=log(PI/(Z1*Z2))-GR;
	  GI=-TH1-TH2-GI;
      X=X1;
      Y=Y1;
    }
    if (KF == 1) {
	  G0=exp(GR);
	  GR=G0*cos(GI);
	  GI=G0*sin(GI);
    }
}
//---------------------------------------------------------------------------
/********************************************************************
*         Calculate the Psi function for a complex argument         *
* ----------------------------------------------------------------- *
* EXPLANATION:                                                      *
*                                                                   *
*      Purpose: This program computes the psi function psi(z)       *
*               for a complex argument using subroutine CPSI        *
*      Input :  x   --- Real part of z                              *
*               y   --- Imaginary part of z                         *
*      Output:  PSR --- Real part of psi(z)                         *
*               PSI --- Imaginary part of psi(z)                    *
*      Examples:                                                    *
*                  x       y      Re[psi(z)]     Im[psi(z)]         *
*                -------------------------------------------        *
*                 3.0     2.0     1.16459152      .67080728         *
*                 3.0    -2.0     1.16459152     -.67080728         *
*                -3.0     2.0     1.39536075     2.62465344         *
*                -3.0    -2.0     1.39536075    -2.62465344         *
*                                                                   *
* NOTE:          Psi(z) = Gamma'(z) / Gamma(z)                      *
* ----------------------------------------------------------------- *
* SAMPLE RUN:                                                       *
*                                                                   *
*  Please enter x and y (z=x+iy): 3 2                               *
*                                                                   *
*     x       y      Re[Psi(z)]      Im[Psi(z)]                     *
*   ----------------------------------------------                  *
*    3.0     2.0     1.16459152      0.67080728                     *
*                                                                   *
* ----------------------------------------------------------------- *
* REFERENCE:"Fortran Routines for Computation of Special Functions, *
*            jin.ece.uiuc.edu/routines/routines.html".              *
*                                                                   *
*                                 C++ Release By J-P Moreau, Paris. *
********************************************************************/
void CPSI(double X, double Y, double &PSR, double &PSI)
{
		double CT2,RI,RR,TH,TM,TN,X0,X1,Y1,Z0,Z2;
		int    K,N;
		X1=X;Y1=Y;

		if (Y == 0.0 && X == int(X) && X <= 0.0) {
		   PSR=1e+200;
		   PSI=0.0;
		}
		else {
		   if (X < 0.0) {
			 X=-X;
			 Y=-Y;
		   }
		   X0=X;
		   if (X < 8.0) {
			 N=8-int(X);
			 X0=X+1.0*N;
		   }
		   if (X0 == 0.0 && Y != 0.0)  TH=0.5*PI;
		   if (X0 != 0.0)  TH=atan(Y/X0);
		   Z2=X0*X0+Y*Y;
		   Z0=sqrt(Z2);
		   PSR=log(Z0)-0.5*X0/Z2;
		   PSI=TH+0.5*Y/Z2;
		   for (K=1; K<9; K++) {
			 PSR=PSR+XP[K]*pow(Z2,-K)*cos(2.0*K*TH);
			 PSI=PSI-XP[K]*pow(Z2,-K)*sin(2.0*K*TH);
		   }
		   if (X < 8.0) {
			  RR=0.0;
			  RI=0.0;
			  for (K=1; K<=N; K++) {
				RR=RR+(X0-K)/(Sqr(X0-K)+Y*Y);
				RI=RI+Y/(Sqr(X0-K)+Y*Y);
			  }
			  PSR=PSR-RR;
			  PSI=PSI+RI;
		   }
		   if (X1 < 0.0) {
			  TN=sin(PI*X)/cos(PI*X);
			  TM=TanH(PI*Y);
			  CT2=TN*TN+TM*TM;
			  PSR=PSR+X/(X*X+Y*Y)+PI*(TN-TN*TM*TM)/CT2;
			  PSI=PSI-Y/(X*X+Y*Y)-PI*TM*(1.0+TN*TN)/CT2;
			  X=X1;
			  Y=Y1;
		   }
		}
} //CPSI()

//---------------------------------------------------------------------------
double DLOG_I_Re(double t,double re,double im)
{
   if(t<1E-10) return re;
   return -0.5*log((1-t*re)*(1-t*re)+t*im*t*im)/t;
}

double DLOG_I_Im(double t,double re,double im)
{
	if(t<1E-10) return im;
	return -atan2(-t*im,1-t*re)/t;
}

void dilog(double re,double im,double &R,double &I)
{
   int N,i;
   double dt,t,xo,yo,x,y;
   N=100; //support points
   dt=1.0/N;
   R=0.0;
   I=0.0;
   t=0.0;
   xo=DLOG_I_Re(t,re,im);
   yo=DLOG_I_Im(t,re,im);
   for(i=0;i<N;i++)
	{
		t=t+dt;
		x=DLOG_I_Re(t,re,im);
	 y=DLOG_I_Im(t,re,im);
		R=R+(xo+x+4*DLOG_I_Re(t-0.5*dt,re,im));
		I=I+(yo+y+4*DLOG_I_Im(t-0.5*dt,re,im));
		xo=x;
		yo=y;
	}
   R=R*dt/6;
   I=I*dt/6;
}
//---------------------------------------------------------------------------
double BesselJ0(double x)
{
 double ax,z,xx,y,ans,ans1,ans2;

	if ((ax=fabs(x)) < 8.0) {
		y=x*x;
		ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
			+y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
		ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
			+y*(59272.64853+y*(267.8532712+y*1.0))));
		ans=ans1/ans2;
	} else {
		z=8.0/ax;
		y=z*z;
		xx=ax-0.785398164;
		ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
			+y*(-0.2073370639e-5+y*0.2093887211e-6)));
		ans2 = -0.1562499995e-1+y*(0.1430488765e-3
			+y*(-0.6911147651e-5+y*(0.7621095161e-6
			-y*0.934945152e-7)));
		ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
	}
	return ans;
}

double BesselY0(double x)
{
	double z,xx,y,ans,ans1,ans2;

	if (x < 8.0) {
		y=x*x;
		ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
			+y*(10879881.29+y*(-86327.92757+y*228.4622733))));
		ans2=40076544269.0+y*(745249964.8+y*(7189466.438
			+y*(47447.26470+y*(226.1030244+y*1.0))));
		ans=(ans1/ans2)+0.636619772*BesselJ0(x)*log(x);
	} else {
		z=8.0/x;
		y=z*z;
		xx=x-0.785398164;
		ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
			+y*(-0.2073370639e-5+y*0.2093887211e-6)));
		ans2 = -0.1562499995e-1+y*(0.1430488765e-3
			+y*(-0.6911147651e-5+y*(0.7621095161e-6
			+y*(-0.934945152e-7))));
		ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
	}
	return ans;
}

double BesselJ1(double x)
{
 double ax,z,xx,y,ans,ans1,ans2;

	if ((ax=fabs(x)) < 8.0) {
		y=x*x;
		ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
			+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
		ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
			+y*(99447.43394+y*(376.9991397+y*1.0))));
		ans=ans1/ans2;
	} else {
		z=8.0/ax;
		y=z*z;
		xx=ax-2.356194491;
		ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
			+y*(0.2457520174e-5+y*(-0.240337019e-6))));
		ans2=0.04687499995+y*(-0.2002690873e-3
			+y*(0.8449199096e-5+y*(-0.88228987e-6
			+y*0.105787412e-6)));
		ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
		if (x < 0.0) ans = -ans;
	}
	return ans;
}

double BesselY1(double x)
{
   double z,xx,y,ans,ans1,ans2;

	if (x < 8.0) {
		y=x*x;
		ans1=x*(-0.4900604943e13+y*(0.1275274390e13
			+y*(-0.5153438139e11+y*(0.7349264551e9
			+y*(-0.4237922726e7+y*0.8511937935e4)))));
		ans2=0.2499580570e14+y*(0.4244419664e12
			+y*(0.3733650367e10+y*(0.2245904002e8
			+y*(0.1020426050e6+y*(0.3549632885e3+y)))));
		ans=(ans1/ans2)+0.636619772*(BesselJ1(x)*log(x)-1.0/x);
	} else {
		z=8.0/x;
		y=z*z;
		xx=x-2.356194491;
		ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
			+y*(0.2457520174e-5+y*(-0.240337019e-6))));
		ans2=0.04687499995+y*(-0.2002690873e-3
			+y*(0.8449199096e-5+y*(-0.88228987e-6
			+y*0.105787412e-6)));
		ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
	}
	return ans;
}
double BesselJn(double x,int n)
{
	const double ACC=160.0;
	const int IEXP=DBL_MAX_EXP/2;
	bool jsum;
	int j,k,m;
	double ax,bj,bjm,bjp,dum,sum,tox,ans;


  if(n<0) return -1;
  else if(n==0) return BesselJ0(x);
  else if(n==1) return BesselJ1(x);

	ax=fabs(x);
	if (ax*ax <= 8.0*DBL_MIN) return 0.0;
	else if (ax > n) {
		tox=2.0/ax;
		bjm=BesselJ0(ax);
		bj=BesselJ1(ax);
		for (j=1;j<n;j++) {
			bjp=j*tox*bj-bjm;
			bjm=bj;
			bj=bjp;
		}
		ans=bj;
	} else {
		tox=2.0/ax;
		m=2*((n+int(sqrt(ACC*n)))/2);
		jsum=false;
		bjp=ans=sum=0.0;
		bj=1.0;
		for (j=m;j>0;j--) {
			bjm=j*tox*bj-bjp;
			bjp=bj;
			bj=bjm;
			dum=frexp(bj,&k);
			if (k > IEXP) {
				bj=ldexp(bj,-IEXP);
				bjp=ldexp(bjp,-IEXP);
				ans=ldexp(ans,-IEXP);
				sum=ldexp(sum,-IEXP);
			}
			if (jsum) sum += bj;
			jsum=!jsum;
			if (j == n) ans=bjp;
		}
		sum=2.0*sum-bj;
		ans /= sum;
	}
	return x < 0.0 && (n & 1) ? -ans : ans;
}

double BesselYn(double x,int n)
{
  int j;
  double by,bym,byp,tox;

  if(n<0) return -1;
  else if(n==0) return BesselY0(x);
  else if(n==1) return BesselY1(x);

	tox=2.0/x;
	by=BesselY1(x);
	bym=BesselY0(x);
	for (j=1;j<n;j++) {
		byp=j*tox*by-bym;
		bym=by;
		by=byp;
	}
	return by;
}
//---------------------------------------------------------------------------
double chebev(const double a, const double b,const double *c, const int m, const double x)
{
	double d=0.0,dd=0.0,sv,y,y2;
	int j;

	if ((x-a)*(x-b) > 0.0)
		return 0.0; //error
	y2=2.0*(y=(2.0*x-a-b)/(b-a));
	for (j=m-1;j>0;j--) {
		sv=d;
		d=y2*d-dd+c[j];
		dd=sv;
	}
	return y*d-dd+0.5*c[0];
}

void beschb(const double x, double &gam1, double &gam2, double &gampl, double &gammi)
{
	const int NUSE1=7, NUSE2=8;
	static const double c1_d[7] = {
		-1.142022680371168e0,6.5165112670737e-3,
		3.087090173086e-4,-3.4706269649e-6,6.9437664e-9,
		3.67795e-11,-1.356e-13};
	static const double c2_d[8] = {
		1.843740587300905e0,-7.68528408447867e-2,
		1.2719271366546e-3,-4.9717367042e-6,-3.31261198e-8,
		2.423096e-10,-1.702e-13,-1.49e-15};
	double xx;

	xx=8.0*x*x-1.0;
	gam1=chebev(-1.0,1.0,c1_d,NUSE1,xx);
	gam2=chebev(-1.0,1.0,c2_d,NUSE2,xx);
	gampl= gam2-x*gam1;
	gammi= gam2+x*gam1;
}

int Bessel(const double x, const double xnu, double &rj, double &ry, double &rjp, double &ryp)
{
	const int MAXIT=10000;
	const double EPS=DBL_EPSILON;
	const double FPMIN=DBL_MIN/EPS;
	const double XMIN=2.0;
	double a,b,br,bi,c,cr,ci,d,del,del1,den,di,dlr,dli,dr,e,f,fact,fact2,
		fact3,ff,gam,gam1,gam2,gammi,gampl,h,p,pimu,pimu2,q,r,rjl,
		rjl1,rjmu,rjp1,rjpl,rjtemp,ry1,rymu,rymup,rytemp,sum,sum1,
		temp,w,x2,xi,xi2,xmu,xmu2;
	int i,isign,l,nl;

	if (x <= 0.0 || xnu < 0.0)
		return -1; //"bad arguments in bessjy"
	nl=(x < XMIN ? int(xnu+0.5) : MAX(0,int(xnu-x+1.5)));
	xmu=xnu-nl;
	xmu2=xmu*xmu;
	xi=1.0/x;
	xi2=2.0*xi;
	w=xi2/PI;
	isign=1;
	h=xnu*xi;
	if (h < FPMIN) h=FPMIN;
	b=xi2*xnu;
	d=0.0;
	c=h;
	for (i=0;i<MAXIT;i++) {
		b += xi2;
		d=b-d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=b-1.0/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=c*d;
		h=del*h;
		if (d < 0.0) isign = -isign;
		if (fabs(del-1.0) <= EPS) break;
	}
	if (i >= MAXIT)
		return -2; //"x too large in bessjy; try asymptotic expansion"
	rjl=isign*FPMIN;
	rjpl=h*rjl;
	rjl1=rjl;
	rjp1=rjpl;
	fact=xnu*xi;
	for (l=nl-1;l>=0;l--) {
		rjtemp=fact*rjl+rjpl;
		fact -= xi;
		rjpl=fact*rjtemp-rjl;
		rjl=rjtemp;
	}
	if (rjl == 0.0) rjl=EPS;
	f=rjpl/rjl;
	if (x < XMIN) {
		x2=0.5*x;
		pimu=PI*xmu;
		fact = (fabs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (fabs(e) < EPS ? 1.0 : sinh(e)/e);
		beschb(xmu,gam1,gam2,gampl,gammi);
		ff=2.0/PI*fact*(gam1*cosh(e)+gam2*fact2*d);
		e=exp(e);
		p=e/(gampl*PI);
		q=1.0/(e*PI*gammi);
		pimu2=0.5*pimu;
		fact3 = (fabs(pimu2) < EPS ? 1.0 : sin(pimu2)/pimu2);
		r=PI*pimu2*fact3*fact3;
		c=1.0;
		d = -x2*x2;
		sum=ff+r*q;
		sum1=p;
		for (i=1;i<=MAXIT;i++) {
			ff=(i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*(ff+r*q);
			sum += del;
			del1=c*p-i*del;
			sum1 += del1;
			if (fabs(del) < (1.0+fabs(sum))*EPS) break;
		}
		if (i > MAXIT)
			return -3; //"bessy series failed to converge"
		rymu = -sum;
		ry1 = -sum1*xi2;
		rymup=xmu*xi*rymu-ry1;
		rjmu=w/(rymup-f*rymu);
	} else {
		a=0.25-xmu2;
		p = -0.5*xi;
		q=1.0;
		br=2.0*x;
		bi=2.0;
		fact=a*xi/(p*p+q*q);
		cr=br+q*fact;
		ci=bi+p*fact;
		den=br*br+bi*bi;
		dr=br/den;
		di = -bi/den;
		dlr=cr*dr-ci*di;
		dli=cr*di+ci*dr;
		temp=p*dlr-q*dli;
		q=p*dli+q*dlr;
		p=temp;
		for (i=1;i<MAXIT;i++) {
			a += 2*i;
			bi += 2.0;
			dr=a*dr+br;
			di=a*di+bi;
			if (fabs(dr)+fabs(di) < FPMIN) dr=FPMIN;
			fact=a/(cr*cr+ci*ci);
			cr=br+cr*fact;
			ci=bi-ci*fact;
			if (fabs(cr)+fabs(ci) < FPMIN) cr=FPMIN;
			den=dr*dr+di*di;
			dr /= den;
			di /= -den;
			dlr=cr*dr-ci*di;
			dli=cr*di+ci*dr;
			temp=p*dlr-q*dli;
			q=p*dli+q*dlr;
			p=temp;
			if (fabs(dlr-1.0)+fabs(dli) <= EPS) break;
		}
		if (i >= MAXIT) return -4; //"cf2 failed in bessjy"
		gam=(p-f)/q;
		rjmu=sqrt(w/((p-f)*gam+q));
		rjmu=SIGN(rjmu,rjl);
		rymu=rjmu*gam;
		rymup=rymu*(p+q/gam);
		ry1=xmu*xi*rymu-rymup;
	}
	fact=rjmu/rjl;
	rj=rjl1*fact;
	rjp=rjp1*fact;
	for (i=1;i<=nl;i++) {
		rytemp=(xmu+i)*xi2*ry1-rymu;
		rymu=ry1;
		ry1=rytemp;
	}
	ry=rymu;
	ryp=xnu*xi*rymu-ry1;
	return 0;
}


//---------------------------------------------------------------------------

double GammaLn(double x)  //real log of Gamma function [NR]
{
  int j;
  double y,tmp,ser;
  y=x;
  tmp=x+5.5;
  tmp -= (x+0.5)*log(tmp);
  ser=1.000000000190015;
  for (j=0;j<6;j++) ser += XGL[j]/++y;
  return -tmp+log(2.5066282746310005*ser/x);
}
//---------------------------------------------------------------------------

double Zeta(double x)  //real Zeta function
{
	return 0.0;
}

//---------------------------------------------------------------------------

//helper function for polygamma
//
#define psin_max 100
void dpsifn(double x, int n, int kode, int m, double *ans, int *nz, int *ierr)
{
	int i, j, k, mm, mx, nn, np, nx, fn;
    double arg, den, elim, eps, fln, fx, rln, rxsq,
	r1m4, r1m5, s, slope, t, ta, tk, tol, tols, tss, tst,
	tt, t1, t2, wdtol, xdmln, xdmy, xinc, xln = 0.0 /* -Wall */,
	xm, xmin, xq, yint;
	double trm[23], trmr[psin_max + 1];

    *ierr = 0;
    if (n < 0 || kode < 1 || kode > 2 || m < 1) {
	*ierr = 1;
	return;
    }
    if (x <= 0.) {
	/* use	Abramowitz & Stegun 6.4.7 "Reflection Formula"
	 *	psi(k, x) = (-1)^k psi(k, 1-x)	-  pi^{n+1} (d/dx)^n cot(x)
	 */
	if (x == (long)x) {
	    /* non-positive integer : +Inf or NaN depends on n */
		//for(j=0; j < m; j++) /* k = j + n : */
		//ans[j] = ((j+n) % 2) ? ML_POSINF : ML_NAN;
		*ierr=1;
		return;
	}
	dpsifn(1. - x, n, /*kode = */ 1, m, ans, nz, ierr);
	/* ans[j] == (-1)^(k+1) / gamma(k+1) * psi(k, 1 - x)
	 *	     for j = 0:(m-1) ,	k = n + j
	 */

	/* Cheat for now: only work for	 m = 1, n in {0,1,2,3} : */
	if(m > 1 || n > 3) {/* doesn't happen for digamma() .. pentagamma() */
	    /* not yet implemented */
	    *ierr = 4; return;
	}
	x *= M_PI; /* pi * x */
	if (n == 0)
	    tt = cos(x)/sin(x);
	else if (n == 1)
	    tt = -1/pow(sin(x),2);
	else if (n == 2)
	    tt = 2*cos(x)/pow(sin(x),3);
	else if (n == 3)
	    tt = -2*(2*pow(cos(x),2) + 1)/pow(sin(x),4);
	else /* can not happen! */
		tt = NAN;
	/* end cheat */

	s = (n % 2) ? -1. : 1.;/* s = (-1)^n */
	/* t := pi^(n+1) * d_n(x) / gamma(n+1)	, where
	 *		   d_n(x) := (d/dx)^n cot(x)*/
	t1 = t2 = s = 1.;
	for(k=0, j=k-n; j < m; k++, j++, s = -s) {
	    /* k == n+j , s = (-1)^k */
	    t1 *= M_PI;/* t1 == pi^(k+1) */
	    if(k >= 2)
		t2 *= k;/* t2 == k! == gamma(k+1) */
	    if(j >= 0) /* by cheat above,  tt === d_k(x) */
		ans[j] = s*(ans[j] + t1/t2 * tt);
	}
	if (n == 0 && kode == 2)
	    ans[0] += xln;
	return;
    } /* x <= 0 */

    *nz = 0;
	mm = m;
	nx = -(DBL_MIN_EXP);
	r1m5 = LG2;
	r1m4 = DBL_EPSILON* 0.5;
	wdtol = r1m4; /* 1.11e-16 */

	/* elim = approximate exponential over and underflow limit */

    elim = 2.302 * (nx * r1m5 - 3.0);/* = 700.6174... */
    xln = log(x);
    for(;;) {
	nn = n + mm - 1;
	fn = nn;
	t = (fn + 1) * xln;

	/* overflow and underflow test for small and large x */

	if (fabs(t) > elim) {
	    if (t <= 0.0) {
		*nz = 0;
		*ierr = 2;
		return;
	    }
	}
	else {
	    if (x < wdtol) {
		ans[0] = pow(x, -n-1.0);
		if (mm != 1) {
		    for(k = 1; k < mm ; k++)
			ans[k] = ans[k-1] / x;
		}
		if (n == 0 && kode == 2)
		    ans[0] += xln;
		return;
	    }

		/* compute xmin and the number of terms of the series,  fln+1 */

		rln = r1m5 *DBL_MANT_DIG;  //15.95
		//rln = rln<18.06?rln:18.06;
		fln = rln - 3.0;
		yint = 3.50 + 0.40 * fln;
	    slope = 0.21 + fln * (0.0006038 * fln + 0.008677);
	    xm = yint + slope * fn;
	    mx = (int)xm + 1;
	    xmin = mx;
	    if (n != 0) {
		xm = -2.302 * rln - MIN(0.0, xln);
		arg = xm / n;
		arg = MIN(0.0, arg);
		eps = exp(arg);
		xm = 1.0 - eps;
		if (ABS(arg) < 1.0e-3)
		    xm = -arg;
		fln = x * xm / eps;
		xm = xmin - x;
		if (xm > 7.0 && fln < 15.0)
		    break;
	    }
	    xdmy = x;
	    xdmln = xln;
	    xinc = 0.0;
	    if (x < xmin) {
		nx = (int)x;
		xinc = xmin - nx;
		xdmy = x + xinc;
		xdmln = log(xdmy);
	    }

	    /* generate w(n+mm-1, x) by the asymptotic expansion */

	    t = fn * xdmln;
	    t1 = xdmln + xdmln;
	    t2 = t + xdmln;
		tk = MAX(ABS(t), MAX(ABS(t1), ABS(t2)));
		if (tk <= elim)
		goto L10;
	}
	nz++;
	mm--;
	ans[mm] = 0.;
	if (mm == 0)
	    return;
    }
    nn = (int)fln + 1;
    np = n + 1;
    t1 = (n + 1) * xln;
    t = exp(-t1);
    s = t;
    den = x;
    for(i=1; i <= nn; i++) {
	den += 1.;
	trm[i] = pow(den, (double)-np);
	s += trm[i];
    }
    ans[0] = s;
    if (n == 0 && kode == 2)
	ans[0] = s + xln;

    if (mm != 1) { /* generate higher derivatives, j > n */

	tol = wdtol / 5.0;
	for(j = 1; j < mm; j++) {
	    t /= x;
	    s = t;
	    tols = t * tol;
	    den = x;
	    for(i=1; i <= nn; i++) {
		den += 1.;
		trm[i] /= den;
		s += trm[i];
		if (trm[i] < tols)
		    break;
	    }
	    ans[j] = s;
	}
    }
    return;

  L10:
    tss = exp(-t);
    tt = 0.5 / xdmy;
    t1 = tt;
    tst = wdtol * tt;
    if (nn != 0)
	t1 = tt + 1.0 / fn;
    rxsq = 1.0 / (xdmy * xdmy);
    ta = 0.5 * rxsq;
    t = (fn + 1) * ta;
	s = t * B32[0];
	if (fabs(s) >= tst) {
	tk = 2.0;
	for(k = 4; k <= 22; k++) {
		t = t * ((tk + fn + 1)/(tk + 1.0))*((tk + fn)/(tk + 2.0)) * rxsq;
	    trm[k] = t * B32[k-3];
	    if (fabs(trm[k]) < tst)
		break;
	    s += trm[k];
	    tk += 2.;
	}
    }
    s = (s + t1) * tss;
    if (xinc != 0.0) {

	/* backward recur from xdmy to x */

	nx = (int)xinc;
	np = nn + 1;
	if (nx > psin_max) {
	    *nz = 0;
	    *ierr = 3;
	    return;
	}
	else {
	    if (nn==0)
		goto L20;
	    xm = xinc - 1.0;
	    fx = x + xm;

	    /* this loop should not be changed. fx is accurate when x is small */
	    for(i = 1; i <= nx; i++) {
		trmr[i] = pow(fx, (double)-np);
		s += trmr[i];
		xm -= 1.;
		fx = x + xm;
	    }
	}
    }
    ans[mm-1] = s;
    if (fn == 0)
	goto L30;

    /* generate lower derivatives,  j < n+mm-1 */

    for(j = 2; j <= mm; j++) {
	fn--;
	tss *= xdmy;
	t1 = tt;
	if (fn!=0)
	    t1 = tt + 1.0 / fn;
	t = (fn + 1) * ta;
	s = t * B32[0];
	if (fabs(s) >= tst) {
	    tk = 4 + fn;
	    for(k=4; k <= 22; k++) {
		trm[k] = trm[k] * (fn + 1) / tk;
		if (fabs(trm[k]) < tst)
		    break;
		s += trm[k];
		tk += 2.;
	    }
	}
	s = (s + t1) * tss;
	if (xinc != 0.0) {
	    if (fn == 0)
		goto L20;
	    xm = xinc - 1.0;
	    fx = x + xm;
	    for(i=1 ; i<=nx ; i++) {
		trmr[i] = trmr[i] * fx;
		s += trmr[i];
		xm -= 1.;
		fx = x + xm;
	    }
	}
	ans[mm - j] = s;
	if (fn == 0)
	    goto L30;
    }
    return;

  L20:
    for(i = 1; i <= nx; i++)
	s += 1. / (x + nx - i);

  L30:
    if (kode!=2)
	ans[0] = s - xdmln;
    else if (xdmy != x) {
	xq = xdmy / x;
	ans[0] = s - log(xq);
    }
    return;
} /* dpsifn() */


double Digamma(double x)   //psi function (see also CPSI)
{

  double ans;
  int nz, ierr;

  dpsifn(x, 0, 1, 1, &ans, &nz, &ierr);

    if(ierr != 0) {
	  //error
	  return 0.0;
    }
	return -ans;
}
//---------------------------------------------------------------------------

double Polygamma(double x,int n) //derivatives of psi/digamma
{
     /* n-th derivative of psi(x);  e.g., psigamma(x,0) == digamma(x) */
    double ans;
	int nz, ierr, k;

	if(n > psin_max) {
	 //100 is the max derivative
	 return 0.0;
	}
	dpsifn(x, n, 1, 1, &ans, &nz, &ierr);
	if(ierr != 0) {
	 //error
	 return 0.0;
	}
	/* ans ==  A := (-1)^(n+1) / gamma(n+1) * psi(n, x) */
	ans = -ans; /* = (-1)^(0+1) * gamma(0+1) * A */
	for(k = 1; k <= n; k++)
	  ans *= (-k);/* = (-1)^(k+1) * gamma(k+1) * A */
	return ans;/* = psi(n, x) */
}
//---------------------------------------------------------------------------
double Factorial(int n)  //n!
{

  	static int ntop=4;
	static double faclist[65]={1.0,1.0,2.0,6.0,24.0};
	int j;

	if (n < 0) return 0.0;
	if (n > 64) return exp(GammaLn(n+1.0));
	while (ntop<n) {
		j=ntop++;
		faclist[ntop]=faclist[j]*ntop;
	}
	return faclist[n];
};



double Bernoulli(int n) //Bernoulli numbers
{
   double b,x;
   if(n==0) b=1.0;
   else if(n==1) b=-0.5;
   else if((n<0) || ((n&1)==1)) b=0.0;
   else if(n<=64) b=B32[(n>>1)-1];
   else //n even and > 64, use B_n=2*n!/(2\pi)^n*Zeta(n)
	{
		//b=-n*Zeta(1.0-1.0*n);
		x=1.0*n;
		b=XBN[0]*exp((x + 0.5)*(log(x) -XBN[1]))*(1.0 + (XBN[2] + (XBN[3] + (XBN[3] + (XBN[4] + XBN[5]/x)/x)/x)/x)/x);
		if(((n>>1)&1)==0) b=-b;
	}
   return b;
}

//---------------------------------------------------------------------------

