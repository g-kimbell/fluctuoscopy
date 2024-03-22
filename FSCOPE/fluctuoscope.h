//---------------------------------------------------------------------------
#if !defined(fluctuoscope_H)
#define fluctuoscope_H
//---------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <string>

#include "AGM_const.h"
#include "AGM_sfunctions.h"

using namespace std;


#define c1 0.405284734569351085775517852838911 //4/pi^2
#define c2 -1.96351002602142347944097633299876 //Psi(1/2)
#define c3 0.63661977236758134307553505349006 //2/pi
#define c4 1.2020569031595942853997381615114 //zeta(3)
#define c5 0.202642367284675542887758926419455 //2/pi^2
#define c6 0.693147180559945309417232121458177 //ln(2)
#define c7 0.2807297417834425849120716073954404 //1/(2\gamma_E)

#define c8 1.270362845461478170023744211540578999118 //-ln(2)-Psi(1/2)=-c6-c2

#define c_MTi 15.503138340149910087738157533551 //pi^3/2 (factor 2 is in the integral)
#define c_MT 0.010265982254684335189152783267119  //1/pi^4

#define c_DOS 0.129006137732797956737688210754255 //4/pi^3

#define c_CC   0.00138688196439446972811978351321894  //4/pi^6/3

#define s_NC 0.032251534433199489184422052688564 //1/pi^3

#define s_IV -0.032251534433199489184422052688564 //-1/pi^3


#define c_NMRi 31.006276680299820175476315067101 //pi^3
#define c_NMRi2 12.566370614359172953850573533118 //4pi
#define c_NMR 0.037829191583088762117100356213279  //1/(Zeta[3]*7*\[Pi])


#define hc20 0.69267287375563603674263246549077793763519897 //=pi^2*exp[-gamma_e]/8

#define INT0 10e-10 //integral cutoff at z=0
#define DX 10e-6

#define PX 10e-4 //integrable pole cut-out


class FSCOPE {
  
public:
    FSCOPE(int ver=0x100); //should be set by main
     ~FSCOPE();
    
    int readparams(int argc,char *argv[]);
    void info();
    
    int execute();
    
private:
    //input parameters
    double Tct,Tctp,delta,aniso;
    double tmin,dt;
    double hmin,dh;
    double vmin,dv;
    int Nt,Nh,Nv;
    int St,Sh,Sv; //scale: 0 - linear, 1 - log10, 2 - ln
    int ctype; //calculation type
    double hc2para_s; //for curves parallel to hc2(t) parametrized by x=0..1: (t,h)=s*(x,hc2(x))
    
    int Nk,mval,kmin; //to test k sums
    
    //internal variables
    double t,h,v;
    string exe;
    bool usedelta;
    int version;
    
    
    //convert Tctp <-> delta, using t!!!
    inline void setTctp()  {Tctp=PI/(8*t*delta);};
    inline void setdelta() {delta=PI/(8*t*Tctp);}; //not really needed since algorithms use Tctp only
    
    
    inline double hc2(double x)
    {
        double h1,h2,hm,c,f1,f2,fm;
        int Nbs=32;//<10^-9 error
        if(x>=1.0) return 0.0;
        if(x<1e-6) return hc20;
        //solve log(t)+Psi(1/2+2/pi^2*h/t)-Psi(1/2)=0 -> bisection
        c=log(x)-c2;
        h2=(1-x);f2=c+Digamma(0.5+c5*h2/x);//f2>0
        h1=hc20*h2;f1=c+Digamma(0.5+c5*h1/x);//f1<0
        for(int i=0;i<Nbs;i++)
        {
            hm=0.5*(h1+h2);fm=c+Digamma(0.5+c5*hm/x);
            if(fm<0.0) {h1=hm;f1=fm;}
            else {h2=hm;f2=fm;}
        }
        
        return 0.5*(h1+h2);
    };
    inline double hc2_approx(double x)
    {// approximation for hc2(t)
        
        if(x>=1.0) return 0.0;
        if(x<1e-6) return hc20;
        
        return hc20*cos(0.5*PI*exp(0.87*log(x)));
    };

    inline void set_th_para_hc2(double x) {t=hc2para_s*x;h=hc2para_s*hc2(x);};
    
    inline double applyscale(int sc,double lin)
        {if(sc==1) return exp(LN10*lin);
         if(sc==2) return exp(lin);
         return lin;};
    string scalename(int sc,string arg)
        {if(sc==1) return ("lg("+arg+")");
         if(sc==2) return ("ln("+arg+")");
         return arg;}
    
    
    //all ctype calculations
    
    int hc2curve();
    int calcFC();
    int calcFC_hc2();
    
    int calctunnelIV();
    int calctunnelconduct();
    int calczerobiastunnelconduct();
    int calctunnelconduct_hc2();
    int testtunnelIV_int();
    
    int calcNernstbeta();
    int testNernstbeta_sum();

    int calcNMR();
    int calcNMR_3D();
    int calcNMR_hc2();
    int calcNMR_3D_hc2();
    
    int calcsuscept();

};




#endif // fluctuoscope_H
