/* the FLUCTUOSCOPE, see main for version info and copyright  
 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "stringutils.h"
#include "fileutils.h"
#include "paramfile.h"

#include "AGM_const.h"
#include "AGM_core.h"
#include "AGM_sfunctions.h"


#include "fluctuoscope.h"
#pragma hdrstop


using namespace std;

//---------------------------------------------------------------------------

//coordiantes & weights for Gauss-Legendre 5 point integration
static double GLx[] ={-0.90617984593866399280,
					  -0.53846931010568309104,
					  0,
					  0.53846931010568309104,
					  0.90617984593866399280};
static double GLw[] ={0.23692688505618908751,
					  0.47862867049936646804,
					  0.56888888888888888889,
					  0.47862867049936646804,
					  0.23692688505618908751};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void inline aswap(double *a,int i,int j)
{   //element swap
    double t=a[i];a[i]=a[j];a[j]=t;
};

void ABSsort(double *a,int beg,int end) //end needs to be the last valid index +1
{
    if (end > beg+1) {
		double piv = ABS(a[beg]);
		int l = beg + 1, r = end;
		while (l < r) {
			if (ABS(a[l]) <= piv)
				l++;
			else
				aswap(a,l, --r);
		}
		aswap(a,--l, beg);
		ABSsort(a,beg, l);
		ABSsort(a,r, end);
	}
};



//return a "good" error reduced sum of arbitary arrays, N+(log(N))^2
//first sort by absolute value, then to pairwise summation
//no Kahan
//z needs to have length N is N is even 
double sumEC(double *z,int N)
{
    if(N==1) return z[0];
    ABSsort(z,0,N);
    int i=0,j=0;
    while(i<N-1) {z[j]=z[i]+z[i+1];i+=2;j++;}
    if (i==N-1) {z[j]=z[i];j++;}
    z[j]=0.0; //just in case, need to double check the quicksort...
    return sumEC(z,j);
}


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void inline E_n(int n,double t,double h,double z,double &Re,double &Im)
{
   double re,im;
   CPSI(0.5+c1*h/t*(n+0.5),0.5*z,re,im);
   Re=log(t)+re-c2;
   Im=im;
};

void inline E_n3D(int n,double t,double h,double z,double E3Da,double &Re,double &Im)
{
    double re,im;
    CPSI(0.5+c1*h/t*(n+0.5)+E3Da,0.5*z,re,im);
    Re=log(t)+re-c2;
    Im=im;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//All integrands
//at z~0 we approximate for z~10^-10!
int MCfunc(int n,double t,double h,double z,double &AL,double &MT,double &DOS,double &CC)
{ double enr,eni,en1r,en1i,dr,di,absn,absn1,D,res,imen1,imen2,px,psip,psim;
  if(ABS(z)<INT0)
   {if(z>=0.0) z=INT0;
	else z=-INT0;
   }
  E_n(n,t,h,z,enr,eni);
  E_n(n+1,t,h,z,en1r,en1i);


  //************************ bad ...
  CPSI(0.5+c1*h/t*(n+0.5)+DX,0.5*z,px,psip);
  CPSI(0.5+c1*h/t*(n+0.5)-DX,0.5*z,px,psim);

  imen1=0.25*(psip-psim)/DX; //Im E_n'
  imen2=0.25*(psip+psim-eni-eni)/DX/DX; //Im E_n''

  //************************

  //denominator
  absn=enr*enr+eni*eni;
  absn1=en1r*en1r+en1i*en1i;
  D=SinH(PI*z);
  D=1/(D*D*absn);

  //nominator
  dr=enr-en1r;
  di=eni-en1i;
  res=(dr*dr-di*di)*eni*en1i-dr*di*(eni*en1r+en1i*enr);

  AL=res*D/absn1;
  MT=eni*eni*D;
  DOS=eni*imen1*D;
  CC=0; //eni*imen2*D;

  return 0;
};

//---------------------------------------------------------------------------

//integrate over z: gaussian quadratures
int MCint(int n,double t,double h,double zmax,int zsteps,double &AL,double &MT,double &DOS,double &CC)
{
  double dz,z,dzh,al,mt,dos,cc;
  double s_al,s_mt,s_dos,s_cc;
  int i,j;

  dz=zmax/zsteps;dzh=0.5*dz;
  s_al=0.0;
  s_mt=0.0;
  s_dos=0.0;
  s_cc=0.0;
  for(i=-zsteps;i<zsteps;i++)
   {
	 for(j=0;j<5;j++)
		{
			z=i*dz+dzh*(1.0+GLx[j]);
			MCfunc(n,t,h,z,al,mt,dos,cc);
			s_al=s_al+GLw[j]*al;
			s_mt=s_mt+GLw[j]*mt;
			s_dos=s_dos+GLw[j]*dos;
			//s_cc=s_cc+GLw[j]*cc;
		}
   }
  AL=dzh*s_al;
  MT=2*dzh*s_mt;
  DOS=dzh*s_dos;
  CC=0; //4*dzh*s_cc;
  return 0;
};

//---------------------------------------------------------------------------

//calculate the k-sums
int MCksum(int n,double t,double h,int kmax,double &MT,double &CC)
{
    int k,res,kM,Nz,j;
    double en,en2,en3,x,xn,z,zmax,dz,dzh,Am;
    
    MT=0.0;
    CC=0.0;
    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    xn=c1*(n+0.5)*h/t;
    
    kM=2000-((int) (2*xn));if(kM<2) kM=2; 
    
    
    //now do the integration
    Am=log(h*(n+0.5))-c2-c6;
    zmax=1.0/(2*c1+t*kM/(h*(n+0.5)));
    //printf("n=%d, kM=%d, Am=%le, zmax=%le\n",n,kM,Am,zmax);fflush(stdout);
    Nz=25; //functions are smooth enough and zmax<1.2
    dz=zmax/Nz;dzh=0.5*dz;
    for(k=0;k<Nz;k++)
    {
        for(j=0;j<5;j++)
        {
            z=k*dz+dzh*(1.0+GLx[j]);
            x=1.0/(Am-log(z));
            CC+=GLw[j]*x*z;
            MT+=GLw[j]*x;
        }
    }
    x=-t/(h*(n+0.5));    
    CC=x*x*dzh*CC;
    MT=x*dzh*MT;
    
    if(kM>2) //need summation
    {
        //shift k by one, then sum from 2 to kmax
        Am=log(t)-c2;
        for(k=kM-1;k>=2;k--)
        {
            x=0.5*k+xn;
            en=Am+Digamma(x);
            en2=Polygamma(x,2);
            en3=Polygamma(x,3);
            MT+=en2/en;
            CC+=en3/en;
        }
    } 
    //k=0 term 
    x=0.5+xn;
    en=log(t)-c2+Digamma(x);
    en2=Polygamma(x,2);
    en3=Polygamma(x,3);
    MT=2*MT+en2/en;
    CC=2*CC+en3/en;
    return res;
};


//calculate the k-sums
int MCksum_test(int n,double t,double h,int kmax,double &MT,double &CC)
{
  int k,res,kM;
  double en,en2,en3,x,xn,Am;

  MT=0.0;
  CC=0.0;
  
  res=0;
  if(t<1e-6) {t=1e-6;res=2;}

   xn=c1*(n+0.5)*h/t;
  kM=100000; 

  if(kM>2) //need summation
  {
   //shift k by one, then sum from 2 to kmax
   Am=log(t)-c2;
   for(k=kM-1;k>=2;k--)
   {
	   x=0.5*k+xn;
	   en=Am+Digamma(x);
	   en2=Polygamma(x,2);
	   en3=Polygamma(x,3);
	   MT+=en2/en;
	   CC+=en3/en;
   }
  } 
  //k=0 term 
  x=0.5+xn;
  en=log(t)-c2+Digamma(x);
  en2=Polygamma(x,2);
  en3=Polygamma(x,3);
  MT=2*MT+en2/en;
  CC=2*CC+en3/en;
  return res;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//zero field conductivity where the m-sum is replaced by integrals
int MC_sigma_h0(double t,double Tctau,double Tctauphi,double &sigAL,double &sigMTsum,double &sigMTint,double &sigDOS,double &sigCC)
{
    int res;
    double gphi,My,y;
    double intAL,sumMT,intMT,intDOS,sumDCR;
    double sAL,sMTsum,sMTint,sDOS,sDCR;
    
    
    My=1.0/(t*Tctau);  //upper limit for y-integration (continous Landau levels)
    gphi=0.125*PI/Tctauphi;
    res=0;
    
    sAL=0.0;
    sMTint=0.0;
    sMTsum=0.0;
    sDOS=0.0;
    sDCR=0.0;
    
    
    
    
    
    
    
    return 0;
}
//---------------------------------------------------------------------------

//full finite t & h conductivity: MT contribution is split into sum and integral parts
int MC_sigma(double t,double h,double Tctau,double Tctauphi,double &sigAL,double &sigMTsum,double &sigMTint,double &sigDOS,double &sigCC)
{
    int K,KZ,m,M,res,rs;
    double Z,gphi;
    
    double iAL,sumMT,iMT,iDOS,sumCC,iCC;
    double sAL,sMTsum,sMTint,sDOS,sCC;
    
    sigAL=0.0;
    sigMTint=0.0;
    sigMTsum=0.0;
    sigDOS=0.0;
    sigCC=0.0;
    
    
    
    //lower t & h limits
    if(t<0.0001) return 0;
    if(h<0.0001) return MC_sigma_h0(t,Tctau, Tctauphi,sigAL,sigMTsum,sigMTint,sigDOS,sigCC);
    
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
    K=1000;  //absolute maximum k index, if estimated value is above it gives unreliable results
    
    sAL=0.0;
    sMTint=0.0;
    sMTsum=0.0;
    sDOS=0.0;
    sCC=0.0;
    
    M=(int) (1.0/(t*Tctau)+0.5);
    gphi=0.125*PI/(t*Tctauphi); //needs factor t!!! seems wrong in paper
    res=0;
    for(m=M;m>=0;m--)
    {
        rs=MCksum(m,t,h,K,sumMT,sumCC);if(rs>res) res=rs;
        MCint(m,t,h,Z,KZ,iAL,iMT,iDOS,iCC);
        sAL+=(1.0+m)*iAL;
        sMTint+=c_MTi/(gphi+2*h/t*(m+0.5))*iMT;
        sMTsum+=sumMT;
        sDOS+=iDOS;
        sCC+=(m+0.5)*sumCC; //should it be (m+1) ?????
    }
    sigAL=sAL/PI;
    sigMTint=c_MT*sMTint*h/t;
    sigMTsum=c_MT*sMTsum*h/t;
    sigDOS=c_DOS*sDOS*h/t;
    sigCC=c_CC*sCC*h*h/(t*t);
    
    
    
    /*
     s=0.0;
     m=0.0;
     for(i=N;i>=0;i--)
     {
     y=(1.0+i)*Fint(i,t,h);
     s=s+y;
     if(ABS(y)>ABS(m)) m=y;
     if(i==N) yN=y;
     }
     m=SGN(m);
     s2=0.0;
     M=N;
     while((ABS(yN)>1E-6) && (SGN(yN)==m) && (M<500))
     {
	 M+=25;
	 yN=(1.0+M)*Fint(M,t,h);
     }
     if(M>N)
     {
     for(i=M;i>N;i--) s2=s2+(1.0+i)*Fint(i,t,h);
     }
     return c3*(s+s2); */
    return res;
};


int MC_sigma_m(int m,double t,double h,double Tctauphi,double &sigAL,double &sigMTsum,double &sigMTint,double &sigDOS,double &sigCC)
{
    int K,KZ,res,rs;
    double Z,gphi;
    
    double iAL,sumMT,iMT,iDOS,sumCC,iCC;
    double sAL,sMT,sDOS,sCC;
    
    //lower t & h limits
    if(t<0.001) t=0.001;
    if(h<0.001) h=0.001;
    
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
    K=1000;  //absolute maximum k index, if estimated value is above it gives unreliable results
    
    sAL=0.0;
    sMT=0.0;
    sDOS=0.0;
    sCC=0.0;
    

    gphi=0.125*PI/(t*Tctauphi);
    //printf("# gphi=%.16le\n",gphi);
    res=0;

    rs=MCksum(m,t,h,K,sumMT,sumCC);if(rs>res) res=rs;
    MCint(m,t,h,Z,KZ,iAL,iMT,iDOS,iCC);
    sAL+=(1.0+m)*iAL;
    sMT+=c_MTi/(gphi+2*h/t*(m+0.5))*iMT; //+sumMT
    sDOS+=iDOS;
    sCC+=(m+0.5)*sumCC; //should it be (m+1) ?????
    
    
    
    sigAL=sAL/PI;
    sigMTint=c_MT*sMT*h/t;
    sigMTsum=c_MT*sumMT*h/t;
    sigDOS=c_DOS*sDOS*h/t;
    sigCC=c_CC*sCC*h*h/(t*t);
    
    return res;
};


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//NMR: 1/T1 normalized to Karringa and Gi(2)


//calculate the k-sums k=-inf to inf
int NMRksum(int n,double t,double h,int kmax,double &NMR)
{
    int k,res,kM,Nz,j;
    double en,en2,x,xn,z,zmax,dz,dzh,Am;
    
    NMR=0.0;
    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    xn=c1*(n+0.5)*h/t;
    
    kM=2000-((int) (2*xn));if(kM<2) kM=2;
    
    
    //now do the integration
    Am=log(h*(n+0.5))-c2-c6;
    zmax=1.0/(2*c1+t*kM/(h*(n+0.5)));
    //printf("n=%d, kM=%d, Am=%le, zmax=%le\n",n,kM,Am,zmax);fflush(stdout);
    Nz=25; //functions are smooth enough and zmax<1.2
    dz=zmax/Nz;dzh=0.5*dz;
    for(k=0;k<Nz;k++)
    {
        for(j=0;j<5;j++)
        {
            z=k*dz+dzh*(1.0+GLx[j]);
            x=1.0/(Am-log(z));
            NMR+=GLw[j]*x;
        }
    }
    x=-t/(h*(n+0.5));
    NMR=x*dzh*NMR;
    
    if(kM>2) //need summation
    {
        //shift k by one, then sum from 2 to kmax
        Am=log(t)-c2;
        for(k=kM-1;k>=2;k--)
        {
            x=0.5*k+xn;
            en=Am+Digamma(x);
            en2=Polygamma(x,2);
            NMR+=en2/en;
        }
    }
    //k=0 term
    x=0.5+xn;
    en=log(t)-c2+Digamma(x);
    en2=Polygamma(x,2);
    NMR=2*NMR+en2/en;
    return res;
};


//calculate the k-sums k=-inf to inf
int NMRksum3D(int n,double t,double h,int kmax,double E3Da,double &NMR)
{
    int k,res,kM,Nz,j;
    double en,en2,x,xn,z,zmax,dz,dzh,Am;
    
    NMR=0.0;
    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    xn=c1*(n+0.5)*h/t+E3Da;
    
    kM=2000-((int) (2*xn));if(kM<2) kM=2;
    
    
    //now do the integration
    Am=log(h*(n+0.5))-c2-c6;
    zmax=1.0/(2*c1+t*(kM+2*E3Da)/(h*(n+0.5)));
    //printf("n=%d, kM=%d, Am=%le, zmax=%le\n",n,kM,Am,zmax);fflush(stdout);
    Nz=25; //functions are smooth enough and zmax<1.2
    dz=zmax/Nz;dzh=0.5*dz;
    for(k=0;k<Nz;k++)
    {
        for(j=0;j<5;j++)
        {
            z=k*dz+dzh*(1.0+GLx[j]);
            x=1.0/(Am-log(z));
            NMR+=GLw[j]*x;
        }
    }
    x=-t/(h*(n+0.5));
    NMR=x*dzh*NMR;
    
    if(kM>2) //need summation
    {
        //shift k by one, then sum from 2 to kmax
        Am=log(t)-c2;
        for(k=kM-1;k>=2;k--)
        {
            x=0.5*k+xn;
            en=Am+Digamma(x);
            en2=Polygamma(x,2);
            NMR+=en2/en;
        }
    }
    //k=0 term
    x=0.5+xn;
    en=log(t)-c2+Digamma(x);
    en2=Polygamma(x,2);
    NMR=2*NMR+en2/en;
    return res;
};

//NMR integrands
//at z~0 we approximate for z~10^-10!
inline int NMRfunc(int n,double t,double h,double z,double &NMR,double &NMR2)
{ double enr,eni,absn,D,px,psip,psim;
    if(ABS(z)<INT0)
    {if(z>=0.0) z=INT0;
	else z=-INT0;
    }
    E_n(n,t,h,z,enr,eni);
    
    //************************ bad ...
    CPSI(0.5+c1*h/t*(n+0.5)+DX,0.5*z,px,psip);
    CPSI(0.5+c1*h/t*(n+0.5)-DX,0.5*z,px,psim);
    
    NMR2=0.25*(psip-psim)/DX; //Im E_n'
    //************************
    
    
    //denominator
    absn=enr*enr+eni*eni;
    D=SinH(PI*z);
    D=1/(D*D*absn);
    
    NMR=eni*eni*D;
    NMR2=eni*NMR2*D;
    
    return 0;
};


//NMR integrands 3D
//at z~0 we approximate for z~10^-10!
inline int NMRfunc3D(int n,double t,double h,double z,double E3Da,double &NMR,double &NMR2)
{ double enr,eni,absn,D,px,psip,psim;
    if(ABS(z)<INT0)
    {if(z>=0.0) z=INT0;
	else z=-INT0;
    }
    E_n3D(n,t,h,z,E3Da,enr,eni);
    
    //************************ bad ...
    CPSI(0.5+c1*h/t*(n+0.5)+DX+E3Da,0.5*z,px,psip);
    CPSI(0.5+c1*h/t*(n+0.5)-DX+E3Da,0.5*z,px,psim);
    
    NMR2=0.25*(psip-psim)/DX; //Im E_n'
    //************************
    
    
    //denominator
    absn=enr*enr+eni*eni;
    D=SinH(PI*z);
    D=1/(D*D*absn);
    
    NMR=eni*eni*D;
    NMR2=eni*NMR2*D;
    
    return 0;
};


//integrate over z: gaussian quadratures
int NMRintegral(int n,double t,double h,double zmax,int zsteps,double &NMR,double &NMR2)
{
    double dz,z,dzh,nmr,nmr2;
    double s_nmr,s_nmr2;
    int i,j;
    
    dz=zmax/zsteps;dzh=0.5*dz;

    s_nmr=0.0;
    s_nmr2=0.0;

    for(i=-zsteps;i<zsteps;i++)
    {
        for(j=0;j<5;j++)
		{
			z=i*dz+dzh*(1.0+GLx[j]);
			NMRfunc(n,t,h,z,nmr,nmr2);
			s_nmr=s_nmr+GLw[j]*nmr;
            s_nmr2=s_nmr2+GLw[j]*nmr2;
		}
    }
    NMR=dzh*s_nmr;
    NMR2=dzh*s_nmr2;
    return 0;
};


//integrate over z, 3D: gaussian quadratures
int NMRintegral3D(int n,double t,double h,double zmax,int zsteps,double E3Da,double &NMR,double &NMR2)
{
    double dz,z,dzh,nmr,nmr2;
    double s_nmr,s_nmr2;
    int i,j;
    
    dz=zmax/zsteps;dzh=0.5*dz;
    
    s_nmr=0.0;
    s_nmr2=0.0;
    
    for(i=-zsteps;i<zsteps;i++)
    {
        for(j=0;j<5;j++)
		{
			z=i*dz+dzh*(1.0+GLx[j]);
			NMRfunc3D(n,t,h,z,E3Da,nmr,nmr2);
			s_nmr=s_nmr+GLw[j]*nmr;
            s_nmr2=s_nmr2+GLw[j]*nmr2;
		}
    }
    NMR=dzh*s_nmr;
    NMR2=dzh*s_nmr2;
    return 0;
};

//full NMR
int T1norm(double t,double h,double Tctau,double Tctauphi,double &NMRsum,double &NMRint)
{
    int K,KZ,m,M,res,rs;
    double Z,gphi;
    
    double sNMR,iNMR,iNMR2,summiNMR,summsNMR;
    
    NMRsum=0.0;
    NMRint=0.0;

    //lower t & h limits
    if(t<0.0001) t=0.0001;
    if(h<0.0001) h=0.0001;
    
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
    K=1000;  //absolute maximum k index, if estimated value is above it gives unreliable results
    
    summiNMR=0.0;
    summsNMR=0.0;
    
    M=(int) (1.0/(t*Tctau)+0.5);
    gphi=0.125*PI/(t*Tctauphi);
    res=0;
    for(m=M;m>=0;m--)
    {
        rs=NMRksum(m,t,h,K,sNMR);if(rs>res) res=rs;
        NMRintegral(m,t,h,Z,KZ,iNMR,iNMR2);
        summiNMR+=c_NMRi/(gphi+2*h/t*(m+0.5))*iNMR+c_NMRi2*iNMR2;
        summsNMR+=2*sNMR;
    }
   
    NMRint=c_NMR*summiNMR*h/t;
    NMRsum=c_NMR*summsNMR*h/t;

    return res;
};


//full NMR
int T1norm3D(double t,double h,double Tctau,double Tctauphi,double aniso,double &NMRsum,double &NMRint)
{
    int K,KZ,m,M,res,rs,i,j,Ny;
    double Z,gphi,sincoeff,sin2y,E3Dadd,y,dy,dyh;
    
    double sNMR,iNMR,iNMR2,summiNMR,summsNMR;
    
    NMRsum=0.0;
    NMRint=0.0;
    
    //lower t & h limits
    if(t<0.0001) t=0.0001;
    if(h<0.0001) h=0.0001;
    
    
    sincoeff=2*aniso/t;
    
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
    K=1000;  //absolute maximum k index, if estimated value is above it gives unreliable results
    
    summiNMR=0.0;
    summsNMR=0.0;
    
    M=(int) (1.0/(t*Tctau)+0.5);
    gphi=0.125*PI/(t*Tctauphi);
    res=0;
    
    
    //y-integral 0 to PI/2: Gaussian
    Ny=5;
    dy=0.5*PI/Ny;
    dyh=0.5*dy;
    for(i=0;i<Ny;i++)
    {
        for(j=0;j<5;j++)
		{
			y=i*dy+dyh*(1.0+GLx[j]);
			sin2y=sin(y);sin2y=sin2y*sin2y;
            E3Dadd=2*sincoeff/PI*sin2y;
            for(m=M;m>=0;m--)
            {
                rs=NMRksum3D(m,t,h,K,E3Dadd,sNMR);if(rs>res) res=rs;
                NMRintegral3D(m,t,h,Z,KZ,E3Dadd,iNMR,iNMR2);
                summiNMR+=GLw[j]*(c_NMRi/(gphi+2*h/t*(m+0.5)+PI*sincoeff*sin2y)*iNMR+c_NMRi2*iNMR2);
                summsNMR+=GLw[j]*2*sNMR;
            }
            
		}
    }
    
    y=dyh*2/PI/PI; //factor from y integration
    
    
    NMRint=c_NMR*summiNMR*h/t*y;
    NMRsum=c_NMR*summsNMR*h/t*y;
    
    return res;
};


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------


//calculate the k-sums k=-inf to inf, for fixed n: sums very slowly converging
int chiksum(int n,double t,double h,int kmax,double Tctau,double &sCHI)
{
    int k,res,kM,K;
    double en,en1,en2,x,xn,Am,a,Y,ym,lY,lym;
    
    sCHI=0.0;
    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    
    K=(int) (1.0/(Tctau)+0.5); //=M, cutoff for full k-sum
    
    xn=c1*(n+0.5)*h/t;
    
    kM=kmax-((int) (2*xn));if(kM<5) kM=5; //actual sum/integral separation
    if(kM>K) kM=K;
    
    if(K>kM) { //calculate the integral contribution
        a=c1*(n+0.5)*h;
        Y=t*(K+1)+a+a;
        ym=t*(kM+1)+a+a;
        
        lY=c8+log(Y);lym=c8+log(ym);
        sCHI=a*(1/(Y*lY)-1/(ym*lym))+log(lY/lym); //explicit expression
    }
    else kM=K; //means to calculate the complete sum
    
    Am=log(t)-c2;
    if(kM>2) //need summation
    {
        //shift k by one, then sum from 2 to kmax
        for(k=kM-1;k>=2;k--)
        {
            x=0.5*k+xn;
            en=Am+Digamma(x);
            en1=0.5*Polygamma(x,1);
            en2=0.25*Polygamma(x,2);
            sCHI+=(en1+xn*(en2-en1*en1/en))/en;
        }
    }
    sCHI*=2.0;
    
    //k=0 term
    x=0.5+xn;
    en=Am+Digamma(x);
    en1=0.5*Polygamma(x,1);
    en2=0.25*Polygamma(x,2);
    sCHI+=(en1+xn*(en2-en1*en1/en))/en;
    return res;
};

//magnetic susceptibility (note, this expression diverges with Tctau^-1)
int chi(double t,double h,double Tctau,double &CHI)
{
    int kmax,m,M,res,rs;
    double sCHI;
    
    CHI=0.0;
    
    //lower t & h limits
    if(t<0.0001) t=0.0001;
    if(h<0.0001) h=0.0001;

    
    //sum parameters
    kmax=2000;  //maximum k index for discrete/continuum summation, if estimated value is above it gives unreliable results
    
    M=(int) (1.0/(Tctau)+0.5);
    res=0;
    for(m=M;m>=0;m--)
    {
        rs=chiksum(m,t,h,kmax,Tctau,sCHI);if(rs>res) res=rs;
        CHI+=(m+0.5)*sCHI;
    }

    return res;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//Larkin beta-function

double LarkinG(int m,double t,double Tctp)
{
  return 1/(log(t)-c2+Digamma(0.5*(1+m))-Polygamma(0.5*(1+m),1)/(4*PI*Tctp*t));
};

double LarkinG2p(int m,double t,double Tctp)
{
	double x,y,a,P0,P1,P2,P3;
	x=0.5*(1+m);
	P0=Digamma(x);P1=Polygamma(x,1);P2=Polygamma(x,2);P3=Polygamma(x,3);
	a=1/(4*PI*Tctp*t);
	x=1/(log(t)-c2+P0-a*P1);
	y=P1-a*P2;
  return 0.5*x*x*(y*y*x-0.5*(P2-a*P3));
};

double LarkinMT(double t,double h,double Tct,double Tctp)
{
  double beta=0.0;
  double x,dY;
  int m,M;
  M=100000; //(int) (1.0/(t*Tct)+0.5);
  x=4*c1*h*Tctp;
  dY=log(x)+Digamma(0.5+1/x);
  x=0.5*PI*t*log(t)/h; //inverse argument
  dY+=log(x)-Digamma(0.5+x);
  beta=LarkinG(0,t,Tctp)/c1;
  for(m=M;m>=1;m--)
  {
    beta+=2*(LarkinG(2*m,t,Tctp)-LarkinG(2*m-1,t,Tctp))/c1-LarkinG2p(2*m-1,t,Tctp);
  } 
  return -c1*beta*dY/16;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//Nernst k-sum, depending on Landau level n, temperature t, and magnetic field h
//the k-sum is limited by the argument of psi being 1000, above we integrate with inverse argument
int NCksum(int n,double t,double h,double &NCs,bool test=false)
{
    int k,res,kM,Nz,j;
    double e0,e1,ex,dep0,dep1,f0,f1,x,xn,z,zmax,dz,dzh,alpha;
    
   

    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    alpha=c1*h/t;  
    xn=alpha*(n+0.5);
    kM=3000-((int) (2*xn));if(kM<10) kM=10; 
    
    
    //now do the integration
    
    zmax=1.0/(0.5*(1.0+kM)+xn);
    
    Nz=25; //function is smooth enough and zmax<<1
    dz=zmax/Nz;dzh=0.5*dz; 
    ex=log(t)-c2;
    
    if(test) printf("#sum: alpha=%le, kM=%d, zmax=%le, ",alpha,kM,zmax);
    
    NCs=0.0;
    for(k=0;k<Nz;k++)
    {
        for(j=0;j<5;j++)
	    {
            z=k*dz+dzh*(1.0+GLx[j]);
            x=1/(1+alpha*z);
            e0=1/(ex-log(z));e1=1/(ex+log(1/z+alpha));
            x=x*(e0-x*e1);
            NCs+=GLw[j]*x*z;
        }
    }    
    NCs=alpha*dzh*NCs;
  
    if(test) printf("int-rest=%le, ",NCs);
    
    if(kM>1) //need summation
    {
        for(k=kM-1;k>=1;k--)
        {
            x=0.5*(1.0+k)+xn;
            dep0=(Polygamma(x, 1)-Polygamma(x+alpha, 1))/alpha;
            e0=1/(ex+Digamma(x));
            e1=1/(ex+Digamma(x+alpha));
            dep1=dep0*e1;
            dep0=dep0*e0;
            f0=e0*Polygamma(x,2);
            f1=e1*Polygamma(x+alpha,2);
            NCs+=(2*xn+k)*(dep0+dep1+f0+f1)+2*alpha*(f1+dep0);
        }
    } 
    
    if(test) printf("+sum=%le, ",NCs);
    
    //k=0 term 
    x=0.5+xn;
    dep0=(Polygamma(x, 1)-Polygamma(x+alpha, 1))/alpha;
    e0=1/(ex+Digamma(x));
    e1=1/(ex+Digamma(x+alpha));
    dep1=dep0*e1;
    dep0=dep0*e0;
    f0=e0*Polygamma(x,2);
    f1=e1*Polygamma(x+alpha,2);
    NCs+=xn*(dep0+dep1+f0+f1)+alpha*(f1+dep0);
    
    if(test) printf("+k0=%le\n",NCs);
    
    return res;
};

//NC Pm function for integration, including the sinh term
//at z~0 we approximate for z~10^-10!
int inline NC_Pm(int n,double t,double h,double z,double &NC)
{ double enr,eni,en1r,en1i,D,res,ienp,ien1p,px,psip,psim;
    double alpha,X0,X1,Y0,Y1,Zp,Zm,Wr,Wi;
    if(ABS(z)<INT0)
     {if(z>=0.0) z=INT0;
	  else z=-INT0;
     }
    
    E_n(n,t,h,z,enr,eni);
    E_n(n+1,t,h,z,en1r,en1i);
    
    alpha=c1*h/t;
    
    //************************ not so good ...
    CPSI(0.5+alpha*(n+0.5)+DX,0.5*z,px,psip);
    CPSI(0.5+alpha*(n+0.5)-DX,0.5*z,px,psim);
    
    ienp=0.25*(psip-psim)/DX; //Im E_n'
    
    CPSI(0.5+alpha*(n+1.5)+DX,0.5*z,px,psip);
    CPSI(0.5+alpha*(n+1.5)-DX,0.5*z,px,psim);
    
    ien1p=0.25*(psip-psim)/DX; //Im E_(n+1)'
    //************************
    
    
    X0=(n+n+1)*alpha*eni+z*enr;
    X1=(n+n+3)*alpha*en1i+z*en1r;
    Zp=eni+en1i;
    Zm=0.5*(en1i-eni);
    
    //denominator
    Y0=enr*enr+eni*eni;
    Y1=en1r*en1r+en1i*en1i;
    D=SinH(PI*z);
    D=1/(D*D);
    
    //nominator
    Wr=(eni*en1i+enr*en1r);
    Wi=(eni*en1r-enr*en1i);
    res=alpha*(X0*Y1*ienp+X1*Y0*ien1p+Zp*(eni*Y1+en1i*Y0))+(Y1*X0+Y0*X1)*Zm-Zp*Wr*(X1-X0);
    px=2*z*atan2(Wi,Wr);
    
    NC=D*(res/(Y0*Y1)+px);
    
    return 0;
};

//Nernst z-integral
int NCzint(int n,double t,double h,double zmax,int zsteps,double &NCi)
{
    double dz,z,dzh,nc;
    double s_nc;
    int i,j;
    
    dz=zmax/zsteps;dzh=0.5*dz;
    s_nc=0.0;
    for(i=-zsteps;i<zsteps;i++)
    {
        for(j=0;j<5;j++)
		{
			z=i*dz+dzh*(1.0+GLx[j]);
			NC_Pm(n,t,h,z,nc);
			s_nc=s_nc+GLw[j]*nc;
		}
    }
    NCi=dzh*s_nc;
    return 0;
};

int NC_beta_xy(double t,double h,double Tctau,double &NCs,double &NCi,bool test=false)
{
    int KZ,m,M,res,rs;
    double Z;
    
    double iNC,sNC;
    
    //lower t & h limits
    if(t<0.001) t=0.001;
    if(h<0.001) h=0.001;
    
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
    
    NCi=0.0;
    NCs=0.0;
  
    M=(int) (1.0/(t*Tctau)+0.5);
    res=0;
    for(m=M;m>=0;m--)
    {
        rs=NCksum(m,t,h,sNC,test);if(rs>res) res=rs;
        NCzint(m,t,h,Z,KZ,iNC);
        NCs+=(1.0+m)*sNC;
        NCi+=(1.0+m)*iNC;
        if(test) printf("%d\t%.16le\t%.16le\n",m,sNC,iNC);
    }
    NCs=NCs*s_NC*h/t;
    NCi=0.5*NCi;
    
    
    return res;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// Tunneling IV
//calculate the k-sums
int IVksum(int m,double t,double h,double v,double &IVs)
{
 int k,res,kM,Nz,j;
    double en,x,xn,z,zmax,dz,dzh,Am,vm,vt;
    double px,psip,psim,imtri;
    
    IVs=0.0;
    
    res=0;
    if(t<1e-6) {t=1e-6;res=2;}
    
    xn=c1*(m+0.5)*h/t;
    vt=-c7*v/t;
    vm=-c7*v/(h*(m+0.5));
    
    kM=2000-((int) (2*xn));if(kM<2) kM=2; 
    
    
    //now do the integration
    Am=log(h*(m+0.5))-c2-c6;
    zmax=1.0/(2*c1+t*kM/(h*(m+0.5)));
    //printf("n=%d, kM=%d, Am=%le, zmax=%le\n",n,kM,Am,zmax);fflush(stdout);
    Nz=25; //functions are smooth enough and zmax<1.2
    dz=zmax/Nz;dzh=0.5*dz;
    for(k=0;k<Nz;k++)
    {
        for(j=0;j<5;j++)
        {
            z=k*dz+dzh*(1.0+GLx[j]);
            x=1.0/((Am-log(z))*(1+vm*vm*z*z));
            IVs+=GLw[j]*x;
        }
    }
    IVs=-2*vm*dzh*IVs;

    
    if(kM>2) //need summation
    {
        //shift k by one, then sum from 1 to kM
        Am=log(t)-c2;
        for(k=kM;k>=1;k--)
        {
            x=0.5*k+xn;
            en=Am+Digamma(x);
            
            //************************ not so good ...
            CPSI(x+DX,vt,px,psip);
            CPSI(x-DX,vt,px,psim);
            imtri=0.5*(psip-psim)/DX; //Im \psi'
            //************************
            if(k==1) imtri*=0.5; //k=0 term has only weight 1/2
            
            IVs+=imtri/en;
        }
    } 
    
    return res;
};


//integrand, with singularities cut out or avoided
int inline IVintegrand(int n,double t,double h,double v,double z,double &IV,char poles=3)
{ 
    double enr,eni,ienp,ienvp,renp,renvp,psrp,psrm,psip,psim,im;
    double alpha,Rm,D,s;
    
    
    E_n(n,t,h,-z,enr,eni);
    
    
    alpha=0.5+c1*h/t*(n+0.5);
    s=0.25/DX;
    
    //************************ not so good ...
    im=0.5*(z+v);
    CPSI(alpha+DX,im,psrp,psip);
    CPSI(alpha-DX,im,psrm,psim);
    
    renvp=s*(psrp-psrm); //Re E_n'(z+v)
    ienvp=s*(psip-psim); //Im E_n'(z+v)
    
    im=-0.5*z;
    CPSI(alpha+DX,im,psrp,psip);
    CPSI(alpha-DX,im,psrm,psim);
    
    renp=s*(psrp-psrm); //Im E_n'(-z)
    ienp=s*(psip-psim); //Im E_n'(-z)
    //************************
    
    Rm=enr*(renvp-renp)+eni*(ienvp-ienp);
    
    D=(enr*enr+eni*eni);
    if(poles&1) D*=SinH(PI*z);
    if(poles&2) D*=SinH(PI*(z-0.5*v));
    
    if(ABS(D)<1e-8) {if(D>=0.0) D=1e-8;else D=-1e-8;}
    
    IV=Rm/D;
    
    return 0;
};


//requires zp>zm, Nz>0
double inline IVintpart(int n,double t,double h,double v,double zm,double zp,int Nz,double *ivals,int &validx,bool inv=false)
{
    double g5,res,dz,dzh,y,z;
    int i,j;

    res=0.0;
    dz=(zp-zm)/(1.0*Nz);dzh=0.5*dz;
    if(inv) {zm=zp-dz;dz=-dz;}; //reverse the summation, optional for decreasing functions
    for(i=0;i<Nz;i++)
    {
        g5=0.0;
        for(j=0;j<5;j++)
		{
			z=zm+i*dz+dzh*(1.0+GLx[j]);
			IVintegrand(n,t,h,v,z,y);
			g5=g5+GLw[j]*y;
		}
        g5*=dzh;
        res=res+g5;
        ivals[validx]=g5;validx++;
    }
    return res;
};


//tunnel IV z-integral
//here we use equal size dz's -> maybe better to use finer dz near the sigularities
int IVzint(int n,double t,double h,double v,double zmax,int zsteps,double &IVi,double &IVres)
{
    double vh,dp,dpr;
    double int1,int2,int3,intt,pole0,polev,pre;
    double px,psip,psim,alpha;
    double *intvals;
    bool neg;
    int k,zfine,Nval,validx;
    
    dpr=0.1; //region size near pole which needs higher accuracy, zmax>>dpr>>dp
    zfine=25; //integration steps in [dp;dpr] near poles
    
    IVi=0.0;
    //convert v to vt
    v=-c7*v/t;
    
    
    //calculate constant residual term
    alpha=0.5+c1*h/t*(n+0.5);
    CPSI(alpha+DX,0.5*v,px,psip);
    CPSI(alpha-DX,0.5*v,px,psim);
    IVi=0.25*(psip-psim)/(DX*(log(t)-c2+Digamma(alpha)));
    IVres=IVi;
    
    if(ABS(v)<2*DX) return 0; 
    neg=false;
    if(v<0.0) {v=-v;neg=true;} //use the antisymmetric behavior to limit the following to positive v
    
    Nval=4*(zsteps+zfine)+100;
    intvals=new double[Nval];
    validx=0;
    
    vh=0.5*v;
    pre=SinH(PI*vh); //integral prefactor
    
    //calculate the poles
    dp=PX; if(vh<2*dp) dp=0.5*vh;
    
    IVintegrand(n,t,h,v,-DX,psim,2);
    IVintegrand(n,t,h,v,DX,psip,2);
    pole0=dp*(psip-psim)/DX;
    
    IVintegrand(n,t,h,v,vh-DX,psim,1);
    IVintegrand(n,t,h,v,vh+DX,psip,1);
    polev=dp*(psip-psim)/DX;
    
    intvals[0]=pole0;
    intvals[1]=polev;
    validx=2;
    
    //split interval in three parts
    
    //left part 1 (two mesh sizes for [-zmax;-dpr] and [-dpr;dp] given by zsteps and zfine)
    int1=IVintpart(n,t,h,v,-zmax,-dpr,zsteps,intvals,validx); //normal steps away from zero pole, left
    intt=IVintpart(n,t,h,v,-dpr,-dp,zfine,intvals,validx);    //now finer near the zero pole, left
    int1+=intt;
    //right part 3 (two mesh sizes [vh+dpr;vh+zmax] [dp+vh;dpr+vh]), integrated backward
    int3=IVintpart(n,t,h,v,vh+dpr,vh+zmax,zsteps,intvals,validx,true); //normal steps away from pole vh, right
    intt=IVintpart(n,t,h,v,vh+dp,vh+dpr,zfine,intvals,validx,true);   //now finer near the vh pole, right
    int3+=intt;
    //between the poles (up to four parts)
    int2=0.0;
    if(vh>2*dp+DX) //otherwise there is nothing in between
    {
        if(vh>2*dpr+DX) //central larger grid piece
        {
            if(vh>2*zmax+DX) //the center can be skipped
             {
                intt=IVintpart(n,t,h,v,dpr,zmax,zsteps,intvals,validx,true); //normal step size, inverse
                int2+=intt;
                intt=IVintpart(n,t,h,v,vh-zmax,vh-dpr,zsteps,intvals,validx);//normal step size
             }
            else 
             {
                 k=(int) (zsteps*(vh-2*dpr)/zmax+0.5); //steps to ensure normal step size
                 if(k<1) k=1;
                 intt=IVintpart(n,t,h,v,dpr,vh-dpr,k,intvals,validx);//normal step size
             }
            int2+=intt;
            
            intt=IVintpart(n,t,h,v,dp,dpr,zfine,intvals,validx,true);
            int2+=intt;
            intt=IVintpart(n,t,h,v,vh-dpr,vh-dp,zfine,intvals,validx);
        }
        else
        {
            k=(int) (zfine*(vh-2*dp)/dpr+0.5); //steps to ensure fine step size   
            if(k<1) k=1;
            intt=IVintpart(n,t,h,v,dp,vh-dp,k,intvals,validx);//fine step size
        }
        int2+=intt;
    }
    
    
    intt=pre*(int1+int2+int3+pole0+polev);
    
    //try more accurate summation ... gives the same :(
    //intvals[validx]=0.0;
    //intt=pre*sumEC(intvals,validx);
    //printf("#%le    %le\n",intt,int1);
    
    if(neg) IVi=IVi-intt;
    else IVi=IVi+intt;
    
    
    delete[] intvals;
    return 0;
};


int tunnelIV(double t,double h,double v,double Tctau,double &IVs,double &IVi,double &IVi0,bool test=false)
{
    int KZ,m,M,res,rs;
    double x,Z,iIV,iIV0;
    
    res=0;
    IVs=0.0;
    IVi=0.0;
    IVi0=0.0;
    
    //lower t & h limits
    if(t<1e-6) t=1e-6;
    if(h<1e-6) h=1e-6;
    
    //V=0 limit
    if(ABS(v)<1e-6) return res;
    
    //m cutoff
    M=(int) (1.0/(t*Tctau)+0.5);
    
    //sum/integration parameters
    Z=5.0; //this makes the sinh^2(pi Z)=(10^-13)
    KZ=200; //-->dz=0.025
 
 
    for(m=M;m>=0;m--)
    {
        rs=IVksum(m,t,h,v,x);if(rs>res) res=rs;
        IVzint(m,t,h,v,Z,KZ,iIV,iIV0);
        IVs+=x;
        IVi+=iIV;
        IVi0+=iIV0;
        if(test) printf("%d\t%.16le\t%.16le\t%.16le\n",m,x,iIV,iIV0);
    }
    IVs=IVs*s_IV*h;
    IVi=IVi*s_IV*h;
    IVi0=IVi0*s_IV*h;
    
    
    return res;
};

//tunnel conductance at zero voltage: current and anomalous part are zero
int zeroVtunnelconduct(double t,double h,double Tctau,double &tsigma)
{
    int m,M,res,rs;
    double x,y,dx,sp,sm;
    
    res=0;
    tsigma=0.0;
    sp=0.0;
    sm=0.0;
 
    dx=DX;
    
    //lower t & h limits
    if(t<1e-6) t=1e-6;
    if(h<1e-6) h=1e-6;
    
    //m cutoff
    M=(int) (1.0/(t*Tctau)+0.5);
    
    for(m=M;m>=0;m--)
    {
        rs=IVksum(m,t,h,dx,x);if(rs>res) res=rs;
        rs=IVksum(m,t,h,-dx,y);if(rs>res) res=rs;
        sp+=x;
        sm+=y;
    }
    tsigma=0.5*(sp-sm)/dx*s_IV*h;
    
    return res;
};

//calculate conductance
int tunnelconduct(double t,double h,double v,double Tctau,double &dIdVs,double &dIdVi,double &Iregav,double &Ianav)
{
 double IVsp,IVsm,IVip,IVim,x,dx;
 int res=0;
    
    dx=DX;
    if(ABS(v)<PX) dx=PX;
    
    res=tunnelIV(t,h,v+dx,Tctau,IVsp,IVip,x);
    if(res==0) {
        res=tunnelIV(t,h,v-dx,Tctau,IVsm,IVim,x);
 
        dIdVs=0.5*(IVsp-IVsm)/dx;
        dIdVi=0.5*(IVip-IVim)/dx;
        Iregav=0.5*(IVsp+IVsm);
        Ianav=0.5*(IVip+IVim);
    }
    return res;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

double PsiNom2(int k2,double a,double b)
 {
   double p=0.0;
   double den;
   double Pj[101];
   int j;
   if(k2>100) return 0.0;
   Pj[0]=Digamma(a);
   den=b+Pj[0];
   for(j=1;j<=2*k2;j++) Pj[j]=Polygamma(a,j);
   switch (k2) {
	case 1: p=Pj[1]*Pj[2];
	case 2: p=0.5*(6*Pj[1]*Pj[1]*Pj[1]*Pj[2]+den*den*Pj[2]*Pj[3]+den*Pj[1]*(-6*Pj[2]*Pj[2]+den*Pj[4]));

	default:p=0.0;
   }

   return p;
 }

double sigma_DOS(double t,double h)
{
 int kmax=50; //the k-summation has to be cut here, since the 100th-polygamma function is the highest possible derivative
 int mmax=50; //1/T\tau ???
 int k,m;
 double *Ak;
 double sum,logdenom2,a,b;

 h=h/t/PI/PI;
 t=log(t)-c2;
 sum=0.0;

 Ak=new double[kmax+1];
 for(k=0;k<=kmax;k++)
  Ak[k]=Bernoulli(k+k)/Factorial(k+k);

 for(m=0;m<mmax;m++)
  {
	a=h*(4*m+2);
	logdenom2=2.0*log(ABS(t+Digamma(0.5+a)));
	for(k=1;k<kmax;k++)
	 {
		 b=0.5*k+a;
		 sum+=Polygamma(b,2)/(t+Digamma(b)); //first part

		 sum-=exp(-k*logdenom2)*Ak[k]*PsiNom2(k,0.5+a,t); //second term
	 }
  }
 delete[] Ak;
 return -0.5*h/PI*sum;
}
//---------------------------------------------------------------------------

double Tc(double h)
{
   double hc=2.13733033448880260609411984906;

   if(h>=hc) return 1e-6;

   return sqrt(1-h/hc);
}


double Delta(double t)
{
	if(t>=1.0) return 0.0;
	return sqrt(cos(c3*t*t));
}


//---------------------------------------------------------------------------
//------------------------ FLUCTUOSCOPE CLASS -------------------------------
//---------------------------------------------------------------------------

FSCOPE::FSCOPE(int ver)
{
    ctype=-1; //output info
    
    tmin=1.0;
    dt=0.1;
    Nt=10;
    St=0;
    
    hmin=0.0;
    dh=0.0;
    Nh=1;
    Sh=0;
    
    vmin=0.0;
    dv=0.0;
    Nv=1;
    Sv=0;
    
    t=tmin;
    h=hmin;
    v=vmin;
    
    Tct=0.01;
    Tctp=0.001;
    usedelta=false;
    setdelta();
    
    aniso=1.0;
    
    exe="FSCOPE";
    
    hc2para_s=1.1;
    
    mval=0;kmin=0;Nk=1;
    
    version=ver;
};

FSCOPE::~FSCOPE()
{
  //nothing to do at the moment
};


int FSCOPE::readparams(int argc, char **argv)
{
    int a;
    exe=argv[0];
    //overwrite deault built-in parameters set in constructor
    
    paramfilereader *pf=new paramfilereader();
    a=0;
    if(argc>1) a=pf->cmdlineopenread(argc,argv);
    //if(a<1) a=pf->openread("FSCOPE.ini"); //printf info instead
    if(a>0)
    {
        ctype=pf->getint("ctype",ctype);
        
        tmin=pf->getdouble("tmin",tmin);t=tmin;
        dt=pf->getdouble("dt",dt);
        hmin=pf->getdouble("hmin",hmin);h=hmin;
        dh=pf->getdouble("dh",dh);
        vmin=pf->getdouble("vmin",vmin);v=vmin;
        dv=pf->getdouble("dv",dv);
        
        Nt=pf->getint("Nt",Nt);
        Nh=pf->getint("Nh",Nh);
        Nv=pf->getint("Nv",Nv);
        
        St=pf->getint("St",St);
        Sh=pf->getint("Sh",Sh);
        Sv=pf->getint("Sv",Sv);
        
        Tct=pf->getdouble("Tc0tau",Tct);
        Tctp=pf->getdouble("Tc0tauphi",Tctp);
        delta=pf->getdouble("delta",-1.0);
        
        
        aniso=pf->getdouble("aniso",1.0);
        
        hc2para_s=pf->getdouble("hc2s",hc2para_s);

        if(delta>0.0)
        {
            usedelta=true;
            setTctp();
        }
        else setdelta();
        
        
        kmin=pf->getint("kmin",kmin);
        Nk=pf->getint("Nk",Nk);
        mval=pf->getint("mval",mval);
    }
    delete pf;
    return 0;
};

void FSCOPE::info()
{
    int Vmaj,Vmin;
    
    Vmaj=version>>8;
    Vmin=version&0xFF;
    
    printf("usage of the FLUCTUOSCOPE - version %d.%d\n",Vmaj,Vmin);
    printf("  %s parameterfile.ini\n",exe.c_str());
    printf("  %s param1=value1 param2=value2 ....\n\n",exe.c_str());
    printf(" The parameter file has one param=value definition per line, lines without = are ignored, # starts a comment\n\nParameters:\n");
    printf("- ctype=[integer] : computation type, see list below\n");
    printf("- tmin=[float] : minimal temperature value in units of Tc0\n");
    printf("- dt=[float] : temperature interval in units of Tc\n");
    printf("- Nt=[integer] : number temperature value steps, i.e., t=tmin,tmin+dt,...,tmin+(Nt-1)*dt\n");
    printf("- St=[integer] : temperature scale: 0 - linear [default], 1 - log10, 2 - ln\n");
    printf("- hmin,dh,Nh,Sh : accoringly for the dimensionless magnetic field h\n");
    printf("- vmin,dv,Nv,Sv : accoringly for the dimensionless voltage v (for tunnel current/conductance)\n");
    printf("- Tc0tau=[float] : value of Tc0*tau\n");
    printf("- Tc0tauphi=[float] : value of Tc0*tau_phi\n");
    printf("- delta=[float] : value of delta=pi/(8*t*Tctp) [if set, overrides Tctp and activates Tctp calculation when temperature is changed]\n");
    
    printf("\nctype values (about 1000: internal tests - avoid):\n");
    printf("1: output hc2(t) for t=0..1 using Nt t-steps (plus approximation)\n");
    
    printf("100: full fluctuation conductivity calculation using t,h\n");
    printf("111: full fluctuation conductivity parallel to hc2(t), use parameter hc2s>1 and Nt\n");
    
    
    printf("200: tunnel IV using t,h,v (should fix one Nx to 1 or get 4D data)\n");
    printf("201: tunnel conductance using t,h,v (should fix one Nx to 1 or get 4D data)\n");
    printf("202: zero bias tunnel conductance using t,h\n");
    printf("211: tunnel conductance parallel to hc2(t), use parameter hc2s>1 and Nt, v\n");
    printf("290: test\n");
    
    printf("300: Nernst beta_xy using t,h\n");
    printf("390: test\n");

    printf("400: NMR 1/T1 normalized to Karringaand Gi(2) using t,h\n");
    printf("410: as 400, but along hc2(t) uses Nt only and hc2s>1\n");
    printf("403: 3D NMR 1/T1 normalized to Karringaand Gi(2) using t,h and aniso (r)\n");
    printf("413: as 403, but along hc2(t) uses Nt only and hc2s>1\n");
    
    printf("500: susceptibility t,h\n");
//    printf("530: susceptibility (Bulavesvskii) at t=0, using h (>0.69)\n");

    
};


//---------------------------------------------------------------------------

int FSCOPE::hc2curve() //uses only Nt
{
    int i;
    printf("# hc2(t) from E_0(0)=0 and TB approx\n#t\thc2(t)\tapprox\n");
    dt=1.0/(1.0*Nt);
    for(i=0;i<=Nt;i++)
    {
        t=i*dt;
        h=hc2(t);
        hmin=hc2_approx(t);
        printf("%le\t%le\t%le\n",t,h,hmin);
    }
    return 0;
};

//---------------------------------------------------------------------------

int FSCOPE::calcFC()
{
    int i,j,k;
    double sAL,sMTsum,sMTint,sDOS,sCC,MC;
    string stn,shn;
    printf("#fluctuation conductivity data\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    printf("#t\th\tSC\tsAL\tsMTsum\tsMTint\tsDOS\tsDCR\tsigma\n");
    for(j=0;j<Nh;j++)
    {h=applyscale(Sh,hmin+j*dh);
        for(i=0;i<Nt;i++)
        {
            t=applyscale(St,tmin+i*dt);
            if (usedelta) setTctp();
            k=1;if(h<hc2(t)) k=0;
            if(k==1)
            {k+=MC_sigma(t,h,Tct,Tctp,sAL,sMTsum,sMTint,sDOS,sCC);
            }
            else {sAL=sMTsum=sMTint=sDOS=sCC=0.0;}
            MC=sAL+sMTsum+sMTint+sDOS+sCC;
            printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n",t,h,k,sAL,sMTsum,sMTint,sDOS,sCC,MC);
        };
    };
    return 0;
};

int FSCOPE::calcFC_hc2()
{//sigma along hc2(t), hmin is positive offset
    int i,k;
    double x,dx,MC,sAL,sMTsum,sMTint,sDOS,sCC;
    
    dx=1.0/(1.0*Nt);
    
    printf("#fluctuation conductivity data along hc2(t)\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);
    printf("#s=%le, N_t=%d , dx=%le\n",hc2para_s,Nt,dx);
    printf("#t\th\tSC\tsAL\tsMTsum\tsMTint\tsDOS\tsDCR\tsigma\n");
    
    
    for(i=0;i<Nt;i++)
    {
        x=i*dx;set_th_para_hc2(x);
        if (usedelta) setTctp();
        k=MC_sigma(t,h,Tct,Tctp,sAL,sMTsum,sMTint,sDOS,sCC);
        MC=sAL+sMTsum+sMTint+sDOS+sCC;
        printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n",t,h,k,sAL,sMTsum,sMTint,sDOS,sCC,MC);
    };
    return 0;
};

//---------------------------------------------------------------------------


int FSCOPE::calctunnelIV()
{//tunnel IV
    int i,j,k,l;
    double y1,y2,y3,y;
    string stn,shn,svn;
    
    printf("#fluctuation tunnel current data\n");
    printf("#Tc0*tau=%le\n",Tct);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    svn=scalename(Sv,"v");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d, %s=[%le..%le], N_v=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh,svn.c_str(),vmin,vmin+Nv*dv-dv,Nv);

    printf("#t\th\tv\tSC\tIreg\tIan\tI(v)\n");
    
    for (j=0; j<Nh; j++) {
        h=applyscale(Sh,hmin+j*dh);
        for (i=0; i<Nt; i++) {
            t=applyscale(St,tmin+i*dt); //tauphi not used in TIV
            for(l=0;l<Nv;l++) {
                v=applyscale(Sv,vmin+l*dv);
                
                k=1;if(h<hc2(t-DX)) k=0;
                if(k==1) k+=tunnelIV(t,h,v,Tct,y1,y2,y3);
                else {y1=y2=0.0;}
                y=y1+y2;
                printf("%le\t%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,v,k,y1,y2,y);fflush(stdout);
            };
        };
    };
    return 0;
};

int FSCOPE::calctunnelconduct()
{//tunnel conductance and IV (av over adjacent points)
    int i,j,k,l;
    double y1,y2,y3,y4,y,z;
    string stn,shn,svn;
    
    printf("#fluctuation tunnel conductance data\n");
    printf("#Tc0*tau=%le\n",Tct);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    svn=scalename(Sv,"v");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d, %s=[%le..%le], N_v=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh,svn.c_str(),vmin,vmin+Nv*dv-dv,Nv);
    
    printf("#t\th\tv\tSC\tsigma_reg\tsigma_an\tsigma_t\tIreg_av\tIan_av\tI_av\n");
    
    for (j=0; j<Nh; j++) {
        h=applyscale(Sh,hmin+j*dh);
        for (i=0; i<Nt; i++) {
            t=applyscale(St,tmin+i*dt);
            for(l=0;l<Nv;l++) {
                v=applyscale(Sv,vmin+l*dv);
                k=1;if(h<hc2(t-DX)) k=0;
                if(k==1) k+=tunnelconduct(t,h,v,Tct,y1,y2,y3,y4);
                else {y1=y2=y3=y4=0.0;}
                y=y1+y2;
                z=y3+y4;
                printf("%le\t%le\t%le\t%d\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n",t,h,v,k,y1,y2,y,y3,y4,z);fflush(stdout);
            };
        };
    };
    return 0;
};

int FSCOPE::calczerobiastunnelconduct()
{//tunnel conductance at zero voltage (t & h dependence)
    int i,j,k;
    double y;
    string stn,shn;
    
    printf("#fluctuation zero-bias tunnel conductance data\n");
    printf("#Tc0*tau=%le\n",Tct);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    
    printf("#t\th\tv\tSC\tsigma_reg0\n");
    
    for (j=0; j<Nh; j++) {
        h=applyscale(Sh,hmin+j*dh);
        for (i=0; i<Nt; i++) {
            t=applyscale(St,tmin+i*dt);
            
                
                k=1;if(h<hc2(t-DX)) k=0;
                if(k==1) k+=zeroVtunnelconduct(t,h,Tct,y);
                else {y=0.0;}

                printf("%le\t%le\t%le\t%d\t%.16le\n",t,h,v,k,y);fflush(stdout);
        };
    };
    return 0;
};

int FSCOPE::calctunnelconduct_hc2()
{//tunnel conductance for long a line "parallel" to hc2(t) (v & t (& h(t)) dependence)
    
    int i,k,l;
    double y1,y2,y3,y4,y,z,dx,x;
    string svn;
    
    dx=1.0/(1.0*Nt);
    
    printf("fluctuation tunnel conductance data along hc2(t)\n");
    printf("#Tc0*tau=%le\n",Tct);
    svn=scalename(Sv,"v");
    printf("#s=%le, N_t=%d , dx=%le, %s=[%le..%le], N_v=%d\n",hc2para_s,Nt,dx,svn.c_str(),vmin,vmin+Nv*dv-dv,Nv);
    
    printf("#t\th\tv\tSC\tsigma_reg\tsigma_an\tsigma_t\tIreg_av\tIan_av\tI_av\n");
    
    for (i=0; i<Nt; i++) {
            x=i*dx;set_th_para_hc2(x);
            for(l=0;l<Nv;l++) {
                v=applyscale(Sv,vmin+l*dv);
                k=1;if(h<hc2(t-DX)) k=0;
                if(k==1) k+=tunnelconduct(t,h,v,Tct,y1,y2,y3,y4);
                else {y1=y2=y3=y4=0.0;}
                y=y1+y2;
                z=y3+y4;
                printf("%le\t%le\t%le\t%d\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n",t,h,v,k,y1,y2,y,y3,y4,z);fflush(stdout);
            };
    };
    return 0;

};

int FSCOPE::testtunnelIV_int()
{
    //test of the integrand, use Nk for M
    //v=vmin fixed
    double x,y,y1;
    int M;

    M=Nk;
    v=vmin;
    x=-5-ABS(v);
    printf("# tunnel IV integrand test, using parameters Nk=m, vmin, t,h\n");
    printf("# m=%d, t=%le, h=%le, v=%le\n",M,t,h,v);
    while(x<5+ABS(v))
    {
        IVintegrand(M,t,h,v,x,y);
        IVintegrand(M,t,h,v,x,y1,0);
        x+=0.001;
        printf("%le\t%le\t%le\n",x,y,y1);fflush(stdout);
    }
    return 0;
};

//---------------------------------------------------------------------------


int FSCOPE::calcNernstbeta()
{//Nerst signal beta_xy
    int i,j,k;
    double y1,y2,y;
    string stn,shn;
    
    printf("#Nernst beta function\n");
    printf("#Tc0*tau=%le\n",Tct);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    printf("#t\th\tSC\tNCsum\tNCint\tbeta_xy\n");
    for(j=0;j<Nh;j++)
    {h=applyscale(Sh,hmin+j*dh);
        for(i=0;i<Nt;i++)
        {
            t=applyscale(St,tmin+i*dt);
            k=1;if(h<hc2(t-DX)) k=0;
            if(k==1) k+=NC_beta_xy(t,h,Tct,y1,y2);
            else {y1=y2=0.0;}
            y=y1+y2;
            printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,k,y1,y2,y);
        };
    };
    return 0;
};

int FSCOPE::testNernstbeta_sum()
{
    //test Nerst signal beta_xy m-sum
    int k;
    double y1,y2;
    
    printf("#Nernst beta sum test\n");
    printf("#Tc0*tau=%le\n",Tct);
    printf("#t=[%le..%le], N_t=%d , h=[%le..%le], N_h=%d\n",tmin,tmin+Nt*dt-dt,Nt,hmin,hmin+Nh*dh-dh,Nh);
    printf("#m\tsum\tint\n");
    
    h=hmin;
    t=tmin;
    k=1;if(h<hc2(t)) k=0;
    if(k==1) {
        NC_beta_xy(t,h,Tct,y1,y2,true);
        printf("# %le\t%le\t%.16le\t%.16le\n",t,h,y1,y2);
    }
    else printf("SC\n");
    return 0;
};

//---------------------------------------------------------------------------


int FSCOPE::calcNMR() {
    int i,j,k;
    double NMRsum,NMRint,NMR;
    string stn,shn;
    printf("#fluctuation contribution to 1/T1, normalized to Karringa&Gi(2)\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    printf("#t\th\tSC\tNMRsum\tNMRint\tNMR\n");
    for(j=0;j<Nh;j++)
    {h=applyscale(Sh,hmin+j*dh);
        for(i=0;i<Nt;i++)
        {
            t=applyscale(St,tmin+i*dt);
            if (usedelta) setTctp();
            k=1;if(h<hc2(t)) k=0;
            if(k==1)
            {k+=T1norm(t,h,Tct,Tctp,NMRsum,NMRint);
            }
            else {NMRsum=NMRint=0.0;}
            NMR=NMRsum+NMRint;
            printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,k,NMRsum,NMRint,NMR);
        };
    };
    return 0;
};

int FSCOPE::calcNMR_hc2()
{//NMR along a line "parallel" to hc2(t) ( t & h(t)) dependence)
    
    int i,k;
    double NMRsum,NMRint,NMR,dx,x;
    string stn,shn;
    
    if(tmin+DX>1.0) return -1;
    
    
    printf("#fluctuation contribution to 1/T1 along hc2(t), normalized to Karringa&Gi(2)\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);
    
    
    dx=(1.0-tmin)/(1.0*Nt);
    printf("#s=%le, N_t=%d, dx=%le\n",hc2para_s,Nt,dx);

    printf("#t\th\tSC\tNMRsum\tNMRint\tNMR\n");
    
    
    
    for (i=0; i<Nt; i++) {
        x=tmin+i*dx;set_th_para_hc2(x);
        if (usedelta) setTctp();
        k=1;if(h<hc2(t-DX)) k=0;
        if(k==1) {k+=T1norm(t,h,Tct,Tctp,NMRsum,NMRint);
        }
        else {NMRsum=NMRint=0.0;}
        NMR=NMRsum+NMRint;
        printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,k,NMRsum,NMRint,NMR);
    };
    
    return 0;
};


int FSCOPE::calcNMR_3D() {
    int i,j,k;
    double NMRsum,NMRint,NMR;
    string stn,shn;
    printf("#fluctuation contribution to 1/T1 in 3D, normalized to Karringa&Gi(2)\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    printf("#r=%le\n",aniso);
    printf("#t\th\tSC\tNMRsum\tNMRint\tNMR\n");
    for(j=0;j<Nh;j++)
    {h=applyscale(Sh,hmin+j*dh);
        for(i=0;i<Nt;i++)
        {
            t=applyscale(St,tmin+i*dt);
            if (usedelta) setTctp();
            k=1;if(h<hc2(t)) k=0;
            if(k==1)
            {k+=T1norm3D(t,h,Tct,Tctp,aniso,NMRsum,NMRint);
            }
            else {NMRsum=NMRint=0.0;}
            NMR=NMRsum+NMRint;
            printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,k,NMRsum,NMRint,NMR);
        };
    };
    return 0;
};


int FSCOPE::calcNMR_3D_hc2()
{//NMR along a line "parallel" to hc2(t) ( t & h(t)) dependence)
    
    int i,k;
    double NMRsum,NMRint,NMR,dx,x;
    string stn,shn;
    
    if(tmin+DX>1.0) return -1;
    
    printf("#fluctuation contribution to 1/T1 in 3D along hc2(t), normalized to Karringa&Gi(2)\n");
    printf("#Tc0*tau=%le, ",Tct);
    if(usedelta) printf("delta=gphi/t=%le\n",delta);
    else printf("Tc0*tauphi=%le\n",Tctp);

    dx=(1.0-tmin)/(1.0*Nt);
    printf("#s=%le, N_t=%d, dx=%le\n",hc2para_s,Nt,dx);
    

    printf("#r=%le\n",aniso);
    printf("#t\th\tSC\tNMRsum\tNMRint\tNMR\n");
    
    
    
    for (i=0; i<Nt; i++) {
        x=tmin+i*dx;set_th_para_hc2(x);
        if (usedelta) setTctp();
        k=1;if(h<hc2(t-DX)) k=0;
        if(k==1) {k+=T1norm3D(t,h,Tct,Tctp,aniso,NMRsum,NMRint);
        }
        else {NMRsum=NMRint=0.0;}
        NMR=NMRsum+NMRint;
        printf("%le\t%le\t%d\t%.16le\t%.16le\t%.16le\n",t,h,k,NMRsum,NMRint,NMR);
    };
    
    return 0;
};

//---------------------------------------------------------------------------


int FSCOPE::calcsuscept() {
    int i,j,k;
    double CHI;
    string stn,shn;
    printf("#fluctuation contribution to magnetic susceptibility\n");
    printf("#Tc0*tau=%le\n",Tct);
    stn=scalename(St,"t");
    shn=scalename(Sh,"h");
    printf("#%s=[%le..%le], N_t=%d , %s=[%le..%le], N_h=%d\n",stn.c_str(),tmin,tmin+Nt*dt-dt,Nt,shn.c_str(),hmin,hmin+Nh*dh-dh,Nh);
    printf("#t\th\tSC\tchi\n");
    for(j=0;j<Nh;j++)
    {h=applyscale(Sh,hmin+j*dh);
        for(i=0;i<Nt;i++)
        {
            t=applyscale(St,tmin+i*dt);
            k=1;if(h<hc2(t)) k=0;
            if(k==1)
            {k+=chi(t,h,Tct,CHI);
            }
            else {CHI=0.0;}
            printf("%le\t%le\t%d\t%.16le\n",t,h,k,CHI);
        };
    };
    return 0;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

int FSCOPE::execute()
{   
    int res=0;
    
    //run the ctypes
	switch (ctype) {
            
        case 1:res=hc2curve();break;
            
        case 100:res=calcFC();break;
        case 111:res=calcFC_hc2();break;
            
        case 200:res=calctunnelIV();break;
        case 201:res=calctunnelconduct();break;
        case 202:res=calczerobiastunnelconduct();break;
        case 211:res=calctunnelconduct_hc2();break;
        case 290:res=testtunnelIV_int();break;
		
        case 300:res=calcNernstbeta();break;
        case 390:res=testNernstbeta_sum();break;
            
        case 400:res=calcNMR();break;
        case 403:res=calcNMR_3D();break;
        case 410:res=calcNMR_hc2();break;
        case 413:res=calcNMR_3D_hc2();break;
            
        case 500: res=calcsuscept();break;
//        case 530: res=calcsuscept_B_t0();break;

		default:info();
			
	}
	
	return res;
};
//---------------------------------------------------------------------------
