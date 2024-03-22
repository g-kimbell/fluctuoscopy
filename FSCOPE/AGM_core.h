//---------------------------------------------------------------------------
#ifndef AGMcoreH
#define AGMcoreH
//---------------------------------------------------------------------------
#include <string>
#include "AGM_const.h"
#include "AGM_data.h"

using namespace std;
//---------------------------------------------------------------------------

#define ABS(a) ((a)<0.0?(-(a)):(a))
#define SGN(a) ((a)<0.0?(-1):(1))

template<class T>
inline T SIGN(const T &a, const T &b)
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

template<class T>
inline T SQR(const T a) {return a*a;}

template<class T>
inline T MAX(const T &a, const T &b)
		{return b > a ? (b) : (a);}

template<class T>
inline T MIN(const T &a, const T &b)
		{return b < a ? (b) : (a);}

template<class T>
inline void SWAP(T &a, T &b)
	{T dum=a; a=b; b=dum;}
//---------------------------------------------------------------------------
//trunctated release ...
//---------------------------------------------------------------------------
#endif
