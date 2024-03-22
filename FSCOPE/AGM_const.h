#ifndef AGMconstH
#define AGMconstH


//---------------------------------------------------------------------------
//floating point constants - copied from float.h
//long double is a 80-bit floating point number with 15bit exponent and 64bit mantissa

#define DBL_DIG             15
#define FLT_DIG             6
#define LDBL_DIG            18

#define DBL_MANT_DIG        53
#define FLT_MANT_DIG        24
#define LDBL_MANT_DIG       64

#define DBL_EPSILON         2.2204460492503131E-16
#define FLT_EPSILON         1.19209290E-07F
#define LDBL_EPSILON        1.084202172485504434e-019L

/* smallest positive IEEE normal numbers */
#define DBL_MIN             2.2250738585072014E-308   //2^-1022
#define FLT_MIN             1.17549435E-38F          //2^-126
#define LDBL_MIN            3.36210314311209350626267781732e-4932L //2^-16382

#define DBL_MAX             1.7976931348623157081E308  //(2-2^-52)*2^1023
#define FLT_MAX             3.4028234663852885981E38F  //(2-2^-23)*2^127
#define LDBL_MAX            1.18973149535723176505351158983e4932L //(2-2^-64)*2^16383


#define DBL_MAX_EXP         +1024
#define FLT_MAX_EXP         +128
#define LDBL_MAX_EXP        +16384

#define DBL_MAX_10_EXP      +308
#define FLT_MAX_10_EXP      +38
#define LDBL_MAX_10_EXP     +4932

#define DBL_MIN_10_EXP      -307
#define FLT_MIN_10_EXP      -37
#define LDBL_MIN_10_EXP     -4931

#define DBL_MIN_EXP         -1021
#define FLT_MIN_EXP         -125
#define LDBL_MIN_EXP        -16381

//---------------------------------------------------------------------------
//memory constants

#define MaxMatSize  10000  //about 1GB for double
#define MAXMATITER   50  //max. # iteration for matrix algorithms (default value)

//---------------------------------------------------------------------------
//******* error codes ***********************
//allgemein
#define noerr 0
#define divzero       -1 //division by zero

#define notimpl     -999

//linalg
#define LAdimerr   -30001  //dim error
#define LAsqrerr   -30002  //not a square matrix
#define LAnotreg   -30003  //matreix not regular
#define LAnokonv   -30010  //matrix iteration does not converge
#define LApivoterr -30020  //pivot error

//******************************************

//math constants
#define PI       3.14159265358979323846264338328
#define TWOPI    6.28318530717958647692528676656
#define PIHALF   1.57079632679489661923132169164
#define PIQUART  0.785398163397448309615660845820

#define SQRT2    1.4142135623730950488016887242097
#define INVLN10  0.434294481903251827651128918917 //  1/ln(10)
#define INVLN2   1.44269504088896340735992468100  //   1/ln(2)
#define LN2      0.693147180559945309417232121
#define LN10     2.3025850929940456840179914547

#define EULERG   0.577215664901532860606512090082 //gamma constant
#define EULER    2.71828182845904523536028747135
#define LG2      0.30102999566398119521373889472449 //lg(2)


//"pseudo" constants

#ifndef NAN
#define NAN (0.0/0.0)
#endif

// self defince constants
#define C_epsh     1e-18  //for iterations
#define C_eps      1e-32
#define C_epsmin   1e-8

#endif
