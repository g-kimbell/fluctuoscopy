//
//  main.cpp
//  Fluctuoscope
//
//  Created by Glatz, Andreas
//  Copyright (c) 2009-2017 Glatz, Andreas. All rights reserved.
//
/* ---------------------------------------------------------------------------
 Calculation of fluctuation contribution to
 + Conductivity
 + Nernst
 + Tunnel-IV
 in the dirty limit for all h & t;
 Version 2.1
 - NME relaxation rate 3D
 Version 2.0
 - NMR relaxation rate 2D
 Version 1.00, 2012-06-27
 - renamed
 - make header
 - use paramfile
 - added linear part (sum) of tunneling IV
 Version 0.99, 2011-08-25
 - add Nernst coefficienct
 Version 0.95, 2011-04-18
 - added constant delta (type 12) calculations
 Version 0.9, 2010-04-22
 
 ---------------------------------------------------------------------------
 ---------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "fluctuoscope.h"


#include "stringutils.h"
#include "fileutils.h"
#include "paramfile.h"

//---------------------------------------------------------------------------
#define FSversion 0x0201
//---------------------------------------------------------------------------

#pragma argsused
int main(int argc, char * argv[])
{

    FSCOPE *fs;
    
    fs=new FSCOPE(FSversion);
    
    if(argc<2)
    {
        fs->info();
    }
    else
    {
        fs->readparams(argc, argv);
        fs->execute();
    }
    
    delete fs;
    
    return 0;
}

