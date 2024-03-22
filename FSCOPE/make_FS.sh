#!/bin/sh

#first parameter is the name of the executable
EXE=$1

if [ "$EXE" = "" ]; then EXE="FSCOPE";fi

#compiler,: g++, icc, ...
COMP=icc
#compiler flags
CFLAGS=-fast -lm

$COMP AGM_sfunctions.cpp fileutils.cpp stringutils.cpp paramfile.cpp fluctuoscope.cpp -c  $CFLAGS
 
$COMP main.cpp fluctuoscope.o  AGM_sfunctions.o fileutils.o stringutils.o paramfile.o $CFLAGS -o $EXE
