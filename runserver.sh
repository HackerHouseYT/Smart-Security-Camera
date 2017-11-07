#!/bin/bash

HOME=/home/pi
VENVDIR=$HOME/.virtualenvs/cv
BINDIR=$HOME/Smart-Security-Camera

cd $BINDIR
source $VENVDIR/bin/activate
python $BINDIR/main.py
