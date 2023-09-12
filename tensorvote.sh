#!/bin/sh

# $1 = input image
# $2 = sigma value

# extract basename from arg 1
FNAME=$(basename "$1")

NPY_NAME="${FNAME%.*}.npy"

# get structure tensor
python structure2d.py $1 $NPY_NAME

# run tensorvoting
./tensorvote.exe $NPY_NAME $NPY_NAME $2