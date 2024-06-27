#! /usr/bin/bash

if [ ! -f ./data/image2tenor/ ]; then
    mkdir ./data/image2tensor/
fi

if [ ! -f ./data/results/ ]; then
    mkdir ./data/tensorvoting/
fi

for noise_level in 0 100 500 1000 5000 10000 50000 100000 500000 1000000
do
    if [ -f boxgrid$noise_level.npy ]; then 
        rm boxgrid$noise_level.npy
    fi

    if [ -f circlegrid$noise_level.npy ]; then
        rm circlegrid$noise_level.npy
    fi
    
    ./image2tensor.exe --input "./data/boxgrid.bmp" --output "./data/image2tensor/boxgrid${noise_level}.npy" --derivative 2 --order 9 --noise $noise_level
    ./image2tensor.exe --input "./data/boxgrid.bmp" --output "./data/image2tensor/circlegrid${noise_level}.npy" --derivative 2 --order 9 --noise $noise_level

    for sigma in 1 5 10
    do

        if [ -f "boxgrid${noise_level}_sigma${sigma}.npy" ]; then
            rm "boxgrid${noise_level}_sigma${sigma}.npy"
        fi


        ./tensorvote.exe --input "./data/image2tensor/boxgrid${noise_level}.npy" --output "./data/results/boxgrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
        ./tensorvote.exe --input "./data/image2tensor/circlegrid${noise_level}.npy" --output "./data/results/circlegrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
    done
done