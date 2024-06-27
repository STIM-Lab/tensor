#! /usr/bin/bash

for noise_level in 0 100 500 1000 5000 10000 50000 100000 500000 1000000
do
    if [ ! -f boxgrid$noise_level.npy ]; then 
        ./image2tensor.exe --input "./data/boxgrid.bmp" --output "boxgrid${noise_level}.npy" --derivative 2 --order 9
    fi

    if [ ! -f circlegrid$noise_level.npy ]; then
        ./image2tensor.exe --input "./data/boxgrid.bmp" --output "circlegrid${noise_level}.npy" --derivative 2 --order 9
    fi

    for sigma in 1 5 10
    do

        if [ -f "boxgrid${noise_level}_sigma${sigma}.npy" ]; then
            rm "boxgrid${noise_level}_sigma${sigma}.npy"
        fi


        ./tensorvote.exe --input "boxgrid${noise_level}.npy" --output "boxgrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
        ./tensorvote.exe --input "circlegrid${noise_level}.npy" --output "circlegrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
    done
done