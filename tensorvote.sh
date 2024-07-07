#! /usr/bin/bash

if [ -d ./data/image2tensor/ ]; then
    rm -rf ./data/image2tensor/
fi

if [ -d ./data/tensorvoting/ ]; then
    rm -rf ./data/tensorvoting/
fi

mkdir -p ./data/image2tensor/{images/{blurred/{box,circle},non-blurred/{box,circle}},npy}
mkdir -p ./data/tensorvoting/{images/{box,circle},npy}

for noise_level in 0 100 500 1000 5000 10000 50000 100000 500000 1000000
do
    ./image2tensor.exe --input "./data/boxgrid.bmp" --output "./data/image2tensor/npy/box/boxgrid${noise_level}.npy" --derivative 2 --order 9 --noise $noise_level
    ./image2tensor.exe --input "./data/circlegrid.bmp" --output "./data/image2tensor/npy/circle/circlegrid${noise_level}.npy" --derivative 2 --order 9 --noise $noise_level

    for sigma in 1 5 10
    do

        if [ ! -d ./data/tensorvoting/npy/circle/sigma-${sigma} ]; then
            mkdir ./data/tensorvoting/npy/circle/sigma-${sigma}
        fi

        ./tensorvote.exe --input "./data/image2tensor/npy/box/boxgrid${noise_level}.npy" --output "./data/tensorvoting/npy/box/sigma-${sigma}/boxgrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
        ./tensorvote.exe --input "./data/image2tensor/npy/circle/circlegrid${noise_level}.npy" --output "./data/tensorvoting/npy/circle/sigma-${sigma}/circlegrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
    done
done

find ./data/tensorvoting/npy/box/ -type f | while read -r file; do
    sigma = $(basename "$(dirname "file")")
    
    if [ ! -d ./data/tensorvoting/images/box/sigma-${sigma} ]; then
        mkdir ./data/tensorvoting/images/box/sigma-${sigma}
    fi

    ./tensorview2.exe --input $file --l0 ./data/tensorovting/images/box/sigma-${sigma}/$(basename $file .npy).bmp --no-gui
done

find ./data/tensorvoting/npy/circle/ -type f | while read -r file; do
    sigma = $(basename "$(dirname "file")")
    
    if [ ! -d ./data/tensorvoting/images/circle/sigma-${sigma} ]; then
        mkdir ./data/tensorvoting/images/circle/sigma-${sigma}
    fi

    ./tensorview2.exe --input $file --l0 ./data/tensorovting/images/circle/sigma-${sigma}/$(basename $file .npy).bmp --no-gui
done

for file in ./data/image2tensor/npy/circle/*.npy
do
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/non-blurred/circle/$(basename $file .npy).bmp --no-gui
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/blurred/circle/$(basename $file .npy).bmp --no-gui --blur 5.0
done

for file in ./data/image2tensor/npy/box/*.npy
do
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/non-blurred/box/$(basename $file .npy).bmp --no-gui
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/blurred/box/$(basename $file .npy).bmp --no-gui --blur 5.0
done