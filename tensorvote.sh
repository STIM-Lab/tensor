#! /usr/bin/bash

if [ -d ./data/image2tensor/ ]; then
    rm -rf ./data/image2tensor/
fi

if [ -d ./data/tensorvoting/ ]; then
    rm -rf ./data/tensorvoting/
fi

mkdir -p ./data/image2tensor/{images/{blurred/{box,circle},non-blurred/{box,circle}},npy/{box,circle}}
mkdir -p ./data/tensorvoting/{images/{box,circle},npy/{box,circle}}

for noise_level in 100 500 1000 5000 10000 50000 100000 500000 1000000
do
    echo "Noise level: $noise_level"

    ./image2tensor.exe --input "./data/boxgrid.bmp" --output "./data/image2tensor/npy/box/boxgrid${noise_level}.npy" --derivative 2 --order 9 --noise $noise_level
    ./image2tensor.exe --input "./data/circlegrid.bmp" --output "./data/image2tensor/npy/circle/circlegrid${noise_level}.npy" --derivative 1 --order 2 --noise $noise_level

    for sigma in 1 5 10
    do

        echo "Sigma: $sigma"

        if [ ! -d ./data/tensorvoting/npy/circle/sigma-${sigma} ]; then
            mkdir ./data/tensorvoting/npy/circle/sigma-${sigma}
        fi

        if [ ! -d ./data/tensorvoting/npy/box/sigma-${sigma} ]; then
            mkdir ./data/tensorvoting/npy/box/sigma-${sigma}
        fi

        ./tensorvote.exe --input "./data/image2tensor/npy/box/boxgrid${noise_level}.npy" --output "./data/tensorvoting/npy/box/sigma-${sigma}/boxgrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
        ./tensorvote.exe --input "./data/image2tensor/npy/circle/circlegrid${noise_level}.npy" --output "./data/tensorvoting/npy/circle/sigma-${sigma}/circlegrid${noise_level}_sigma${sigma}.npy" --sigma $sigma
    done
done

for sigma in 1 5 10
do

    if [ ! -d ./data/tensorvoting/images/circle/sigma-${sigma} ]; then
        mkdir -p ./data/tensorvoting/images/circle/sigma-${sigma}
    fi

    if [ ! -d ./data/tensorvoting/images/box/sigma-${sigma} ]; then
        mkdir -p ./data/tensorvoting/images/box/sigma-${sigma}
    fi

    for file in ./data/tensorvoting/npy/circle/sigma-${sigma}/*.npy
    do
        echo $(basename $file .npy).bmp
        ./tensorview2.exe --input $file --l0 ./data/tensorvoting/images/circle/sigma-${sigma}/$(basename $file .npy).bmp --nogui
    done

    for file in ./data/tensorvoting/npy/box/sigma-${sigma}/*.npy
    do
        echo $(basename $file .npy).bmp
        ./tensorview2.exe --input $file --l0 ./data/tensorvoting/images/box/sigma-${sigma}/$(basename $file .npy).bmp --nogui
    done
done

for file in ./data/image2tensor/npy/box/*.npy
do
    echo $(basename $file .npy).bmp
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/non-blurred/box/$(basename $file .npy).bmp --nogui
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/blurred/box/$(basename $file .npy).bmp --nogui --blur 3
done

for file in ./data/image2tensor/npy/circle/*.npy
do
    echo $(basename $file .npy).bmp
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/non-blurred/circle/$(basename $file .npy).bmp --nogui
    ./tensorview2.exe --input $file --l0 ./data/image2tensor/images/blurred/circle/$(basename $file .npy).bmp --nogui --blur 3
done