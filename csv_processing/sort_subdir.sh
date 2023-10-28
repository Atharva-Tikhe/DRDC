#!/bin/bash

#cat test_binary.csv | awk -F ',' '{print $2 ".png" "," $3}' >> test_processing.csv

zero=( $(cat test_processing.csv | awk -F ',' '{if($2==1){print $1}}') )


for file in ${zero[@]};
do
    echo Copying $file
    cp ./test_images/$file './test/1/'

done


