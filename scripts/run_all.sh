#!/bin/sh


for i in 1 2 3 4 5 6
do
for j in 4 5
do

python roman_cosmic_shear_fisher_reference.py sompz_sc${i}_d${j}.yml

done
done

