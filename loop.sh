#! /bin/bash
for T in {250..300}
do
   for a in {1..200}
      do
        python ./2d_Ising_Wolff.py $T 100    
        printf $T" "$a"\n"
      done
done