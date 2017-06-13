#!/bin/bash
for i in {1..9}
do
    wget "http://openconnecto.me/data/public/MR/archive/MIGRAINE_v1_0/KKI-42/small_graphs/KKI-21_KKI2009-0${i}_small_graph.mat"
done

for i in {10..42}
do
    wget "http://openconnecto.me/data/public/MR/archive/MIGRAINE_v1_0/KKI-42/small_graphs/KKI-21_KKI2009-${i}_small_graph.mat"
done
