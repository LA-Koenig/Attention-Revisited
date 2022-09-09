#!/bin/bash

# $1 is the number of times to run each combo
corpus='len5_10000-train.txt'

base=$(pwd)
mkdir results
cd results

for task in 'SG' 'SA' 'FC' 'NF'
do
	echo 'Task:' $task
	for model in LSTM_LSTM LSTM_Transformer Transformer_LSTM Transformer_Transformer
	do
		echo 'Model:' $model 'running' $1 'times'
		mkdir -p ./$task/$model
		cd ./$task/$model
		pwd
		echo $base
		ln -s base/test.py ./test.py 
		ln -s base/single.sh ./single.sh 
		for ((x=0; x<$1; x++))
		do
			./single.sh 'test' $corpus $task'-10-train.txt' $task'-10-test.txt'
			#pwd
			#./single.sh $model $corpus $task'-10-train.txt' $task'-10-test.txt'
		done
		echo
		cd ../..
	done
	echo 
done
