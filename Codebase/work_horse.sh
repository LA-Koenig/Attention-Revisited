#!/bin/bash

# $1 is the number of times to run each combo
corpus='len5_10000-train.txt'

for task in 'SG' 'SA' 'FC' 'NF'
do
	for model in LSTM_LSTM LSTM_Transformer Transformer_LSTM Transformer_Transformer
	do
		for ((x=0; x<$1; x++))
		do
			./single.sh $model'.py' $corpus $task'-10-train.txt' $task'-10-test.txt' $x	
		done
	done
	echo 
done
