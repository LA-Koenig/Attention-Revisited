#!/bin/bash
# $1 is the number of times to run each combo

for task in 'SG' 'SA' 'FC' 'NF'
do
	for ((x=0; x<$1; x++))
	do
		echo $task 
		echo $x
	done
done
