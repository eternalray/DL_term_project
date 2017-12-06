#!/bin/bash
counter=0
list=$1
prefix=$2
start_number=$3
while read line
do
	file_name=$prefix'_'$start_number
	youtube-dl -o $file_name.tmp -x --audio-format "wav" $line
	(( start_number ++ ))
done < $list