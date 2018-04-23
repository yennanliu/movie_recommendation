#!/bin/sh

function ml_latest_download() {

file='datasets/ml-latest'
echo $file

 if [[ -d "$file" ]]; then 
    echo "the file exists."
else
    echo "the file does not exist, start download the datasets....."
    hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
	hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }
	wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
	unzip -o "ml-latest.zip"
	DESTINATION="./datasets/"
	mkdir -p $DESTINATION
	mv ml-latest $DESTINATION
fi 
}



ml_latest_download