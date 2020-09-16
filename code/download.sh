#!/bin/bash

echo "Downloading data..."
wget http://moinnadeem.com/csa/models.zip
wget http://moinnadeem.com/csa/gigaword_data.zip

echo "Setting up Gigaword..."
unzip gigaword_data.zip
mkdir data
mv gigaword_public data/gigaword 

echo "Setting up models..."
unzip models.zip
mv models_to_download/models .
rm -r models_to_download

echo "Deleting temporary files."
rm models.zip
rm gigaword_data.zip
