#!/bin/bash


source ~/.bashrc

conda deactivate
conda activate py3

model_path=$1:-"/home/hltcoe/rwicks/ParlAI/models/forward.model.checkpoint"

python run.py -mf $model_path --sandbox 
