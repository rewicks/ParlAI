#$ -l gpu=4,h_rt=100:0:0
#$ -q gpu.q@@2080
#$ -N parlai
#$ -cwd
source ~/.bashrc

conda deactivate
conda activate py3

pwd
echo $(python --version)

#python -m parlai.scripts.build_dict -t twitter --dict-file models/multi-gpu.model.dict

python -m parlai.scripts.multiprocessing_train \
        -bs 10 \
        -t twitter \
        -m seq2seq/seq2seq \
        -mf models/multi-gpu.model \
        -im models/multi-gpu.model.checkpoint \
        --embedding-type fasttext_cc \
        --numlayers 6 \
        --embeddingsize 300 \
        -stim 1800 \
        --max-train-time 147600 \
        --bidirectional True \
        --attention general \
        -veps 0.5 \
        --dropout 0.1 \
        -sval True
