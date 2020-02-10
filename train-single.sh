#$ -l gpu=1,h_rt=100:0:0
#$ -q gpu.q@@2080
#$ -N single-parlai
#$ -cwd
source ~/.bashrc

conda deactivate
conda activate py3

pwd
echo $(python --version)

python -m parlai.scripts.build_dict -t twitter --dict-file single-model/single.model.dict

python -m parlai.scripts.train_model \
        -bs 10 \
        -t twitter \
        -m seq2seq/seq2seq \
        -mf single-model/single.model \
        --embedding-type fasttext_cc \
        --numlayers 2 \
        --embeddingsize 300 \
        -stim 1800 \
        --max-train-time 147600 \
        -veps 0.5 \
        -lr 0.001 \
        --dropout 0.1 \
        -sval True
