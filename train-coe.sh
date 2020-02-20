#$ -l gpu=1,h_rt=100:0:0
#$ -q gpu.q@@2080
#$ -N single-parlai
#$ -cwd
source ~/.bashrc
#
#conda deactivate
#conda activate py3

conda deactivate
conda activate parlai
module load cuda10.1/toolkit/ cudnn/7.6.1_cuda10.1
pwd
echo $(python --version)


TASK=carlostriples
model_dir="${TASK}_2-19"
rm -rf ${model_dir}
model_name='single'

python -m parlai.scripts.build_dict -t ${TASK} --dict-file ${model_dir}/${model_name}.model.dict

python -m parlai.scripts.train_model \
        -bs 8 \
        -t ${TASK} \
        -m hred/hred \
        -mf ${model_dir}/${model_name}.model \
        --embedding-type fasttext_cc \
        --embeddingsize 300 \
        -stim 1800 \
        --max-train-time 147600 \
        -veps 5 \
        -vme 5000 \
        -lr 0.001 \
        --dropout 0.1 \
        -sval True