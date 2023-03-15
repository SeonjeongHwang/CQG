#!/bin/bash -l

#SBATCH -J  MHCQG-element           # --job-name=singularity
#SBATCH -o  MHCQG-element.%j.out     # slurm output file (%j:expands to %jobId)
#SBATCH -p  A100-80GB         # queue or partiton name ; sinfo  output
#SBATCH -t  72:00:00          # 작업시간(hh:mm:ss) 1.5 hours 설정
#SBATCH -N  1                 # --nodes=1 (고정)
#SBATCH -n  1                 # --tasks-per-node=1  노드당 할당한 작업수 (고정)
#SBATCH --gres=gpu:1          # gpu Num Devices  가급적이면  1,2,4.6,8  2배수로 합니다.

conda activate hotpot

DATA_DIR=/home/seonjeongh/data/MHCQG/element_based
OUTPUT_DIR=element_level

python main.py --train_file $DATA_DIR/element_train.json \
               --dev_file $DATA_DIR/element_dev.json \
               --test_file $DATA_DIR/element_test.json \
               --model_folder $OUTPUT_DIR \
               --entity-level element \
               --mode train