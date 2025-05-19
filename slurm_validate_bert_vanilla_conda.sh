#!/bin/sh

#SBATCH -J  shinyoungLee_bertTrain           # Job name
#SBATCH -o  %j.out    # Name of stdout output file (%j expands to %jobId)


#### Select  GPU
#SBATCH -p A100-80GB       # queue  name  or  partiton
#SBATCH   --gres=gpu:1          # gpus per node

##  node 지정하기
#SBATCH   --nodes=1              # the number of nodes 
#SBATCH   --ntasks-per-node=1
#SBATCH   --cpus-per-task=8

#SBATCH -q hpgpu

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

function cleanup {
    conda deactivate
    exit
}

## Set Trap -> Slurm에서 SIGTERM signal을 사용
trap cleanup 0

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

## 사용할 GPU의 UUID 구하기

echo "UUID GPU List"

UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"

## 실행할 명령어 기술 (아래는 예시 코드) ##
## 실행할 명령어들은 docker exec bash -c <명령어> 를 통해서 가능함

echo "source $HOME/miniconda3/etc/profile.d/conda.sh"
source $HOME/miniconda3/etc/profile.d/conda.sh

## wandb 설정
bash $HOME/set_wandb.sh
echo "Wandb Login Complete"


chmod +x $HOME/effi-ml-25/scripts/bert-vanilla-validate.sh

# docker exec ${DOCKER_NAME} bash -c "pip install -r /NAS/effi-ml-25/requirements.txt"
conda activate bert
echo "conda activate finished"

## script 실행
cd $HOME/effi-ml-25
bash scripts/bert-vanilla-validate.sh
#####################################

conda deactivate

## slurm 실행 정보 출력
date
squeue  --job  $SLURM_JOBID

echo  "##### END #####"
