# NOTE: Run the following in "results" directory.

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

dataset=ml-20m
rng_seed=574575
num_gram=1
max_time=45
sv_threshold=10
log_every_sec=1

DIR=../getdata/${dataset}

mkdir $dataset

for k in 8 16 32 64; do
  for accelerated in true false; do
    for soft_threshold in true; do    # Soft or hard, doesn't matter much it seems.
      for (( num=0; num<1; num++ )); do
        for use_gpu in true false; do
          ../impute_main.o \
          --use_gpu=$use_gpu \
          --rng_seed=$rng_seed \
          --max_time=$max_time \
          --train_filename=$DIR/train_${num}.csr \
          --test_filename=$DIR/validate_${num}.csr \
          --train_t_filename=$DIR/train_${num}.t.csr \
          --train_perm_filename=$DIR/train_${num}.perm \
          --num_gram=$num_gram \
          --k=$k \
          --sv_threshold=$sv_threshold \
          --log_every_sec=$log_every_sec \
          --accelerated=$accelerated \
          --randn_iters=100000 \
          --output_filename=./$dataset/impute_${num}_${k}_${sv_threshold}_${soft_threshold}_${num_gram}_${use_gpu}_${accelerated}.tsv
        done
      done
    done
  done
done

for k in 8 16 32 64; do
  for (( num=0; num<5; num++ )); do
    ../sgd_main.o \
    --rng_seed=$rng_seed \
    --max_time=$max_time \
    --train_filename=$DIR/train_${num}.csr \
    --test_filename=$DIR/validate_${num}.csr \
    --log_every_sec=$log_every_sec \
    --k=$k \
    --output_filename=./$dataset/sgd_${num}_${k}.tsv
  done
done

./cf.o \
--use_gpu=true \
--rng_seed=$rng_seed \
--max_time=$max_time \
--train_filename=$DIR/train.txt \
--train_t_filename=$DIR/train-t.txt \
--test_filename=$DIR/test.txt \
--train_perm_filename=$DIR/train-map.txt \
--num_gram=$num_gram \
--num_iters=$num_iters \
--k=$k \
--sv_threshold=$sv_threshold \
--log_every_n=10 \
--stat_output=/tmp/$dataset/softimpute.txt

./cf.o \
--use_gpu=false \
--rng_seed=$rng_seed \
--max_time=$max_time \
--train_filename=$DIR/train.txt \
--train_t_filename=$DIR/train-t.txt \
--test_filename=$DIR/test.txt \
--train_perm_filename=$DIR/train-map.txt \
--num_gram=$num_gram \
--num_iters=$num_iters \
--k=$k \
--sv_threshold=$sv_threshold \
--log_every_n=5 \
--stat_output=/tmp/$dataset/softimpute_cpu.txt
