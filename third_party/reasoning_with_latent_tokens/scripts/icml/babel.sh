

# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu -d sudoku-puzzle
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-full -d sudoku-puzzle
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar -d sudoku-puzzle
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-ntp -d sudoku-puzzle
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-mtp-window-32 -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-mtp-window-128 -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-mtp-window--1 -d sudoku-puzzle

# ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle 
# ./scripts/icml/gen_master.sh -m diffu-full -d sudoku-puzzle
# ./scripts/icml/gen_master.sh -m ar -d sudoku-puzzle
# ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle
# ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle
# ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle


{
  echo "============================================================"
  echo "ICML Sudoku runs started at $(date)"
  echo "============================================================"

  ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle -b 10 -- sampling.unmask_policy=topp
  ./scripts/icml/gen_master.sh -m diffu-full -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m diffu-full -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m diffu-full -d sudoku-puzzle -b 10 -- sampling.unmask_policy=topp
  ./scripts/icml/gen_master.sh -m ar -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m ar -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 10
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0

  echo "============================================================"
  echo "ICML Sudoku runs finished at $(date)"
  echo "============================================================"

} 2>&1 | tee -a results/sudoku_puzzle_gen_runs.txt

{
  echo "============================================================"
  echo "ICML Sudoku runs started at $(date)"
  echo "============================================================"

  ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m diffu-full -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 10 -- sampling.noise_scale=0.0

  echo "============================================================"
  echo "ICML Sudoku runs finished at $(date)"
  echo "============================================================"

} 2>&1 | tee -a results/sudoku_puzzle_gen_runs_noise_scale.txt


{
  echo "============================================================"
  echo "ICML Sudoku runs started at $(date)"
  echo "============================================================"
  ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle -b 1 -- sampling.kv_cache=False
  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 1 -- sampling.kv_cache=False
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 1 -- sampling.kv_cache=False
  ./scripts/icml/gen_master.sh -m ar-ntp -d sudoku-puzzle -b 1 -- sampling.noise_scale=0.0 sampling.kv_cache=False
  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 1 -- sampling.noise_scale=0.0 sampling.kv_cache=False
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 1 -- sampling.noise_scale=0.0 sampling.kv_cache=False

  ./scripts/icml/gen_master.sh -m diffu -d sudoku-puzzle -b 1 -- sampling.noise_scale=0.0 algo.ar_noise=True


  echo "============================================================"
  echo "ICML Sudoku runs finished at $(date)"
  echo "============================================================"

} 2>&1 | tee -a results/sudoku_puzzle_gen_runs_no_kv_cache.txt


  ./scripts/icml/gen_master.sh -m ar-mtp-window-32 -d sudoku-puzzle -b 1 -c last.ckpt -- sampling.noise_scale=0.0
  ./scripts/icml/gen_master.sh -m ar-mtp-window-128 -d sudoku-puzzle -b 1 -c last.ckpt -- sampling.noise_scale=0.0


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-mtp-window--1 -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-full -d sudoku-puzzle 
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal -d sudoku-puzzle 

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar -d zebra
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-mtp-window--1 -d zebra
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-full -d zebra 
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal -d zebra 



sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d sudoku-puzzle

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d zebra
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d zebra

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d game-of-24
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d game-of-24



# replicate old results
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal -d sudoku-small
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-sminy -d sudoku-small
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output -d sudoku-small
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-small

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output -d sudoku-small
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d sudoku-small


./scripts/icml/sweep_latent_steps.sh -m diffu-causal -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128

./scripts/icml/sweep_latent_steps.sh -m diffu-causal-sminy -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128

./scripts/icml/sweep_latent_steps.sh -m diffu-causal-output -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128

./scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-puzzle \
  --steps-list=256 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d zebra \
  --steps-list=384 --latent-list=0,8,32,64,128,256 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m ar-causal-output-sminy -d sudoku-puzzle \
  --steps-list=256 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m ar-causal-output-sminy -d zebra \
  --steps-list=384 --latent-list=0,8,32,64,128,256 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-puzzle \
  --steps-list=256 --latent-list=0,8,32,64,128 --ckpt=14-50000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d zebra \
  --steps-list=384 --latent-list=0,8,32,64,128,256 --ckpt=17-50000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-l2r -d sudoku-puzzle \
  --steps-list=256 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-l2r -d zebra \
  --steps-list=384 --latent-list=0,8,32,64,128,256 -b 10

./scripts/icml/gen_master.sh -m diffu-causal-output-sminy -d game-of-24 -b 1

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d game-of-24 \
  --steps-list=64 --latent-list=0,4,8,16,32 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m ar-causal-output-sminy -d game-of-24 \
  --steps-list=64 --latent-list=0,4,8,16,32 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-l2r -d game-of-24 \
  --steps-list=64 --latent-list=0,4,8,16,32 -b 10



sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m ar-causal-output-sminy -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-l2r -d sudoku-small \
  --steps-list=128 --latent-list=0,8,32,64,128 -b 10



./scripts/icml/gen_master.sh -m ar-causal-output-sminy -d game-of-24 -b 1



# retrain these after bug fix
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d game-of-24
# and other models
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d game-of-24



sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy-iglm -d sudoku-conditional
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m ar-causal-output-sminy -d sudoku-conditional


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional \
  --steps-list=256 --latent-list=0,8,32,64,128,256 -b 1

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-iglm -d sudoku-conditional \
  --steps-list=256 --latent-list=0,8,32,64,128,256 -b 1


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-uncond

./scripts/icml/gen_master.sh -m diffu-causal-output-sminy -d sudoku-conditional -b 1


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-uncond \
  --steps-list=256 --latent-list=0,8,32,64,128,256 -b 5

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-small-solver
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-small-solver \
  --steps-list=128 --latent-list=0,8,32,64,128 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-192 \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-20given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-30given


./scripts/icml/gen_master.sh -m esolmb -d openwebtext-split -b 1


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m esolmb -d openwebtext-split \
  --steps-list=1024 --latent-list=0,16,32,64,128,256,512 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-20given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-sminy -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-sminy-spt -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny-spt -d sudoku-conditional-192-0given

# already done
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-conditional-192-0given


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-sminy -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-tiny-spt -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-sminy -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-sminy-spt -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-sminy-spt -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-sminy-spt -d sudoku-conditional-192-0given

  
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-20given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given


# waiting for above...
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-20given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 -b 10


# 102-20000.ckpt  153-30000.ckpt  204-40000.ckpt  255-50000.ckpt  51-10000.ckpt

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=102-20000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=153-30000.ckpt -b 10 
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=51-10000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=102-20000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=153-30000.ckpt -b 10 
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=51-10000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=102-20000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=153-30000.ckpt -b 10 
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=51-10000.ckpt -b 10

./scripts/icml/gen_master.sh -m diffu-causal-output-sminy -d sudoku-small-solver -b 1


# diffu-causal-sminy-spt
# diffu-causal-sminy
# diffu-causal-tiny
# diffu-causal-tiny-spt

# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-conditional-192-0given
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-conditional-192-10given
# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-conditional-192-20given

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-sminy-spt -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-sminy -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny-spt -d sudoku-conditional-192-30given

# sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-tiny -d sudoku-puzzle



sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10 


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny -d sudoku-puzzle

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-micro -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-mini -d sudoku-puzzle
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-miny -d sudoku-puzzle

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-micro -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-mini -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-miny -d sudoku-conditional-192-30given

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-micro -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-mini -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-miny -d sudoku-conditional-192-10given

# sweeps for above models
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-micro -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=last.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-mini -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=last.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=last.ckpt -b 10  

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-micro -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-mini -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-miny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=255-50000.ckpt -b 10


sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-0given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-10given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/train_master.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-30given

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-30given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-10given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deeper -d sudoku-conditional-192-0given \
  --steps-list=192 --latent-list=0,8,32,64,128,192 --ckpt=204-40000.ckpt -b 10

# Run the sweep: n_latent_tokens in [0, 8, 16, 32, 64, 128] x target_givens in [0, 10, 20, 30]
for latent in 0 8 16 32 64 128; do
  for givens in 0 10 20 30; do
    echo "Running latent=$latent, givens=$givens"
    python scripts/eval_sudoku_infilling.py \
      --auto_checkpoint \
      --target_givens $givens \
      --n_latent_tokens $latent \
      --num_samples 100 \
      --output_dir results/infilling_sweep
  done
done

for latent in 0 8 16 32 64 128; do
  for givens in 0 10 20 30; do
    echo "Running latent=$latent, givens=$givens"
    python scripts/eval_sudoku_infilling.py \
      --auto_checkpoint \
      --target_givens $givens \
      --n_latent_tokens $latent \
      --num_samples 100 \
      --lock_special_tokens \
      --output_dir results/infilling_sweep_lock_special_tokens
  done
done

  python scripts/eval_sudoku_infilling.py \
    --auto_checkpoint \
    --target_givens 0 \
    --n_latent_tokens 0 \
    --num_samples 100 \
    --output_dir results/infilling_sweep


grep "valid_rate" results/infilling_sweep/*.txt | sed 's/.*givens\([0-9]*\)_latent\([0-9]*\).*valid_rate: \([0-9.]*\)%.*/\1 \2 \3/' | sort -t' ' -k1,1n -k2,2n | awk 'BEGIN {
    print "╔═══════════╦═══════════════════════════════════════════════════════════╗"
    print "║           ║                    n_latent_tokens                        ║"
    print "║   givens  ╠═════════╦═════════╦═════════╦═════════╦═════════╦═════════╣"
    print "║           ║    0    ║    8    ║   16    ║   32    ║   64    ║   128   ║"
    print "╠═══════════╬═════════╬═════════╬═════════╬═════════╬═════════╬═════════╣"
}
{
    data[$1][$2] = $3
}
END {
    for (g = 0; g <= 30; g += 10) {
        printf "║    %2d     ║", g
        for (l = 0; l <= 128; l = (l == 0 ? 8 : l * 2)) {
            if (l == 0) l_idx = 0
            printf " %5.1f%% ║", data[g][l]
        }
        print ""
    }
    print "╚═══════════╩═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝"
}'

sbatch scripts/icml/train_master.sh -m diffu-causal-output-tiny-deep -d sudoku-small-solver -- "+algo.log_position_losses=True"
sbatch scripts/icml/train_master.sh -m diffu-causal-output-sminy -d sudoku-small-solver -- "+algo.log_position_losses=True"


sbatch scripts/icml/train_master.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-0given -s 1

# Evaluate diffu-causal-output-tiny-deep-fq models across training checkpoints
# Sweeping latent tokens {0, 192} to check if latent scaling emerges during training
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=20-4000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=30-6000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=40-8000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=51-10000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=61-12000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=71-14000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=81-16000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-10given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=91-18000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=20-4000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=30-6000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=40-8000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=51-10000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=61-12000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=71-14000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=81-16000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-20given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=91-18000.ckpt -b 10

sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=20-4000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=30-6000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=40-8000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=51-10000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=61-12000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=71-14000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=81-16000.ckpt -b 10
sbatch --partition=general --gres=gpu:L40S:1 scripts/icml/sweep_latent_steps.sh -m diffu-causal-output-tiny-deep -d sudoku-conditional-192-30given \
  --run-suffix=fq --steps-list=192 --latent-list=0,192 --ckpt=91-18000.ckpt -b 10

# game-of-24 2x2grid evaluation (4 methods, 10 batches = ~2.5k problems)
sbatch scripts/icml/gen_v2.sh -m diffu-full -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10
sbatch scripts/icml/gen_v2.sh -m diffu-solo-full -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10
sbatch scripts/icml/gen_v2.sh -m ar -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10
sbatch scripts/icml/gen_v2.sh -m ar-mtp-causal-context -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10
sbatch scripts/icml/gen_v2.sh -m diffu-full -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10 -- "sampling.unmask_policy=topp"
sbatch scripts/icml/gen_v2.sh -m diffu-solo-full -z tiny -d game-of-24 --run-suffix=-2x2grid --batches=10 -- "sampling.unmask_policy=topp sampling.trim_masked_tokens=False"



sbatch scripts/icml/train_v2.sh -m ar -z sminy -d sudoku-small-solver --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m ar-mtp-full -z sminy -d sudoku-small-solver --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m diffu-full -z sminy -d sudoku-small-solver --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m diffu-solo-full -z sminy -d sudoku-small-solver --run-suffix=-exp1

sbatch scripts/icml/train_v2.sh -m ar -z sminy -d game-of-24 --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m ar-mtp-full -z sminy -d game-of-24 --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m diffu-full -z sminy -d game-of-24 --run-suffix=-exp1
sbatch scripts/icml/train_v2.sh -m diffu-solo-full -z sminy -d game-of-24 --run-suffix=-exp1



# 18-62000.ckpt
sbatch --output=out/sudoku-puzzle-diffu-full-tiny-2x2grid.out --error=out/sudoku-puzzle-diffu-full-tiny-2x2grid.err scripts/icml/gen_v2.sh -m diffu-full -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-62000.ckpt
sbatch --output=out/sudoku-puzzle-diffu-full-tiny-2x2grid-topp.out --error=out/sudoku-puzzle-diffu-full-tiny-2x2grid-topp.err scripts/icml/gen_v2.sh -m diffu-full -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-62000.ckpt -- "sampling.unmask_policy=topp"

sbatch --output=out/sudoku-puzzle-diffu-solo-full-tiny-2x2grid.out --error=out/sudoku-puzzle-diffu-solo-full-tiny-2x2grid.err scripts/icml/gen_v2.sh -m diffu-solo-full -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-64000.ckpt
sbatch --output=out/sudoku-puzzle-ar-tiny-2x2grid.out --error=out/sudoku-puzzle-ar-tiny-2x2grid.err scripts/icml/gen_v2.sh -m ar -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-64000.ckpt
sbatch --output=out/sudoku-puzzle-ar-mtp-causal-context-tiny-2x2grid.out --error=out/sudoku-puzzle-ar-mtp-causal-context-tiny-2x2grid.err scripts/icml/gen_v2.sh -m ar-mtp-causal-context -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-64000.ckpt
sbatch --output=out/sudoku-puzzle-diffu-solo-full-tiny-2x2grid-topp.out --error=out/sudoku-puzzle-diffu-solo-full-tiny-2x2grid-topp.err scripts/icml/gen_v2.sh -m diffu-solo-full -z tiny -d sudoku-puzzle --run-suffix=-2x2grid --batches=10 --ckpt=18-64000.ckpt -- "sampling.unmask_policy=topp sampling.trim_masked_tokens=False"


sbatch scripts/icml/train_v2.sh -m ar -z sminy -d zebra --run-suffix=-exp2
sbatch scripts/icml/train_v2.sh -m ar-mtp-full -z sminy -d zebra --run-suffix=-exp2
sbatch scripts/icml/train_v2.sh -m diffu-full -z sminy -d zebra --run-suffix=-exp2
sbatch scripts/icml/train_v2.sh -m diffu-solo-full -z sminy -d zebra --run-suffix=-exp2