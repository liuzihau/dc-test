
# method, dataset, num steps, num batches, checkpoint name
./scripts/icml/gen_master.sh ar zebra 384 1 40-120000.ckpt
./scripts/icml/gen_master.sh diffu-full zebra 384 1 40-120000.ckpt
./scripts/icml/gen_master.sh ar-ntp zebra 384 1 40-120000.ckpt


./scripts/icml/train_master.sh diffu-full-lr1e-4-bsz64 zebra false

./scripts/icml/gen_master.sh -m diffu-full-lr1e-4-bsz64 -d zebra -s 384 -b 1 \
    -- sampling.unmask_policy=random



sbatch scripts/icml/train_master.sh -m diffu -d zebra
sbatch scripts/icml/train_master.sh -m diffu-full -d zebra
sbatch scripts/icml/train_master.sh -m ar -d zebra
sbatch scripts/icml/train_master.sh -m ar-ntp -d zebra
sbatch scripts/icml/train_master.sh -m ar-mtp-window-32 -d zebra
sbatch scripts/icml/train_master.sh -m ar-mtp-window-128 -d zebra


sbatch scripts/icml/train_master.sh -m diffu -d sudoku-puzzle
sbatch scripts/icml/train_master.sh -m ar -d sudoku-puzzle
sbatch scripts/icml/train_master.sh -m ar-ntp -d sudoku-puzzle
sbatch scripts/icml/train_master.sh -m ar-mtp-window-32 -d sudoku-puzzle
sbatch scripts/icml/train_master.sh -m ar-mtp-window-128 -d sudoku-puzzle
