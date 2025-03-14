python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps >> run.out

python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --ref average >> run.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --ref max_reward >> run.out

uython train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 0.5 >> run.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 2.0 >> run.out



