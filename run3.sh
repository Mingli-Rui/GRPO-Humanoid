python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 0.5 --ref max_return >> run3.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 1.0 --ref max_return >> run3.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 1.5 --ref max_return >> run3.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 2.0 --ref max_return >> run3.out
python train_similarity.py --run-name "run_sim" --seed 1 --n-epochs 500 --mps --sigma 2.5 --ref max_return >> run3.out

