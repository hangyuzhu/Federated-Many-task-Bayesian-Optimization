cd ..
for fn in 'Ackley' 'Griewank' 'Ellipsoid' 'Rastrigin' 'Rosenbrock'
do
  python fmbo_main.py --num_runs 10 --iid --total_clients 10 --func $fn --dim 10 --save_results
done