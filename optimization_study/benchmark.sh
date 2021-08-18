
kernprof -l ../package/example_noplots.py
python -m line_profiler example_noplots.py.lprof >> benchmark_2.txt

