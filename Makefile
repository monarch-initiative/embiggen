test:
	pytest -s ./tests

test_no_jit:
	env NUMBA_DISABLE_JIT=1 pytest -s ./tests


