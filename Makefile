UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
OPEN_COMMAND := "xdg-open"
else
OPEN_COMMAND := "open"
endif

test:
	pytest -s ./tests

test_no_jit:
	env NUMBA_DISABLE_JIT=1 pytest -s ./tests

coverage:
	env NUMBA_DISABLE_JIT=1 pytest -s --cov=embiggen --cov-report=html ./tests
	$(OPEN_COMMAND) ./htmlcov/index.html & > /dev/null
