UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
OPEN_COMMAND := "xdg-open"
else
OPEN_COMMAND := "open"
endif

test:
	pytest -s ./tests

coverage:
	pytest -s --cov=embiggen --cov-report=html ./tests
	$(OPEN_COMMAND) ./htmlcov/index.html & > /dev/null