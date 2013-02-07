build:
	python setup.py build_ext --inplace

test: build
	python test.py

clean:
	rm -f *.pyc *.so Q1.c poisson.c
	python setup.py clean --all