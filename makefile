all: hello.c setup.py
	python3 setup.py build_ext --inplace
