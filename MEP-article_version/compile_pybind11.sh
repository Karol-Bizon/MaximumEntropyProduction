# compile.sh

c++ -O3 -shared -std=c++14 -fPIC $(python3.11 -m pybind11 --includes) radiatif.cpp -o radiatif2$(python3.11-config --extension-suffix)
