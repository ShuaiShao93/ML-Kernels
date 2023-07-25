set -ex

cd build
cmake ../cuda
cmake --build .
./gemm_test

ncu -f --page=source --print-source=cuda --set=detailed -o /tmp/gemm_test ./gemm_test
