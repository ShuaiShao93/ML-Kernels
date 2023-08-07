set -ex

cd build
cmake ../cuda
cmake --build .
./conv2d_test

ncu -f --page=source --print-source=cuda --set=detailed -o /tmp/conv2d_test ./conv2d_test
