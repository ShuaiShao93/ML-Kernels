set -ex

cd build
cmake ../cuda
cmake --build .
./conv3d_test

ncu -f --page=source --print-source=cuda --set=detailed -o /tmp/conv3d_test ./conv3d_test
