# cuda_canny_image_filter
## Prerequisites
If ur on WSL 2 check `nvidia-smi` to get driver info. Then, get the corresponding toolkit (the meta package section in CUDA quick start guide).
Otherwise good luck.

## How to build and run:
- `mkdir build && cd build`
- `cmake ..  -DBLOCK_SIZE=<X> -DNUM_THREADS=<Y>`
- `make all`
- `./cuda_test`
