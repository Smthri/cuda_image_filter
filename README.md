# cuda_canny_image_filter
## Prerequisites
If ur on WSL 2 check `nvidia-smi` to get driver info. Then, get the corresponding toolkit (the meta package section in CUDA quick start guide).
Otherwise good luck.

## How to build and run:
- `mkdir build && cd build`
- `cmake ..  -DBLOCK_SIZE=<X> -DNUM_THREADS=<Y>`
- `make all`
- `./cuda_canny <src/image/path> <sigma> <thr_low> <thr_high>`

Will drop the resulting images for cpu and cuda executions, and will print out the timings to console.
