#!/bin/bash

start=356000
end=456000
step=4000

for (( number=start; number<=end; number+=step )); do
  python test.py --result_dir ./results/Deraining_Max320_642111/$number --weights ./models/Deraining_Max320_642111/models/net_g_$number.pth
done
  