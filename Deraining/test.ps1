$start = 376000
$end = 536000
$step = 4000

for ($number = $start; $number -le $end; $number += $step) {
    python test.py --result_dir "./results/Deraining_Max320_442111_2/$number" --weights "./models/Deraining_Max320_442111_2/models/net_g_$number.pth"
}