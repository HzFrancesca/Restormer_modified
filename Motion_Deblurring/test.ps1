$start = 376000
$end = 536000
$step = 4000
$name = "Motion_Deblurring"


for ($number = $start; $number -le $end; $number += $step) {
    python test.py --result_dir "./results/$name/$number" --weights "./models/$name/models/net_g_$number.pth" --dataset GoPro
    python test.py --result_dir "./results/$name/$number" --weights "./models/$name/models/net_g_$number.pth" --dataset HIDE
    python test.py --result_dir "./results/$name/$number" --weights "./models/$name/models/net_g_$number.pth" --dataset RealBlur_J
    python test.py --result_dir "./results/$name/$number" --weights "./models/$name/models/net_g_$number.pth" --dataset RealBlur_R
}