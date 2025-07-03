$start = 376000
$end = 536000
$step = 4000
$name = "Motion_Deblurring"
$datasets = @("GoPro", "HIDE", "RealBlur_J", "RealBlur_R")

for ($number = $start; $number -le $end; $number += $step) {
    $resultDir = "./results/$name/$number"
    $weightsPath = "./models/$name/models/net_g_$number.pth"
    
    # # 创建结果目录
    # if (-not (Test-Path $resultDir)) {
    #     New-Item -ItemType Directory -Path $resultDir | Out-Null
    # }
    
    # 执行测试
    $datasets | ForEach-Object {
        $dataset = $_
        Write-Host "Processing $dataset for $number"
        & python test.py `
            --result_dir $resultDir `
            --weights $weightsPath `
            --dataset $dataset
    }
}
