$start = 352000
$end = 352000
$step = 4000

$name = "DefocusDeblur_Single_Max320_CxC_432000"

$inputDir = "./Datasets/test/DPDD/"

# --- 主循环 ---

for ($number = $start; $number -le $end; $number += $step) {
    

    $resultDir = "./results/$name/$number"
    $weightsPath = "./models/$name/models/net_g_$number.pth"
    # 检查权重文件是否存在，如果不存在则跳过当前循环
    if (-not (Test-Path $weightsPath)) {
        Write-Host "权重文件未找到: $weightsPath, 跳过测试。" -ForegroundColor Yellow
        continue
    }
    Write-Host "==================================================================="
    Write-Host "开始测试模型: net_g_$number.pth" -ForegroundColor Green
    Write-Host "结果将保存至: $resultDir" -ForegroundColor Cyan
    Write-Host "==================================================================="

    & python E:\2024HZF\Programs\Restormer_modified\Defocus_Deblurring\test_single_image_defocus_deblur.py `
        --input_dir $inputDir `
        --result_dir $resultDir `
        --weights $weightsPath `
        --save_images
    
    Write-Host "模型 $number 测试完成。" -ForegroundColor Green
    Write-Host ""
}

Write-Host "所有测试已全部完成！"



