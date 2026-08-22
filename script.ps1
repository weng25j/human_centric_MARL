# run_eval.ps1

# 1. Define your variables here so they are easy to change
$ModelDir = "./results/YOUR_MODEL_FOLDER_NAME"  # <-- Change this to your actual trained model path
$Demands  = "800,1200,1700"
$PRs      = "0.05,0.10,0.20"


# 2. Print a status message
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Starting Traffic Shepherd Evaluation" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# 3. Execute the Python script (Change 'evaluate.py' to your actual eval script name)


python record.py --model-dir ./results_dongchen_baseline/0809/models

python record_shepherd.py --model-dir ./results/MAPPO_Curriculum_seed_0809/models



Write-Host "Evaluation Complete!" -ForegroundColor Green