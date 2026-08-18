# HiPA: Hill Climbing for Patch Attack on Face Verification Models

# Setup
1. Download Required Files
   ```cmd
   cd Multi-Objective-Face-Patch-Attack
   bash download_required_files.sh
   ``` 
You can change the download path in download_required_files.sh if needed.

2. Clone the Repository, Navigate to the Source Directory and Install the requirement packages.
    ```cmd
    pip install -r requirements.txt
    ``` 

# Reproducing the results
- You can reproduce the results in our paper by running the following scripts:

    1. Attack by using Genetic Algorithm with `combined-objective` approach.
        ```cmd
        python main_evo.py --baseline GA --fitness_type normal --seed 42 --max_query 10000 --terminated_condition query --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```
    2. Attack by using Genetic Algorithm with `reconstruction-bias` approach.
        ```cmd
        python main_evo.py --baseline GA --fitness_type adaptive --seed 42 --max_query 10000 --terminated_condition query --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```
    3. Attack by using Genetic Algorithm with `attack-bias`.
        ```cmd
        python main_evo.py --baseline GA_rules --fitness_type normal --seed 42 --max_query 10000 --terminated_condition query --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```
    4. Attack by using HiPA
       ```cmd
       python main_ours.py --seed 42 --max_query 10000 --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
       ```
# Ablation Study
1. HiPA but randomly sampling location at step 1
   ```cmd
   python main_ours.py --step1_random --seed 42 --max_query 10000 --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
   ```
2. HiPA but random search at step 1
   ```cmd
   python main_ours.py --step2_random --seed 42 --max_query 10000 --victim_model_name webface --img_dir <LFW dataset directory> --mask_dir <mask path> --model_dir <pre-trained model directory>
   ```
[//]: # (    4. Attack by using NSGA-2)

[//]: # (        ```cmd)

[//]: # (        python main.py --baseline NSGAII --fitness_type normal --seed 42 --max_query 10000 --terminated_condition query --img_dir <LFW dataset directory>  --mask_dir <mask path> --model_dir <pre-trained model directory>)

[//]: # (        ```)


- More details of parameters can be found in `main_evo.py` and `main_ours.py`

