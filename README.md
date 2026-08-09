# Multi-objective Black-box Patch Attacks on Face Verification

# Setup
1. Download Required Files
   ```cmd
    bash download_required_files.sh
    ``` 
You can change the download path in download_required_files.sh if needed.
2. Clone the Repository, Navigate to the Source Directory and Install the requirement packages.

    ```cmd
    cd Multi-Objective-Face-Patch-Attack
    pip install -r requirements.txt
    ``` 

# Reproducing the results
- You can reproduce the results in our paper by runing the following scripts:

    1. Attack by using Genetic Algorithm with `combined-objective` approach.
        ```cmd
        python main.py --baseline GA --fitness_type normal --seed 42 --max_iter 10 --terminated_condition generation --pair_path <pair annotation path> --img_dir <LFW dataset directory>  --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```
    2. Attack by using Genetic Algorithm with `reconstruction-bias` approach.
        ```cmd
        python main.py --baseline GA --fitness_type adaptive --seed 42 --max_iter 10 --terminated_condition generation --pair_path <pair annotation path> --img_dir <LFW dataset directory>  --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```

    3. Attack by using Genetic Algorithm with `attack-bias`.
        ```cmd
        python main.py --baseline GA_rules --fitness_type normal --seed 42 --max_iter 10 --terminated_condition generation --pair_path <pair annotation path> --img_dir <LFW dataset directory>  --mask_dir <mask path> --model_dir <pre-trained model directory>
        ```

- More details of parameters can be found in `main.py`

