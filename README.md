# CS 8674: Project Course - GANs with Text
Master's Project course towards MSCS at Northeastern

## Environment setup
 - Copy `sample.env` to `.env` and enter appropriate values for the variables
 - A brief description of each is provided as a comment in that file
 - Post that run,
   ```bash
   ./setup_env.sh
   ```
 - Uses env file to configure project environment
 - Builds required docker images
 - Makes a python environment and installes required packages in it
 - Prepares an `lock.env` file. Do not edit/ delete it

## Rebuilding environment
 - You may change environment config in the process of development
 - This includes adding a new python package to requirements.txt
 - After changing run,
    ```
    ./setup_env.sh
    ```

## Results
 - We compare the performance of one question answering model on the generated questions with its performance on original questions
 - The original SQuAD eval script for v1.1 was used

|        Source       |   F1   | Exact Match |
|:-------------------:|:------:|:-----------:|
|  Original questions | 84.65% |    77.07%   |
| Predicted Questions | 66.09% |    55.85%   |