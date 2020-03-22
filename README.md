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