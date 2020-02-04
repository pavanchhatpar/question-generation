# CS 8674: Project Course - GANs with Text
Master's Project course towards MSCS at Northeastern

## Environment setup

```bash
./setup_env.sh
```

 - Creates a relocatable virtual environment that can
be mounted inside any docker container
 - Builds required docker images
 - Sets a random password for accessing jupyter at 
 `jupyter.password`
 - You may change it. After changing run,
    ```
    ./remove_images.sh
    ./build_images.sh
    ```