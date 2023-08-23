# HuMob Challenge 2023

## Information
- url : https://connection.mit.edu/humob-challenge-2023
- data description : https://arxiv.org/pdf/2307.03401.pdf
- metrics repository : https://github.com/yahoojapan/geobleu
    ```
    git submodule add git@github.com:yahoojapan/geobleu.git  # submodule or
    git clone git@github.com:yahoojapan/geobleu.git
    ```

## Script

- Movement Uids Over Time

    ```
    poetry run python -m src.dash.human_movement
    ```
- XY Trend

    ```
    poetry run python -m src.dash.xy_trend
    ```
-  Run

    ```
    sh bin/run.sh
    ```
