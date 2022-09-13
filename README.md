Fast organs localization on CT
==============================

### Task 1

|                     | UNET (exp1, 128x128)  | UNETR (exp1, 128) | UNETR (exp1, 256) |
|:-------------------:|-----------------------|-------------------|-------------------|
|       spleen        | 0.3669                | 0.5756            | 0.5483            |
|    right kidney     | 0.5549                | 0.4809            | 0.5604            |
|     left kidney     | 0.5578                | 0.4645            | 0.5203            |
|     gallbladder     | 0.0070                | 0.2871            | 0.4026            |
|      esophagus      | 0.4014                | 0.3094            | 0.3653            |
|        liver        | 0.1428                | 0.6533            | 0.6937            |
|       stomach       | 0.2902                | 0.3077            | 0.3506            |
|        aorta        | 0.5905                | 0.5424            | 0.5712            |
|      postcava       | 0.3920                | 0.3731            | 0.3929            |
|      pancreas       | 0.2834                | 0.2700            | 0.3102            |
| right adrenal gland | 0.1209                | 0.2605            | 0.2879            |
| left adrenal gland  | 0.1773                | 0.2078            | 0.2230            |
|      duodenum       | 0.2497                | 0.1841            | 0.2088            |
|       bladder       | 0.2561                | 0.3902            | 0.4367            |
|   prostate/uterus   | 0.3977                | 0.4087            | 0.4463            |


### Task 2
| iou_score | f1_score | accuracy | recall |
|:---------:|----------|----------|--------|
|   0.84    | 0.91     |  0.98    | 0.91   |

### Task 3 
| iou_score | f1_score | accuracy | recall | channels |
|:---------:|----------|----------|--------|----------|
|   0.91    | 0.94     | 0.99     | 0.94   | 3        |
|   0.90    | 0.94     | 0.99     | 0.93   | 5        |
|   0.90    | 0.93     | 0.99     | 0.93   | 10        |



|                             Predictions                             |                           Targets                           |
|:-------------------------------------------------------------------:|:-----------------------------------------------------------:|
| ![reports/figures/predictions.png](reports/figures/predictions.png) | ![reports/figures/targets.png](reports/figures/targets.png) |

Project Organization
------------
    ├── README.md   
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, zipped data.
    ├── docs
    ├── models             <- Trained and serialized models
    ├── notebooks
    ├── reports   
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt 
    └──  src                <- Source code for use in this project.
       │
       ├── data           <- Scripts generate dataset
       │   └── make_dataset.py
       ├── models         <- Scripts to train models and then use trained models to make predictions
       │   ├── predict_model.py
       │   └── train_model.py
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py

--------