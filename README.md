<h2>Domain adaptation for segmentation of microscopy images using CycleGAN and YeaZ.</h2>

This repository combines the style transfer capabilities of *CycleGAN* and segmentation capabilities of *YeaZ*,
to boost the segmentation performance on out-of-domain microscopy data.

Paper: *Pan-microscope image segmentation based on a single training set*

*main.py* script performs:
1.  style transfer on GT images in opt.dataroot
2.  performs segmentation on the style transferred images, 
3.  evaluates metrics on the segmented images and GT masks (from opt.dataroot).

**Ground truth data folder structure:**
```
    GT_DATA_FOLDER
    ├── testA
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── testB
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── testA_masks
    │   ├── 1_mask.h5
    │   ├── 2_mask.h5
    │   └── ...
    └── testB_masks
        ├── 1_mask.h5
        ├── 2_mask.h5
        └── ...
```
Depending on usage, testA(_masks) and testB(_masks) can be empty.

<h2>Usage</h2>
Script arguments follow the established nomenclature from two combined projects (CUT and YeaZ)

```
    $ python main.py \
        --dataroot GT_DATA_FOLDER \
        --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \
        --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \
        --model cycle_gan \
        --preprocess none \
        --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS (i.e. ./yeaz/unet/weights_budding_BF.pt) \
        --threshold 0.5 \
        --min_seed_dist 3 \
        --min_epoch 1 \
        --max_epoch 201 \
        --epoch_step 5 \
        --results_dir RESULTS_FOLDER (i.e. D:/GAN_grid_search/results)
        --metrics_path METRICS_PATH (i.e. D:/GAN_grid_search/results/metrics.csv)

    other options:
        --original_domain A (default) or B (i.e. if GT images are in B domain, specify B)
        --skip_style_transfer (i.e. if style transfer has already been performed, skip)
        --skip_segmentation (i.e. if segmentation has already been performed, skip)
        --skip_metrics (i.e. if metrics have already been evaluated, skip)
        --metrics_patch_borders METRICS_PATCH_BORDERS (i.e. 480 736 620 876)
        --plot_metrics
```