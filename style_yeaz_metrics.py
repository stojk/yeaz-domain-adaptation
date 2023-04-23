import argparse
import os
import sys

import numpy as np

sys.path.append("./cycle_gan")

import metrics.metrics as metrics
# from cycle_gan.data import create_dataset
# from cycle_gan.models import create_model
# from cycle_gan.util import html
# from cycle_gan.util.visualizer import save_images
from options.test_options import TestOptions
from yeaz.predict import YeazPredict as yeaz_predict


def initialzie_options() -> argparse.Namespace:
    """Initialize options
    
    Style transfer options are hard-coded for test setting. 
    Metrics options are set to default values.

    Returns:
        Initialized options
    """
    # get test options
    opt = TestOptions().parse()

    ### Style transfer options ###
    # test code only supports num_threads = 1
    opt.num_threads = 0
    # test code only supports batch_size = 1
    opt.batch_size = 1 
    # disable data shuffling; commcent this line if results on randomly chosen images are needed.   
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    
    ### Metrics options ###
    # Set output metrics path if not specified
    if opt.metrics_path is None:
        opt.metrics_path = os.path.join(opt.results_dir, opt.name, 'metrics')

    return opt

def style_transfer(
    opt: argparse.Namespace, 
    epoch_range: range
) -> None:
    """Perform style transfer on images in opt.dataroot

    Arguments:
        opt: Options
        epoch_range: Range of epochs to perform style transfer on
    """
    
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)  
    
    # create a model given opt.model and other options
    model = create_model(opt)

    for epoch in epoch_range:

        opt.epoch = str(epoch)
        # create a webpage for viewing the results
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        for i, data in enumerate(dataset):
            if i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
        webpage.save()  # save the HTML

def yeaz_segmentation(
    opt: argparse.Namespace, 
    epoch_range: range, 
    style_transfer_path: str
) -> None:
    """Perform segmentation on style transferred images
    
    Arguments:
        opt: Options
        epoch_range: Range of epochs to perform segmentation on
        style_transfer_path: Path to style transferred images
    """
    for epoch in epoch_range:

        generated_images_path = os.path.join(
            style_transfer_path,'test_{}'.format(epoch),'images/fake_B')
        image_names = [
            filename for filename in os.listdir(generated_images_path) 
            if not filename.endswith('.h5')
        ]

        for image_name in image_names:
            print(generated_images_path, image_name)
            
            image_path = os.path.join(generated_images_path, image_name)
            mask_name = image_name.replace('.png','_mask.h5')
            mask_path = os.path.join(generated_images_path, mask_name)
            yeaz_predict(
                image_path=image_path,
                mask_path=mask_path,
                imaging_type=None,
                fovs=[0],
                timepoints=[0,0],
                threshold=0.5,
                min_seed_dist=opt.min_seed_dist,
                weights_path=opt.path_to_weights
            )

def yeaz_metrics(
    epoch_range: range, 
    gt_path: str, 
    style_transfer_path: str
) -> None:
    """Evaluate metrics on style transferred and segmented images

    Arguments:
        epoch_range: Range of epochs to evaluate metrics on
        gt_path: Path to ground truth images
        style_transfer_path: Path to style transferred images and masks
    
    Returns:
        Dictionary of average metrics (J, SD, Jc) on segmented style transferred images for each epoch
    """
    avg_metrics_per_epoch = {}
    for epoch in epoch_range:

        generated_images_path = os.path.join(
            style_transfer_path,'test_{}'.format(epoch),'images/fake_B')
        image_names = [
            filename for filename in os.listdir(generated_images_path) 
            if not filename.endswith('.h5')
        ]
        
        J = []
        SD = []
        Jc = []

        for image_name in image_names:
            
            # get paths
            mask_name = image_name.replace('.png','_mask.h5')
            mask_path = os.path.join(generated_images_path, mask_name)
            gt_mask_path = os.path.join(gt_path,'testA_masks', mask_name)

            # evaluate metrics
            j, sd, jc, succ = metrics.evaluate(
                gt_mask_path,
                mask_path
            )
            if not succ:
                J=SD=Jc=[-1]
                break

            J.append(j)
            SD.append(sd)
            Jc.append(jc)

        avg_metrics_per_epoch[epoch] = (
            np.mean(J), np.mean(SD), np.mean(Jc)
        )

    return avg_metrics_per_epoch

def save_metrics(
    metrics_dict: dict,
    path: str
) -> None:
    """Save metrics to CSV file

    Numpy is used to save the metrics to a CSV file, as we did not want to add dependencies on new packages.
    The CSV file is structured as follows:
        epoch,J,SD,Jc

    Arguments:
        metrics_dict: Dictionary of average metrics (J, SD, Jc) on segmented style transferred images for each epoch
        path: Path to save CSV file
    """

    # Convert metrics_per_epoch to a structured NumPy array
    metrics_arr = np.empty(
        len(metrics_dict), 
        dtype=[('epoch', int), ('J', float), ('SD', float), ('Jc', float)]
    )
    for i, (epoch, metrics) in enumerate(metrics_dict.items()):
        metrics_arr[i] = (epoch, *metrics)

    # Write to CSV
    np.savetxt(path, metrics_arr, delimiter=',',
               header='epoch,J,SD,Jc', fmt='%d,%f,%f,%f')

def main():
    """Main function that runs: style transfer -> segmentation -> metrics"""
    # initialize style transfer options
    opt = initialzie_options()

    # create a range of epochs to test
    epoch_range = range(
        opt.min_epoch, opt.max_epoch+1, opt.epoch_step)

    # run style transfer
    # style_transfer(style_opt, epoch_range)
    
    style_transfer_path = os.path.join(
        opt.results_dir, opt.name)

    # run yeaz segmentation
    yeaz_segmentation(opt, epoch_range, style_transfer_path)

    # calculate and save segmentation metrics
    metrics_per_epoch = yeaz_metrics(
        epoch_range, opt.dataroot, style_transfer_path)
    save_metrics(metrics_per_epoch, opt.metrics_path)

    print(metrics_per_epoch)


if __name__ == '__main__':
    main()
