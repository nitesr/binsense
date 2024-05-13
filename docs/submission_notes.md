# Submission Form

## Please share a 100-word summary of your approach for solving the problem
Based on the objective (find bin for an item quantity) & dataset, our initial thought was to find a zero-shot/one-shot object detection model. After that manually annotate the dataset and fine tune the model. There could be a need for data augmentation (as the dataset being low) which we delve into after trying out the model. It was hard at first to figure out where the issue is (code or dataset), so we tried the model on known (imagenet) dataset.  Eventually, Data Augmenation using Albumentations was used with the Dual Encoder to increased the number of entries from 10144 to 101,440 i.e. created 9 synthentic images for a total of 10 instances per image to correlated image with text.


## What were the major challenges faced during the implementation of your solution? What can be the future improvements on top of this?
The dataset is challenging due - very high class imbalance, data size is low, poor image quality, and occlusions due to tape & cluttered packing. Finding a zero-shot / one-shot model has been challenging which have a good generalization weights to fine tune upon. The model performance is poor, need to work on data augmentation and also hyper parameter search to fine tune the model. Transfomer related architectures even when fine-tuning require extensive compute resources and long training cycles outside the scope of free services e.g. Colab.

MLOPS has been challenging to work with different technologies in the given time. There were compatibility issues e.g. iterative cml configuration used an old Nvidia driver which is not compatible with torch. There is a learning curve and couldn't trade optimally between exploration and exploitation to bring all technologies (aws cloud-formation, terraform, react, cml, dvc, lightning, etc..) together.

We spent lot of time on the model research compared to later tasks considering the model is key for the usecase to get it right and fell short of time for other tasks.

Future Work: 
 - Altering the data augmentation pipeline to normalied class imbalance by product id and hyper param search
 - Currently, data augmentation generated target instances.  This is not preferred as the data augmenation should collapse to the same label e.g. remove augmentation suffix.
 - Try siamese network to understand the features well (we currently froze the VIT layer for OWL)
 - Try leveraging other item attributes to augment the features
 - End-End flow automation with very minimum manual touch points
 - Model Monitoring based on the user feedback (on UI) & also other metrics like confidence score distribution

## Any additional comments or insights to share.
README file in the zip attached should have the video recordings which will talk you through the details of work.
