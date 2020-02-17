# Semantic-Segmentation-Pytorch-

This is a Semantic Segmentation Project

Object detection is a computer vision and image processing related problem that incorporates detecting
instances of semantic objects of a particular class of objects. Object detection has various applications
in many areas and is still a wildly growing research field. To pursue object detection, there are
various methodologies. But most of them fall into either machine learning or deep learning based 
approaches. We aim to detect objects using Semantic Segmentation, which involves classifying each and every pixel 
in an input image into a class. With Semantic Segmentation we can detect and classify a variable number of regions, 
each of arbitrary size. It is a very important task for self-driving cars, so that they can detect and recognize 
objects in the environment to navigate safely.

In our project we investigate multiple neural networks architectures based on convolutional neural networks. 
This includes a simple Encoder-Decoder backbone architecture, a more complex architecture, U-Net model and transfer learning.


Simulation runs are saved in 2 logfiles: 
logfile.txt - real time logfile
logfile_summary.txt - summary log file


To run the different experiments, run the following files:
starter.ipynb - baseline and data augmentation baseline
srtunet.ipynb - unet experiment
startTransfer.ipynb - transfer learning experiment
startCust.ipynb
