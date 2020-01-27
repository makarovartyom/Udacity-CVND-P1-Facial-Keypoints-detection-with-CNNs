# Facial Keypoint Detection

## Project Overview

First project from Udacity Computer Vision [Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
The goal is to build a complete pipeline for facial keypoint detection in images, combining computer vision and deep learning techniques.
The completed code performs:

 * Face detection in image sample (detector based on Haar Cascade);
 * Cropping of detected face;
 * Allocation of **68 facial keypoints**, including points around eyes, nose and mouth.
 <br>
 

<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/images/landmarks_numbered.jpg" width=450, height=400 align="center"/>


## Results

Results of the model are represented below. 
Since the most of training dataset consists of front-face samples,  the model performs higher error on profile faces. 

Upcoming improvements will include: 
 
 - Enlarge training and test datasets in order to remove bias between front and profile face samples. 
 - Modify the model's architecture in order to reduce the error. 

<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img1.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img2.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img3.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img4.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img5.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img6.png" width=200, height=150 align="center"/>
<img src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img7.png" width=200, height=150 align="center"/>
<img
src="https://github.com/MakarovArtyom/Udacity-CVND-P1-Facial-Keypoints-detection-with-CNNs/blob/master/model_results/img8.png" width=200, height=150 align="center"/>
