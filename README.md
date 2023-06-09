# K-means Clustering

## Introduction 
This project implemented K-means algorithm and applied to image segmentation and handwritten digits recognition. 

## General info
- Language: Python 
- Dataset: 
    - Image segmentation: custom images 
    - Handwritten digits recognition: MNIST dataset
    
## Repository's structure
- `main.py`: implementation of the algorithm. 
- `test.ipynb`: test the algorithm on the custom dataset and compare the result to Scikit-learn's benchmark. 
- `image_segmentation.ipynb`: applied the algorithm to segment RBG(A) images. 
- `mnist_classification.ipynb`: applied the algorithm to cluster and classify handwritten digits on MNIST dataset.
- `util.py`: helper functions. 
- `requirements.txt`: essential packages for the project. 
- `images`: result images achieved by using the algorithm to image segmentation and digits recognition. 
    
## Details 
1. Image segmentation 

Here are some result images obtained from using the algorithm for image segmentation: 

![image4](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/seg_img7.jpg)

![image 1](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/seg_img1.jpg?raw=true)

![image 2](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/seg_img3.jpg?raw=true)

![image3](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/seg_img4.jpg?raw=true)

2. Handwritten digits recognition 

Here are the digit clusters that the algorithm found for the corresponding 28 x 28 and 8 x 8 images:

![28x28](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/digit_centers_28x28.jpg)

![8x8](https://github.com/haongnd2280/K-means-Clustering/blob/main/images/digit_centers_8x8.jpg)
