# ***Vehicle Detection Project***



The goal of this project is to write a software pipeline to detect and track vehicles
in a stream of video images.

The following steps were chosen to achieve this goal: 

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* A color transform and binned color features, as well as histograms of color were also extracted and appended to the HOG feature vector. 
* A sliding-window technique was used to run the previously trained classifier in order to search for vehicles in images.
* A heatmap was defined to keep track of areas of detection and then a bounding box around this areas was estimated.
* Finally the software pipeline was applied to the project video.

The project is organized into two iPython Notebooks. In the notebook "train.ipynb" the training of the SVM classifier takes place and in the jupyter notebook "main_pipeline.ipynb" is where the magic happens. Namely the previously trained classifier is used to detect vehicles in a video. Furthermore useful functions are defined and imported from the python script file "useful_functions.py".

[//]: # (Image References)
[image1]: ./output_images/car_not_car2.jpg
[image2]: ./output_images/hog_feat.jpg
[image3]: ./output_images/output_bboxes.png
[image4]: ./output_images/bboxes.jpg
[image5]: ./output_images/labeled_bboxes.jpg
[video1]: ./project_solution.mp4

## Writeup and [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


#### 1. Data visualization and preparation.

The dataset is composed of two classes: "non-vehicles" and "vehicle" images with 17760 samples in total.
Since most images correspond to a time-series I decided to divide the dataset manually into a training and test set to avoid using similar pictures when testing the model.

Thus the resulting data distribution is the following: 13920 training samples (7011 "non-car" and 6909 "car" images) and 3840 test samples (1957 "non-vehicles" and 1883 "vehicle" examples). Both training and test data were well balanced between the two classes which is important to avoid the classifier having a bias towards the majority class.

Here is an example image for each of the two classes:
![alt text][image1]

#### 2. Feature extraction


The code for this step is contained in the lines #169 through #186 of the file called `useful_functions.py`. The defined function "get_hog_features()" makes use of skimage.hog() corresponding to the skimage library.

In the second cell of the iPython Notebook "train.ipynb" an example image of the hog feature extraction is generated (see picture below) with previous conversion to the YCrCb color space. 

![alt text][image2]

Before settling on the hog parameters different color spaces, orientations and other parameters were explored.

The ones that got me best results (especially on classifier accuracy) were the following:

* orientations = 12
* pixels per cell = 16
* cells per block = 2
* color space = YCrCb 

Apart of the hog feature extraction the feature vector was extended with spatially binned color and color histograms. The main function for feature extraction "extract_features()" is defined in the file useful_functions.py (line #209) and called from the iPython notebook "train.ipynb" (cell no. 4). 


#### 3. Training the linear SVM

Training takes place in the fourth cell of the "train.ipynb" Notebook. I experimented with different classifiers and while a linear SVM classifier was the quickest to access for detections I had the most success (accuracy) when training with gridsearch(), which searches for the best classifier accross a range of parameters. In my case I searched across a linear and rbf kernel and the range 1 to 10 for the C parameter with gridsearch.

The classifier scored 99% precision and 99% recall on the test set. After training the classifier was saved to a pickle file.

Note: the pickle file I used for the  project video solution is not included in Github because it was larger than 100MB and could not be uploaded.


#### 4. Main Pipeline

The main pipeline takes place in the notebook "main_pipeline.ipynb". First the pipeline was tried on images and then on the video.

For this purpose the function find_cars() defined in line #9 of the file "useful_functions.py" is used, which basically only extracts hog features once and then sub-samples to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. 


Here is the an example on two different test images with the scale 1.5:

![alt text][image4]

Finally a heatmap was created of the pixels inside of the detected areas and a bounding box was later estimated around them with help of the function scipy.ndimage.measurements.label() (see cells no. 2 and 3 and image below).


![alt text][image5]

I timed my pipeline, which tooked 2.27 seconds to run, which is a bit far from real-time capability (I tried another pipeline with another classifier with only took 0.3 seconds, but this one produced better results on the video).

The function apply_threshold() is called upon but not really used (parameter of zero). This function is supposed to help with false detections, but because my pipeline is already taking 2,27 seconds to run I decided to use one scale only and rely on the robustness of the classifier.

The alternative approach is to make redundant detections with several window scales and then filtering out the false positives with apply_threshold().


---

#### 5. Video Implementation


Finally the described software pipeline was applied to the project video (see fourth cell and following in the notebook):


Here's a [link to my video result](./project_solution.mp4)

---

### Discussion

I am satisfied with the precision of my vehicle tracking pipeline. One thing to improve upon is a couple of random false positives ocurring in the video solution.

The reason for this is that I settled for one scale only (mainly because of speed efficiency as stated above) and hence I decided not to threshold the results (which helps with false positives).

I experimented with another classifier which was much quicker when used for the pipeline (like 0.3 seconds per image), but was having trouble detecting white cars. I think this path looks promising and further experimentation with other color spaces to fix the white car issue would be interesting to try out. Also the speed gains would make it possible with some tweaks to achieve real-time capacity and also the use of several scales and thresholding for higher robustness (and elimination of false positives).

Another way of improving efficiency would be to re-arange the function find_cars() to only use the parts of its code that are explicitly needed.

Furthermore to make the classifier more robust negative mining of "non-vehicle" images would be a good thing to do, since the negative examples in the dataset are mostly  highway road images and probably would not generalize well to other test-cases. This could be observed when seeing the many false positive detections around the trees in the images when not thresholding the search to the road (approx. lower half of the images).

Finally, it  would have been really interesting to try a deep learning approach to this project with for example Faster R-CNN which needs around 0.2 seconds on high resolution images (this means that neural networks for localization and detection are getting where they need to be for real-time usability). Still I learned a lot using the traditional computer vision approach and will profit in the future of the adquired insights. 




