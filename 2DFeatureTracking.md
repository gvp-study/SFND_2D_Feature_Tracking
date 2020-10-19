# SFND 2D Feature Tracking
## George V. Paul
https://github.com/gvp-study/SFND_2D_Feature_Tracking


<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures.

### Task MP.1
This task is to set up a ring buffer to load the images that are needed to track the features. I implemented this using a ring buffer of size 3 as shown below.
```
int dataBufferSize = 3;   
vector<DataFrame> dataBuffer;
//...
if(dataBuffer.size()+1 == dataBufferSize)
    dataBuffer.erase(dataBuffer.begin());
dataBuffer.push_back(frame);

```
### TASK MP.2
The second task is to detect keypoints in the series of given images using OpenCV methods. The type of detectors used are  SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT. I modified the detKeyPointsModern() function to create the appropriate OpenCV detector and then use it to find the features in it as shown below.

```
vector<string> detector_types = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
...
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
//...
cv::Ptr<cv::Feature2D> detector;

if(detectorType.compare("FAST") == 0)
{
   detector = cv::FastFeatureDetector::create();
   detector->detect(img, keypoints);
}
...

```
### TASK MP.3
The third part of the project is to limit the features detected by the detect keypoints function to a bounding box enclosing the preceding vehicle.

I used the OpenCV Rectdatatype for the bounding box with the following parameters : cx = 535, cy = 180, w = 180, h = 150.

```
cv::Rect vehicleRect(535, 180, 180, 150);
vector<cv::KeyPoint> keypoints_vehicle;

if (bFocusOnVehicle)
{
    for(auto keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
    {
      if (vehicleRect.contains(keypoint->pt))
      {
        cv::KeyPoint vehiclePoint(keypoint->pt, 1);
        keypoints_vehicle.push_back(vehiclePoint);
      }
    }
    keypoints =  keypoints_vehicle;
    cout << "Vehicle has " << keypoints.size()<<" keypoints"<<endl;
    // ...
}

```
### TASK MP.4
The fourth task is to implement a variety of keypoint descriptors to the already implemented BRISK methods. These methods (BRIEF, ORB, FREAK, AKAZE and SIFT) are chosen based on the descriptor_types string. I changed the descKeypoints() function to call the OpenCV function to create the appropriate descriptor and compute it as shown below.
```
vector<string> descriptor_types = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
//...
string descriptorType2 = "DES_BINARY"; // DES_BINARY, DES_HOG
if(descriptorType.compare("SIFT") == 0)
     descriptorType2 = "DES_HOG";
//...
matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
				     (dataBuffer.end() - 2)->descriptors,
             (dataBuffer.end() - 1)->descriptors,
				     matches, descriptorType2, matcherType, selectorType);
//...
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors,
		   string descriptorType)
{
  cv::Ptr<cv::DescriptorExtractor> extractor;
//...
  else if(descriptorType.compare("ORB") == 0)
  {
    extractor = cv::ORB::create();
  }
//...
  extractor->compute(img, keypoints, descriptors);
//...
}
```
### TASK MP.5
The fifth task focuses on the matching part. I modified the matchDescriptors function to add FLANN as an alternative to brute-force as well as the K-Nearest-Neighbor approach. The matche is created with the appropriate OpenCV functions and then matched with the selectorType as shown below.
```
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
		      cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType,
		      std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (matcherType.compare("MAT_BF") == 0)
    {
      int normType = cv::NORM_HAMMING;
      matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
      if (descSource.type() != CV_32F)
      {
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
      }
      matcher = cv::FlannBasedMatcher::create();              
    }
    // perform matching task
     if (selectorType.compare("SEL_NN") == 0)
     { // nearest neighbor (best match)
         // Finds the best match for each descriptor in desc1
         matcher->match(descSource, descRef, matches);
     }
     else if (selectorType.compare("SEL_KNN") == 0)
     { // k nearest neighbors (k=2)

         // ...add start: MP.6 Descriptor Distance Ratio
         vector<vector<cv::DMatch>> knn_matches;
         matcher->knnMatch(descSource, descRef, knn_matches, 2);

         double minDescDistRatio = 0.8;
         for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
         {
             if( ((*it)[0].distance) < ((*it)[1].distance * minDescDistRatio) )
             {
                 matches.push_back((*it)[0]);
             }                
         }
         cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
     }
 }

```
### TASK MP.6
The sixth task is to implement the descriptor distance ratio test as a filtering method to remove bad keypoint matches. This is done in by setting the minDescDistRation to 0.8 and checking the match distances.
```
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)

    // ...add start: MP.6 Descriptor Distance Ratio
    vector<vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, 2);

    double minDescDistRatio = 0.8;
    for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
        if( ((*it)[0].distance) < ((*it)[1].distance * minDescDistRatio) )
        {
            matches.push_back((*it)[0]);
        }                
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
}
```
### TASK MP.7
The seventh task is to count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. This is done for all the detectors in the list.
The result is in this file [detectors.csv](<file://detectors.csv>).
The summary below shows that the FAST detector detects the maximum number of features while the HARRIS finds the least number of features.

SHITOMASI	111-125

HARRIS 14-43

FAST 386-427

BRISK	254-297

ORB	92-130

AKAZE	155-179

SIFT	124-159

### TASK MP.8
The eighth task is to count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. The result is shown in this file [descriptors.csv](<file://descriptors.csv>).

### TASK MP.9
The ninth task is to log the time it takes for keypoint detection and descriptor extraction. The results of the data collected is in this file [performance.csv](<file://performance.csv>). Based on this analysis, I would recommend these three combination of detector/descriptors in this order.
1. FAST+BRIEF	6.85
2. FAST+ORB	7.90
3. ORB+BRIEF	9.73