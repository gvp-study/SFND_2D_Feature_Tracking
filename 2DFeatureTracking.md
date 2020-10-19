# SFND 2D Feature Tracking
## George V. Paul
[Github link to this project is here](https://github.com/gvp-study/SFND_2D_Feature_Tracking)


<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures.

### Task MP.1
This task is to set up a ring buffer to load the images that are needed to track the features. I implemented this using a ring buffer of size 3 as shown below.
```cpp
int dataBufferSize = 3;   
vector<DataFrame> dataBuffer;
//...
if(dataBuffer.size()+1 == dataBufferSize)
    dataBuffer.erase(dataBuffer.begin());
dataBuffer.push_back(frame);

```
### TASK MP.2
The second task is to detect keypoints in the series of given images using OpenCV methods. The type of detectors used are  SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT. I modified the detKeyPointsModern() function to create the appropriate OpenCV detector and then use it to find the features in it as shown below.

```cpp
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
else if(detectorType.compare("BRISK") == 0)
{
    detector = cv::BRISK::create();
    detector->detect(img, keypoints);
}
else if(detectorType.compare("ORB") == 0)
{
    detector = cv::ORB::create();
    detector->detect(img, keypoints);
}
else if(detectorType.compare("AKAZE") == 0)
{
    detector = cv::AKAZE::create();
    detector->detect(img, keypoints);   
}
else if(detectorType.compare("SIFT") == 0)
{
    detector = cv::SIFT::create();
    detector->detect(img, keypoints);        
}
else
{
    throw invalid_argument(detectorType +
         " is not a valid detectorType. Try FAST, BRISK, ORB, AKAZE, SIFT.");
}

...

```
### TASK MP.3
The third part of the project is to limit the features detected by the detect keypoints function to a bounding box enclosing only the preceding vehicle in the same lane.

I used the OpenCV Rectdatatype for the bounding box with the following parameters : cx = 535, cy = 180, w = 180, h = 150. All keypoints found to lie outside this box are rejected from further processing.

```cpp
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
The fourth task is to implement a variety of keypoint descriptors in addition to the already implemented BRISK method . These methods (BRIEF, ORB, FREAK, AKAZE and SIFT) are chosen based on the descriptor_types string.

I also set the selectorType to use the BINARY or HOG variations of the match based on the descriptor type.

I changed the descKeypoints() function to call the OpenCV function to create the appropriate descriptor and compute it as shown below.
```cpp
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
  if (descriptorType.compare("BRISK") == 0)
  {
      int threshold = 30;        // FAST/AGAST detection threshold score.
      int octaves = 3;           // detection octaves (use 0 to do single scale)
      float patternScale = 1.0f; // apply this scale to the pattern used for sampling
                           // the neighbourhood of a keypoint.

      extractor = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if(descriptorType.compare("ORB") == 0)
  {
      extractor = cv::ORB::create();
  }
  else if(descriptorType.compare("AKAZE") == 0)
  {
      extractor = cv::AKAZE::create();
  }
  else if(descriptorType.compare("SIFT") == 0)
  {
      extractor = cv::SIFT::create();
  }
  else if(descriptorType.compare("FREAK") == 0)
  {
      extractor = cv::xfeatures2d::FREAK::create();
  }
  else if(descriptorType.compare("BRIEF") == 0)
  {
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
//...
}
```
### TASK MP.5
The fifth task focuses on the matching part. I modified the matchDescriptors function to add FLANN as an alternative to brute-force (BF) for the matcher. After the matches are found some of the matches can be selected using the standard best match approach or the K-Nearest-Neighbor (KNN) approach as shown below. The KNN method seems to be useful in eliminating about 50-60 matches and reducing the computation.

Note: An OpenCV bug workaround had to be used to overcome an issue with the cv::Mat::type() during this step.
```cpp
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
      if (descSource.type() != CV_32F || descRef.type() != CV_32F)
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
The sixth task is to implement the descriptor distance ratio test as a filtering method to remove bad keypoint matches. This is done by setting the minDescDistRatio to 0.8 and checking the match distances between features in successive frames.
```cpp
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
The result is in this file [detectors.csv](./detectors.csv).
The summary below shows that the FAST detector detects the maximum number of features while the HARRIS finds the least number of features.

| Detector | No of Matches |
|----------|--------------:|
| SHITOMASI |	111-125 |
| HARRIS | 14-43 |
| FAST | 386-427 |
| BRISK	| 254-297 |
| ORB	| 92-130 |
| AKAZE	| 155-179 |
| SIFT	| 124-159 |

### TASK MP.8
The eighth task is to count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. The result is a table of 35 combinations as shown in this file [descriptors.csv](./descriptors.csv). The FAST detector still seems to find the maximum number of matched features between images.

| Rank | Detector+Descriptor | No of Matches |
|------|:-------------------:|------|
| 1 | FAST+SIFT| 260 |
| 2 | FAST+BRIEF	| 230 |
| 3 | FAST+BRISK	| 221 |

The code to do the combinations of detector+descriptors is shown below. I used two nested loops with one for the detectors and the other for the descriptors as shown below.

Exception was made for the AKAZE detector which did not seem to work with any other detector.
```cpp
vector<string> detector_types = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
vector<string> descriptor_types = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

for(auto detector_type:detector_types)
{
//...
  for(auto descriptor_type:descriptor_types)
  {
//...
  }
}
```

### TASK MP.9
The ninth task is to log the time it takes for all combinations of the keypoint detection and descriptor extraction. The results is a table of 35 combinations and is in this file [performance.csv](./performance.csv). Based on this analysis, I would recommend these three combination of detector/descriptors in this order.

| Rank | Detector+Descriptor | Time |
|------|:-------------------:|------|
| 1 | FAST+BRIEF| 10.68 ms |
| 2 | ORB+BRIEF	| 10.69 ms |
| 3 | FAST+ORB	| 11.37 ms |
