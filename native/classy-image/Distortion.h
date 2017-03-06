#pragma once
#include "stdafx.h"

using namespace std;
using namespace cv;

Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs = Mat(1, 5, CV_32FC1);

void fixDistortion(Mat &frame)
{
  Mat orig = frame.clone();
  undistort(orig, frame, intrinsic, distCoeffs);
}

static void initFixer()
{
  intrinsic.at<float>(0, 0) = 195.5714336278603;
  intrinsic.at<float>(0, 1) = 0;
  intrinsic.at<float>(0, 2) = 178.6476786136367;
  intrinsic.at<float>(1, 0) = 0;
  intrinsic.at<float>(1, 1) = 194.5796138283209;
  intrinsic.at<float>(1, 2) = 97.27419008359774;
  intrinsic.at<float>(2, 0) = 0;
  intrinsic.at<float>(2, 1) = 0;
  intrinsic.at<float>(2, 2) = 1;


  distCoeffs.at<float>(0, 0) = -0.3484670336051714;
  distCoeffs.at<float>(0, 1) = 0.1410247684223272;
  distCoeffs.at<float>(0, 2) = 0.0004922727032282937;
  distCoeffs.at<float>(0, 3) = -0.001280684300971166;
  distCoeffs.at<float>(0, 4) = -0.02798480357488543;
}
