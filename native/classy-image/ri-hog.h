#pragma once
#ifndef _RIHOG_
#define _RIHOG_

#include "stdafx.h"

typedef struct _PixelInfo
{
  float g;
  float theta;
  float vectorAngle;
} PixelInfo;

typedef struct _PreprocessedData
{
  cv::Mat straight;
  cv::Mat diagonal;
  bool empty = true;

  _PreprocessedData() : empty(true)
  {}
  _PreprocessedData(cv::Mat &s, cv::Mat &d, bool e) : straight(s), diagonal(d), empty(e)
  {}

  _PreprocessedData operator()(const cv::Rect &r)
  {
    return _PreprocessedData(
      straight(r),
      diagonal(r),
      false
    );
  }
} PreprocessedData;

class RIHOG
{
public:
  RIHOG(int numSpatialBins = 4,
        int deltaRadius = 4,
        int numOrientationBins = 13,
        bool normalize = true, 
        float normalizeThreshold = 0.2f,
        bool gaussianFilter = true,
        int gaussianKernelSize = 1,
        int sigma = 1,
        bool varFeature = true,
        int varSplit = 8);

  void preprocessData(cv::Mat &image, PreprocessedData &data);
  std::vector<float> processImage(const char *filename, bool drawRegions = false, bool rsz = false);
  std::vector<float> processData(cv::Mat &image, bool drawRegions = false, PreprocessedData &preproc = PreprocessedData());

private:
  int _numSpatialBins;
  int _deltaRadius;
  int _numOrientationBins;
  bool _normalize;
  float _normalizeThreshold;
  bool _gaussianFilter;
  int _gaussianKernelSize;
  int _sigma;
  bool _varFeature;
  int _varSplit;

  std::vector<float> _normalizeFeatures(std::vector<float> &features);
  void _binValues(std::vector<float> &bins, std::vector<std::vector<float> > &varHist, PixelInfo info);
  PixelInfo _handlePixel(int cx, int cy, int x, int y, cv::Mat &data, cv::Mat &drawing, bool drawRegions, PreprocessedData &pre);
  cv::Point2i _getCirclePoint(int y, int radius);
  void _buildLut(int resolution);

  std::vector<float> _lut;
};

#endif

