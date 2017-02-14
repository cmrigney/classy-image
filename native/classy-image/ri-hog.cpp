#include "stdafx.h"
#include "ri-hog.h"

using namespace std;
using namespace cv;

template<class T>
static void vectorconcat(vector<T> &a, vector<T> &b)
{
  for (int i = 0; i < b.size(); i++)
  {
    a.push_back(b[i]);
  }
}

RIHOG::RIHOG(int numSpatialBins,
  int deltaRadius,
  int numOrientationBins,
  bool normalize,
  float normalizeThreshold,
  bool gaussianFilter,
  int gaussianKernelSize,
  int sigma,
  bool varFeature,
  int varSplit)
{
  _numSpatialBins = numSpatialBins;
  _deltaRadius = deltaRadius;
  _numOrientationBins = numOrientationBins;
  _normalize = normalize;
  _normalizeThreshold = normalizeThreshold;
  _gaussianFilter = gaussianFilter;
  _gaussianKernelSize = gaussianKernelSize;
  _sigma = sigma;
  _varFeature = varFeature;
  _varSplit = varSplit;
  _buildLut(64);
}

void RIHOG::_buildLut(int resolution)
{
  _lut.clear();
  for (int r = 0; r < resolution; r++)
  {
    float y = float(r) / float(resolution - 1);
    float x = sqrt(1.f - y*y);
    _lut.push_back(x);
  }
}

vector<float> RIHOG::processImage(const char *filename, bool drawRegions)
{
  Mat image = imread(filename);
  cvtColor(image, image, CV_BGR2GRAY);
  image.convertTo(image, CV_32F);
  PreprocessedData pre;
  preprocessData(image, pre);
  return processData(image, drawRegions, pre);
}

Point2i RIHOG::_getCirclePoint(int y, int radius)
{
  if (radius == 0)
    return Point2i(0, 0);

  int resolution = int(_lut.size());
  float nY = (float(y) / float(radius));
  float rY = nY * float(resolution - 1);
  if (rY >= resolution)
    return Point2i(0, 0);
  float nX = _lut[int(rY)];
  return Point2i(int(nX * radius), y);
}

void RIHOG::_binValues(vector<float> &bins, vector<vector<float> > &varHist, PixelInfo info)
{
  float g = info.g;
  float theta = info.theta + 90 / _numOrientationBins;
  float vectorAngle = info.vectorAngle;
  float idx = float(int(theta + 360) % 180) / 180.f;
  idx *= bins.size();
  idx -= 0.5f;
  int lowerIdx = int(idx) % int(bins.size());
  int higherIdx = int(idx + 1.f) % int(bins.size());
  if (lowerIdx == higherIdx) {
    bins[lowerIdx] += g;
    return;
  }
  float lowerPercent = 1.f - float((int(abs(idx - float(lowerIdx)) * 1000) % 1000))/1000.f;
  float higherPercent = 1.f - float((int(abs(float(higherIdx) - idx) * 1000) % 1000)) / 1000.f;

  bins[lowerIdx] += g * lowerPercent;
  bins[higherIdx] += g * higherPercent;

  float varIdxf = float(int(vectorAngle + 180) % 360) / 360.f;
  varIdxf *= _varSplit;
  int varIdx = int(varIdxf);
  varHist[lowerIdx][varIdx] += g * lowerPercent;
  varHist[higherIdx][varIdx] += g * higherPercent;
}

PixelInfo RIHOG::_handlePixel(int cx, int cy, int x, int y, Mat &data, Mat &drawing, bool drawRegions, PreprocessedData &pre)
{
  if (drawRegions)
    drawing.at<float>(cy + y, cx + x) /= 2.f;

  float vectorAngle = atan2(y, x) * 180.f / CV_PI;
  int lowerY = abs(x) / 2;
  int upperY = 2 * abs(x);
  int absY = abs(y);
  Point2i dir;
  if (absY >= lowerY && absY <= upperY)
    dir = Point2i(1, 1);
  else if (absY < lowerY)
    dir = Point2i(1, 0);
  else
    dir = Point2i(0, 1);

  if (x < 0)
    dir.x *= -1;
  if (y < 0)
    dir.y *= -1;

  int px = cx + x;
  int py = cy + y;

  if (px <= 0 || py <= 0 || px >= data.size().width - 1 || py >= data.size().height - 1)
  {
    return PixelInfo{ 0, 0, vectorAngle };
  }

  float g, theta;

  if (!pre.empty)
  {

    if (dir.y != 0 && dir.x == 0)
    {
      Vec3f pt = pre.straight.at<Vec3f>(py, px);
      g = pt[0];
      theta = pt[1];
    }
    else if (dir.y == 0 && dir.x != 0)
    {
      Vec3f pt = pre.straight.at<Vec3f>(py, px);
      g = pt[0];
      theta = pt[2];
    }
    else //diagonal
    {
      Vec3f pt = pre.diagonal.at<Vec3f>(py, px);
      g = pt[0];
      if (dir.x == dir.y) // ex: 1, 1 would be top-right or bottom-left vector (vertical case)
        theta = pt[1];
      else
        theta = pt[2];
    }

  }
  else
  {

    Point2i gyl(-dir.y, dir.x),
      gyr(dir.y, -dir.x),
      gxl(dir.x, dir.y),
      gxr(-dir.x, -dir.y);

    float gx = data.at<float>(py + gxr.y, px + gxr.x) * -1 + data.at<float>(py + gxl.y, px + gxl.x);
    float gy = data.at<float>(py + gyl.y, px + gyl.x) * -1 + data.at<float>(py + gyr.y, px + gyr.x);

    if (gx == 0)
      gx = 1;
    if (gy == 0)
      gy = 1;

    float g2 = sqrt(gx*gx + gy*gy);
    float theta2 = atan(gy / gx) * 180. / CV_PI;
    if (abs(g - g2) > 0.00001f || abs(theta - theta2) > 0.00001f)
      return PixelInfo{ g, theta, vectorAngle };
  }

  if (drawRegions)
    drawing.at<float>(py, px) = g;

  return PixelInfo{ g, theta, vectorAngle };
}

static inline float calcVariance(vector<float> &v)
{
  float sum = accumulate(v.begin(), v.end(), 0.0f);
  float mean = sum / v.size();

  vector<float> diff(v.size());
  transform(v.begin(), v.end(), diff.begin(), [mean](float x) { return x - mean; });
  float sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.f);
  return sq_sum / v.size();
}

vector<float> RIHOG::_normalizeFeatures(vector<float> &features)
{
  vector<float> normFeatures;
  for (int spatialBin = 0; spatialBin < _numSpatialBins; spatialBin++)
  {
    int startFeatureIdx = max((spatialBin - 1) * _numOrientationBins, 0);
    int endFeatureIdx = min((spatialBin + 1) * _numOrientationBins, int(features.size()));
    auto first = features.begin() + startFeatureIdx;
    auto last = features.begin() + endFeatureIdx;
    vector<float> workingFeatures(first, last);
    normalize(workingFeatures, workingFeatures, 1, 0, NORM_L2);
    for (int i = 0; i < workingFeatures.size(); i++)
    {
      if (workingFeatures[i] > _normalizeThreshold)
        workingFeatures[i] = _normalizeThreshold;
    }
    normalize(workingFeatures, workingFeatures, 1, 0, NORM_L2);
    vectorconcat(normFeatures, workingFeatures);
  }
  return normFeatures;
}

void RIHOG::preprocessData(Mat &image, PreprocessedData &data)
{
  if (data.straight.empty())
  {
    data.straight.create(image.size(), CV_32FC3);
    data.straight.setTo(0);
  }
  if (data.diagonal.empty())
  {
    data.diagonal.create(image.size(), CV_32FC3);
    data.diagonal.setTo(0);
  }

  data.empty = false;

  for (int i = 1; i < image.rows - 1; i++)
  {
    Vec3f *dstStraight = data.straight.ptr<Vec3f>(i) + 1;
    Vec3f *dstDiagnonal = data.diagonal.ptr<Vec3f>(i) + 1;

    float *p = image.ptr<float>(i) + 1;
    float *pt = image.ptr<float>(i - 1) + 1;
    float *pb = image.ptr<float>(i + 1) + 1;

    for (int j = 1; j < image.cols - 1; j++)
    {
      float *pl = p - 1;
      float *pr = p + 1;
      float gx = (*pl) * -1 + (*pr);
      float gy = (*pt) * -1 + (*pb);

      if (gx == 0)
        gx = 1;
      if (gy == 0)
        gy = 1;

      float g = sqrt(gx*gx + gy*gy);
      float thetaY = atan(gy / gx) * 180. / CV_PI;
      float thetaX = atan(gx / gy) * 180. / CV_PI;

      (*dstStraight)[0] = g;
      (*dstStraight)[1] = thetaY;
      (*dstStraight)[2] = thetaX;

      float *dpl = pt - 1;
      float *dpr = pb + 1;
      float *dpt = pt + 1;
      float *dpb = pb - 1;

      gx = (*dpl) * -1 + (*dpr);
      gy = (*dpb) * -1 + (*dpt);

      if (gx == 0)
        gx = 1;
      if (gy == 0)
        gy = 1;

      g = sqrt(gx*gx + gy*gy);
      thetaY = atan(gy / gx) * 180. / CV_PI;
      thetaX = atan(gx / gy) * 180. / CV_PI;

      (*dstDiagnonal)[0] = g;
      (*dstDiagnonal)[1] = thetaY;
      (*dstDiagnonal)[2] = thetaX;

      p++;
      pt++;
      pb++;
      dstStraight++;
      dstDiagnonal++;
    }
  }
}

vector<float> RIHOG::processData(Mat &image, bool drawRegions, PreprocessedData &preproc)
{
  Mat data = image.clone();
  //int padding = 100;
  //data.create(image.rows + 2 * padding, image.cols + 2 * padding, image.type());
  //data.setTo(Scalar::all(0));

  //image.copyTo(data(Rect(padding, padding, image.cols, image.rows)));

  if (_gaussianFilter)
    GaussianBlur(data, data, Size(_gaussianKernelSize, _gaussianKernelSize), _sigma, _sigma);

  Mat drawing = data.clone();
  vector<float> features;

  int width = data.size().width;
  int height = data.size().height;

  int cx = width / 2;
  int cy = width / 2;

  vector<float> varFeatures;

  for (int spatialBin = 0; spatialBin < _numSpatialBins; spatialBin++)
  {
    int radius = _deltaRadius * (spatialBin + 1);
    vector<float> bins = vector<float>(_numOrientationBins, 0);
    vector<vector<float> > varHist = vector<vector<float> >(_numOrientationBins, vector<float>(_varSplit, 0));
    for (int y = 0; y < radius; y++)
    {
      Point2i lp = _getCirclePoint(y, radius);
      Point2i stopPoint = _getCirclePoint(y, radius - _deltaRadius);
      for (int x = stopPoint.x; x < lp.x + 1; x++)
      {
        _binValues(bins, varHist, _handlePixel(cx, cy, x, y, data, drawing, drawRegions, preproc));
        if (x != 0)
          _binValues(bins, varHist, _handlePixel(cx, cy, -x, y, data, drawing, drawRegions, preproc));
        if (y != 0)
          _binValues(bins, varHist, _handlePixel(cx, cy, x, -y, data, drawing, drawRegions, preproc));
        if (x != 0 && y != 0)
          _binValues(bins, varHist, _handlePixel(cx, cy, -x, -y, data, drawing, drawRegions, preproc));
        if (drawRegions && x == stopPoint.x)
          drawing.at<float>(cy + y, cx + x) = 255;
      }
    }

    vectorconcat<float>(features, bins);
    
    if (_varFeature)
    {
      for (int j = 0; j < _numOrientationBins; j++)
      {
        vector<float> *vh = &varHist[j];
        float variance = calcVariance(varHist[j]);
        varFeatures.push_back(variance);
        float a = 360.f / _varSplit;
        float mx = 0;
        float my = 0; 
        for (int i = 0; i < _varSplit; i++)
        {
          float rad = (a*float(i + 1) - a/2.f) * CV_PI / 180.f;
          Point2f vec = Point2f(float(radius) * sin(rad), float(radius) * cos(rad));
          mx += vh->at(i) * vec.x;
          my += vh->at(i) * vec.y;
        }
        varFeatures.push_back(mx*mx + my*my);
      }
    }
  }

  //normalize
  if (_normalize)
    features = _normalizeFeatures(features);

  if (_varFeature)
    vectorconcat<float>(features, varFeatures);

  return features;
}
