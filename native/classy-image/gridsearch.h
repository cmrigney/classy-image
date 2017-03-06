#pragma once
#include "stdafx.h"

#define toStringProp(a) string(#a) + string(": ") + to_string(a) + string("\n")

typedef struct _GridSearchParams
{
  int boostType;
  int boostWeakCount;
  bool usePriors;
  float prior;

  int numSpatialBins;
  int deltaRadius;
  int numOrientationBins;
  bool normalize;
  bool gaussianFilter;
  int gaussianKernel;
  int sigma;
  bool varFeature;
  int varSplit;

  std::string toString()
  {
    return string() +
      toStringProp(boostType) +
      toStringProp(boostWeakCount) +
      toStringProp(usePriors) +
      toStringProp(prior) +
      toStringProp(numSpatialBins) +
      toStringProp(deltaRadius) +
      toStringProp(numOrientationBins) +
      toStringProp(normalize) +
      toStringProp(gaussianFilter) +
      toStringProp(gaussianKernel) +
      toStringProp(sigma) +
      toStringProp(varFeature) +
      toStringProp(varSplit);
  }
} GridSearchParams;

typedef struct _GridSearchResponse
{
  float trainAccuracy;
  float testAccuracy;
  float validationAccuracy;

  std::string toString()
  {
    return string() +
      toStringProp(trainAccuracy) +
      toStringProp(testAccuracy) +
      toStringProp(validationAccuracy);
  }
} GridSearchResponse;

static std::vector<GridSearchParams> getParams()
{
  return std::vector<GridSearchParams>({
    { cv::ml::Boost::REAL, 500, false, 0, 6, 4, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::DISCRETE, 500, false, 0, 6, 4, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::GENTLE, 500, false, 0, 6, 4, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, true, 0.5, 6, 4, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 4, 6, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 3, 7, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 4, 6, 9, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 4, 6, 9, true, true, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 3, 6, 9, true, false, 1, 1, true, 13 },
    { cv::ml::Boost::DISCRETE, 500, false, 0, 4, 6, 13, true, false, 1, 1, true, 16 },
    { cv::ml::Boost::REAL, 500, false, 0, 4, 5, 9, true, false, 1, 1, true, 13 },
  });
}

#define FORPARAM(a,b) for(int i##a = 0; i##a < a.size(); i##a++) { b }
#define GP(a) a[i##a]
#define DOCHECKS() if(GP(gaussianFilter) == false && igaussianKernel > 0) continue; \
                   if(GP(varFeature) == false && ivarSplit > 0) continue; \
                   if(GP(numSpatialBins) * GP(deltaRadius) > 24) continue;
#define ADDPARAM(a) a.push_back({ GP(boostType), GP(boostWeakCount), GP(usePriors), GP(prior), \
                                  GP(numSpatialBins), GP(deltaRadius), GP(numOrientationBins), GP(normalize), \
                                  GP(gaussianFilter), GP(gaussianKernel), GP(sigma), GP(varFeature), GP(varSplit) })

static std::vector<GridSearchParams> generateParams(
std::vector<int> boostType,
std::vector<int> boostWeakCount,
std::vector<bool> usePriors,
std::vector<float> prior,
std::vector<int> numSpatialBins,
std::vector<int> deltaRadius,
std::vector<int> numOrientationBins,
std::vector<bool> normalize,
std::vector<bool> gaussianFilter,
std::vector<int> gaussianKernel,
std::vector<int> sigma,
std::vector<bool> varFeature,
std::vector<int> varSplit
)
{
  std::vector<GridSearchParams> results;

  FORPARAM(boostType,
  FORPARAM(boostWeakCount,
  FORPARAM(usePriors,
  FORPARAM(prior,
  FORPARAM(numSpatialBins,
  FORPARAM(deltaRadius,
  FORPARAM(numOrientationBins,
  FORPARAM(normalize,
  FORPARAM(gaussianFilter,
  FORPARAM(gaussianKernel,
  FORPARAM(sigma,
  FORPARAM(varFeature,
  FORPARAM(varSplit,
    DOCHECKS()
    ADDPARAM(results);
  )))))))))))))

  return results;
}


static std::vector<GridSearchParams> getParamsCombo()
{
  return generateParams(
  { cv::ml::Boost::REAL, cv::ml::Boost::DISCRETE, cv::ml::Boost::GENTLE },
  { 100 },
  { false },
  { 0.1f },
  { 12, 6, 5, 4, 3 },
  { 2, 4, 5, 6, 8 },
  { 9, 13 },
  { true, false },
  { true, false },
  { 1, 3 },
  { 1, 2 },
  { true, false },
  { 16, 32, 9 }
  );
}
