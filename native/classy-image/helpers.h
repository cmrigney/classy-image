#pragma once
#ifndef _HELPERS_
#define _HELPERS_
#include "stdafx.h"

template<class T>
static void vectorVectorToMat(std::vector<std::vector<T> > X, cv::Mat &output)
{
  output.create(X.size(), X.at(0).size(), CV_32FC1);
  for (int i = 0; i<output.rows; ++i)
    for (int j = 0; j<output.cols; ++j)
      output.at<float>(i, j) = X.at(i).at(j);
}

static void trainTestSplitData(cv::Mat &data, float testRatio, cv::Mat &train, cv::Mat &test)
{
  int numTest = int(float(data.rows) * testRatio);
  int numTrain = data.rows - numTest;

  train.create(numTrain, data.cols, data.type());
  test.create(numTest, data.cols, data.type());

  data(cv::Rect(0, 0, data.cols, numTrain)).copyTo(train);
  data(cv::Rect(0, numTrain, data.cols, numTest)).copyTo(test);
}

static void trainTestSplit(cv::Mat &samples, cv::Mat &responses, float testRatio, cv::Mat &trainSamples, cv::Mat &trainResponses, cv::Mat &testSamples, cv::Mat &testResponses)
{
  trainTestSplitData(samples, testRatio, trainSamples, testSamples);
  trainTestSplitData(responses, testRatio, trainResponses, testResponses);
}

void shuffleRows(const cv::Mat &samples, const cv::Mat &responses, cv::Mat &outputSamples, cv::Mat &outputResponses)
{
  std::vector <int> seeds;
  for (int cont = 0; cont < samples.rows; cont++)
    seeds.push_back(cont);

  cv::randShuffle(seeds);

  for (int cont = 0; cont < samples.rows; cont++)
  {
    outputSamples.push_back(samples.row(seeds[cont]));
    outputResponses.push_back(responses.row(seeds[cont]));
  }
}

/* Returns a list of files in a directory (except the ones that begin with a dot) */

static void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory)
{
#ifdef _WIN32
  HANDLE dir;
  WIN32_FIND_DATAA file_data;

  if ((dir = FindFirstFileA((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
    return; /* No files found */

  do {
    const std::string file_name = file_data.cFileName;
    const std::string full_file_name = directory + "/" + file_name;
    const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

    if (file_name[0] == '.')
      continue;

    if (is_directory)
      continue;

    out.push_back(full_file_name);
  } while (FindNextFileA(dir, &file_data));

  FindClose(dir);
#else
  DIR *dir;
  class dirent *ent;
  class stat st;

  dir = opendir(directory);
  while ((ent = readdir(dir)) != NULL) {
    const string file_name = ent->d_name;
    const string full_file_name = directory + "/" + file_name;

    if (file_name[0] == '.')
      continue;

    if (stat(full_file_name.c_str(), &st) == -1)
      continue;

    const bool is_directory = (st.st_mode & S_IFDIR) != 0;

    if (is_directory)
      continue;

    out.push_back(full_file_name);
  }
  closedir(dir);
#endif
} // GetFilesInDirectory

#endif

