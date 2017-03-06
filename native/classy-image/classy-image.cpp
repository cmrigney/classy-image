// classy-image.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "helpers.h"
#include "ri-hog.h"
#include "ThreadPool.h"
#include "Distortion.h"
#include "gridsearch.h"

using namespace cv;
using namespace ml;
using namespace std;

const int blockSize = 64;
const int halfBlockSize = blockSize / 2;

#ifndef NDEBUG
#define STARTOVER 0
#else
#define STARTOVER 0
#endif

GridSearchResponse runSearch(GridSearchParams params, bool resize, string xmlName)
{
  RIHOG c = RIHOG(params.numSpatialBins, params.deltaRadius, params.numOrientationBins, params.normalize, 0.2f, params.gaussianFilter, params.gaussianKernel, params.sigma, params.varFeature, params.varSplit);
  
  vector<string> pos;
  GetFilesInDirectory(pos, "../../data/pos");
  vector<string> neg;
  GetFilesInDirectory(neg, "../../data/neg");

  std::cout << "Processing " << pos.size() << " positives." << endl;
  std::cout << "Processing " << neg.size() << " negatives." << endl;

  vector<vector<float> > X(pos.size() + neg.size());
  vector<int> y(pos.size() + neg.size());

  //auto x = pool.partition(pos.size(), 4, [&](int start, int end) {
  int start = 0;
  int end = pos.size();
    for (int i = start; i < end; i++)
    {
      X[i] = c.processImage(pos[i].c_str());
      y[i] = 1;
    }
  //});

  //x.wait();

  int negStart = pos.size();

  //x = pool.partition(neg.size(), 4, [&](int start, int end) {
  start = 0;
  end = neg.size();
    for (int i = start; i < end; i++)
    {
      X[i + negStart] = c.processImage(neg[i].c_str());
      y[i + negStart] = 0;
    }
  //});

  //x.wait();

  cv::Mat Xm;
  vectorVectorToMat<float>(X, Xm);

  Mat ym(y.size(), 1, CV_32S, &y[0]);

  shuffleRows(Xm.clone(), ym.clone(), Xm, ym);

  Mat Xtrain, Xtest, ytrain, ytest, Xsplit, ysplit, Xvalidation, yvalidation;

  trainTestSplit(Xm, ym, 0.6, Xtrain, ytrain, Xsplit, ysplit);

  trainTestSplit(Xsplit, ysplit, 0.5, Xtest, ytest, Xvalidation, yvalidation);

  Ptr<Boost> clf = Boost::create();

  clf->setBoostType(params.boostType);
  clf->setWeakCount(params.boostWeakCount);
  if (params.usePriors)
  {
    float priors[] = { 1, params.prior };
    clf->setPriors(Mat(2, 1, CV_32F, priors));
  }

  clf->train(Xtrain, ROW_SAMPLE, ytrain);

  clf->save("test.xml");

  cout << "Saved xml" << endl;

  auto runPredict = [&clf](const char *title, Mat &samples, Mat &responses)
  {
    Mat predictions(responses.size(), 1, CV_32S);
    clf->predict(samples, predictions);
    predictions.convertTo(predictions, CV_32S);

    Mat incorrect(responses.size(), 1, CV_32S);
    absdiff(responses, predictions, incorrect);

    float wrongCount = sum(incorrect)[0];

    float rightCount = responses.size().height - wrongCount;
    cout << title << " accuracy: " << setprecision(4) << (rightCount / responses.size().height) * 100 << endl;
    Mat diff = predictions - responses;
    Mat falsePositives, falseNegatives;
    diff.convertTo(falsePositives, CV_8U);
    diff *= -1;
    diff.convertTo(falseNegatives, CV_8U);

    cout << "False positives: " << sum(falsePositives)[0] << endl;
    cout << "False negatives: " << sum(falseNegatives)[0] << endl;

    return rightCount / responses.size().height;
  };

  return GridSearchResponse{
    runPredict("Train", Xtrain, ytrain),
    runPredict("Test", Xtest, ytest),
    runPredict("Validation", Xvalidation, yvalidation)
  };
}

void doGridSearch()
{
  vector<GridSearchParams> params = getParamsCombo();
  ThreadPool pool(3);
  mutex m;

  GridSearchParams *best = NULL;
  int bestIdx = 0;
  GridSearchResponse response;

  auto x = pool.partition(params.size(), 3, [&](int start, int end)
  {
    for (int i = start; i < end; i++)
    {
      GridSearchParams *p = &params[i];
      auto r = runSearch(*p, false, string("test") + to_string(i) + string(".xml"));
      lock_guard<mutex> lock(m);
      if (best == NULL || response.testAccuracy < r.testAccuracy)
      {
        best = p;
        response = r;
        bestIdx = i;
      }
    }
  });

  x.wait();

  cout << "\n\mGot best: \nIndex: " << bestIdx << "\n" << best->toString() << "\n" << response.toString();

  //for (int i = 0; i < params.size(); i++)
  //{
  //  auto r = runSearch(p, true);
  //}
}

int main()
{
  initFixer();

  doGridSearch();

  while(true)
    system("pause");
  return 0;
}

int main2()
{
  initFixer();

  RIHOG c = RIHOG(6, 4/2, 13, true, 0.2f, false, 1, 1, true, 16);
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<float> elapsed_seconds;

#if STARTOVER
  start = std::chrono::system_clock::now();

  vector<string> pos;
  GetFilesInDirectory(pos, "../../data/pos");
  vector<string> neg;
  GetFilesInDirectory(neg, "../../data/neg");

  cout << "Processing " << pos.size() << " positives." << endl;
  cout << "Processing " << neg.size() << " negatives." << endl;

  vector<vector<float> > X(pos.size() + neg.size());
  vector<int> y(pos.size() + neg.size());

  auto x = pool.partition(pos.size(), 5, [&](int start, int end) {
    for (int i = start; i < end; i++)
    {
      X[i] = c.processImage(pos[i].c_str());
      y[i] = 1;
    }
  });

  x.wait();

  int negStart = pos.size();

  x = pool.partition(neg.size(), 5, [&](int start, int end) {
    for (int i = start; i < end; i++)
    {
      X[i + negStart] = c.processImage(neg[i].c_str());
      y[i + negStart] = 0;
    }
  });

  x.wait();

  end = std::chrono::system_clock::now();

  elapsed_seconds = end - start;

  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  cv::Mat Xm;
  vectorVectorToMat<float>(X, Xm);

  Mat ym(y.size(), 1, CV_32S, &y[0]);

  shuffleRows(Xm.clone(), ym.clone(), Xm, ym);

  {
    FileStorage file("samples.dat", FileStorage::WRITE);
    file << "samples" << Xm;
    file << "responses" << ym;
  }

  Mat Xtrain, Xtest, ytrain, ytest, Xsplit, ysplit, Xvalidation, yvalidation;

  trainTestSplit(Xm, ym, 0.2, Xtrain, ytrain, Xsplit, ysplit);

  trainTestSplit(Xsplit, ysplit, 0.5, Xtest, ytest, Xvalidation, yvalidation);

  Ptr<Boost> clf = Boost::create();

  clf->setBoostType(Boost::REAL);
  clf->setWeakCount(500);
  float priors[] = { 1, 0.1 };
  clf->setPriors(Mat(2, 1, CV_32F, priors));
  
  clf->train(Xtrain, ROW_SAMPLE, ytrain);

  clf->save("test.xml");

  cout << "Saved xml" << endl;

  auto runPredict = [&clf](const char *title, Mat &samples, Mat &responses)
  {
    Mat predictions(responses.size(), 1, CV_32S);
    clf->predict(samples, predictions);
    predictions.convertTo(predictions, CV_32S);

    Mat incorrect(responses.size(), 1, CV_32S);
    absdiff(responses, predictions, incorrect);

    float wrongCount = sum(incorrect)[0];

    float rightCount = responses.size().height - wrongCount;
    cout << title << " accuracy: " << setprecision(4) << (rightCount / responses.size().height) * 100 << endl;
    Mat diff = predictions - responses;
    Mat falsePositives, falseNegatives;
    diff.convertTo(falsePositives, CV_8U);
    diff *= -1;
    diff.convertTo(falseNegatives, CV_8U);

    cout << "False positives: " << sum(falsePositives)[0] << endl;
    cout << "False negatives: " << sum(falseNegatives)[0] << endl;
  };

  runPredict("Train", Xtrain, ytrain);
  runPredict("Test", Xtest, ytest);
  runPredict("Validation", Xvalidation, yvalidation);

#else

  Ptr<Boost> clf = StatModel::load<Boost>("test.xml");

  Mat Xm, ym;
  FileStorage file("samples.dat", FileStorage::READ);
  file["samples"] >> Xm;
  file["responses"] >> ym;

  Mat Xtrain, Xtest, ytrain, ytest;

  trainTestSplit(Xm, ym, 0.2, Xtrain, ytrain, Xtest, ytest);


#endif

  auto processImage = [&](Mat &image, const char *title)
  {
    cvtColor(image, image, CV_BGR2GRAY);
    image.convertTo(image, CV_32F);
    resize(image, image, image.size() / 2);

    PreprocessedData pre;

    auto scan = [&image, &c, &clf, &pre](int x, int y) {
      Rect r(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize);
      if (r.x < 0 || r.y < 0 || r.x + r.width >= image.size().width || r.y + r.height >= image.size().height)
        return false;
      Mat roi = image(r);
      auto X = c.processData(roi, false, pre(r));
      vector<float> vec;
      transform(X.begin(), X.end(), back_inserter(vec), [](float d) { return float(d); });
      float val = clf->predict(vec);
      if (val >= 0.5f)
      {
        //cout << "Found x: " << x << " y: " << y << endl;
        return true;
      }
      return false;
    };

    cout << "Starting scan..." << endl;
    start = std::chrono::system_clock::now();

    c.preprocessData(image, pre);

    vector<Mat> layers;
    mutex m;

    auto imageSize = image.size();
    int begin = 0;
    int last = imageSize.height;
    //auto x2 = pool.partition(image.size().height, 3, [&scan, &imageSize, &m, &layers](int begin, int last) {
    Mat b = Mat::zeros(imageSize, CV_8UC3);
    int imageWidth = imageSize.width;
    for (int y = begin; y < last; y += 2)
    {
      for (int x = 0; x < imageWidth; x += 2)
      {
        bool found = scan(x, y);
        if (found)
        {
          rectangle(b, Rect(x - 1, y - 1, 2, 2), Scalar(0, 0, 255));
        }
      }
    }
    //lock_guard<mutex> guard(m);
    layers.push_back(b);
    //});

    //x2.wait();

    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    Mat drawing;
    Mat f;
    image.convertTo(f, CV_32F);
    cvtColor(f, drawing, COLOR_GRAY2RGB);
    drawing.convertTo(drawing, CV_8UC3);
    for (int i = 0; i < layers.size(); i++)
      bitwise_or(drawing, layers[i], drawing);

    imshow(title, drawing);
  };

  auto scanImage = [&](const char *filename)
  {
    Mat image = imread(filename);
    processImage(image, filename);
  };

  auto scanVideo = [&](const char *filename, int skip=0)
  {
    VideoCapture cap = VideoCapture(filename);
    if (!cap.isOpened())
      throw runtime_error("Video not open");

    Mat image;
    while (cap.grab() && cap.retrieve(image))
    {
      fixDistortion(image);
      if(--skip < 0)
        processImage(image, "video");
      waitKey(1);
    }
  };

  scanImage("../../data/test/test7.png");
  scanImage("../../data/test/test6.png");
  waitKey(-1);

  scanVideo("../../data/test/test.mp4", 950);

  while(true)
    system("pause");
  return 0;
}

