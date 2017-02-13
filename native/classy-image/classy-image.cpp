// classy-image.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "helpers.h"
#include "ri-hog.h"
#include "ThreadPool.h"

using namespace cv;
using namespace ml;
using namespace std;

const int blockSize = 64;
const int halfBlockSize = blockSize / 2;

#ifndef NDEBUG
#define STARTOVER 0
#else
#define STARTOVER 1
#endif


int main()
{
  ThreadPool pool(3);
  RIHOG c;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

#if STARTOVER
  start = std::chrono::system_clock::now();

  vector<string> pos;
  GetFilesInDirectory(pos, "../../data/pos");
  vector<string> neg;
  GetFilesInDirectory(neg, "../../data/neg");

  vector<vector<float> > X(pos.size() + neg.size());
  vector<int> y(pos.size() + neg.size());

  auto x = pool.partition(pos.size(), 3, [&](int start, int end) {
    for (int i = start; i < end; i++)
    {
      X[i] = c.processImage(pos[i].c_str());
      y[i] = 1;
    }
  });

  x.wait();

  int negStart = pos.size();

  x = pool.partition(neg.size(), 3, [&](int start, int end) {
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

  Mat Xtrain, Xtest, ytrain, ytest;

  trainTestSplit(Xm, ym, 0.2, Xtrain, ytrain, Xtest, ytest);

  Ptr<Boost> clf = Boost::create();

  clf->setBoostType(Boost::REAL);
  clf->setWeakCount(1000);
  float priors[] = { 1, 0.01 };
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

    double wrongCount = sum(incorrect)[0];

    double rightCount = responses.size().height - wrongCount;
    cout << title << " accuracy: " << setprecision(4) << (rightCount / responses.size().height) * 100 << endl;
  };

  runPredict("Train", Xtrain, ytrain);
  runPredict("Test", Xtest, ytest);

#else

  Ptr<Boost> clf = StatModel::load<Boost>("test.xml");

  Mat Xm, ym;
  FileStorage file("samples.dat", FileStorage::READ);
  file["samples"] >> Xm;
  file["responses"] >> ym;

  Mat Xtrain, Xtest, ytrain, ytest;

  trainTestSplit(Xm, ym, 0.2, Xtrain, ytrain, Xtest, ytest);


#endif

  Mat image = imread("../../data/test/test7.png");
  cvtColor(image, image, CV_BGR2GRAY);
  image.convertTo(image, CV_32F);

  auto scan = [&image, &c, &clf](int x, int y) {
    Rect r(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize);
    if (r.x < 0 || r.y < 0 || r.x + r.width >= image.size().width || r.y + r.height >= image.size().height)
      return false;
    Mat roi = image(r);
    auto X = c.processData(roi);
    float val = clf->predict(X);
    if (val >= 0.5f)
    {
      //cout << "Found x: " << x << " y: " << y << endl;
      return true;
    }
    return false;
  };

  cout << "Starting scan..." << endl;
  start = std::chrono::system_clock::now();

  vector<Mat> layers;
  mutex m;

  auto imageSize = image.size();
  auto x2 = pool.partition(image.size().area(), 2, [&scan, &imageSize, &m, &layers](int start, int end) {
    Mat b = Mat::zeros(imageSize, CV_8UC3);
    int imageWidth = imageSize.width;
    for (int i = start; i < start + (end - start)/2; i++) {
      int x = (i*2 - start) % imageWidth;
      int y = (i*2 - start) / imageWidth;
      bool found = scan(x, y);
      if (found)
      {
        rectangle(b, Rect(x - 1, y - 1, 2, 2), Scalar(0, 0, 255));
      }
    }
    lock_guard<mutex> guard(m);
    layers.push_back(b);
  });

  x2.wait();

  end = std::chrono::system_clock::now();

  elapsed_seconds = end - start;

  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  Mat drawing;
  cvtColor(image, drawing, COLOR_GRAY2RGB);
  drawing.convertTo(drawing, CV_8UC3);
  for (int i = 0; i < layers.size(); i++)
    bitwise_or(drawing, layers[i], drawing);
  
  imshow("test", drawing);
  waitKey(-1);

  system("pause");
  return 0;
}

