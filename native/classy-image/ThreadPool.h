#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include "stdafx.h"
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
  ThreadPool(size_t);
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
    ->std::future<typename std::result_of<F(Args...)>::type>;
  auto partition(int count, int split, std::function<void(int, int)>&& f);
  ~ThreadPool();
private:
  // need to keep track of threads so we can join them
  std::vector< std::thread > workers;
  // the task queue
  std::queue< std::function<void()> > tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
  : stop(false)
{
  for (size_t i = 0; i<threads; ++i)
    workers.emplace_back(
      [this]
  {
    for (;;)
    {
      std::function<void()> task;

      {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        this->condition.wait(lock,
          [this] { return this->stop || !this->tasks.empty(); });
        if (this->stop && this->tasks.empty())
          return;
        task = std::move(this->tasks.front());
        this->tasks.pop();
      }

      task();
    }
  }
  );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type>
{
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared< std::packaged_task<return_type()> >(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

auto ThreadPool::partition(int count, int split, std::function<void(int, int)>&& f)
{
  std::vector<std::future<void> > results(split);
  int chunkSize = count / split;
  for (int i = 0; i < split; i++)
  {
    int start = i * chunkSize;
    int end = (i == split - 1 ? count : (i+1)*chunkSize);
    results[i] = enqueue([](auto start, auto end, auto f) {
      f(start, end);
    }, start, end, f);
  }
  std::promise<void> p;
  auto r = p.get_future();
  auto pp = std::make_shared<std::promise<void> >(std::move(p));
  auto resultsp = std::make_shared<std::vector<std::future<void> > >(std::move(results));
  enqueue([split](std::shared_ptr<std::promise<void> > p, std::shared_ptr<std::vector<std::future<void> > > results) {
    for (int i = 0; i < split; i++)
    {
      results->at(i).wait();
    }
    p->set_value();
  }, pp, resultsp);
  return r;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers)
    worker.join();
}

#endif