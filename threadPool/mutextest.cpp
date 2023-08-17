#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <random>
#include <ctime>

std::mutex g_mutex;
int g_count = 0;


void Counter()
{
    g_mutex.lock();
    // srand(time(nullptr));
    int i = ++g_count;
    int nn = rand() % 10;
    sleep(nn);
    std::cout << "count: " << i << "  sleep: " << nn << std::endl;

    // 前面代码如有异常，unlock 就调不到了。
    g_mutex.unlock();
}

int main()
{
    const std::size_t SIZE = 6;

    // 创建一组线程。
    std::vector<std::thread> v;
    v.reserve(SIZE);

    for (std::size_t i = 0; i < SIZE; ++i)
    {
        v.emplace_back(&Counter);
    }

    // 等待所有线程结束。
    for (std::thread &t : v)
    {
        t.join();
    }

    return 0;
}
