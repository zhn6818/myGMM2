#include <iostream>
#include <map>
#include <deque>

void insertdeque(std::map<int, std::deque<float>>& idInfo, float value)
{
    std::deque<float> tmp(128, value);
}

int main(int argc, char **argv)
{

    std::map<int, std::deque<float>> idInfo;

    return 0;
}