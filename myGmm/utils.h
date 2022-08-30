//
// Created by 张海宁 on 2022/8/17.
//

#ifndef ZHANG_UTILS_H
#define ZHANG_UTILS_H
#include <iostream>
#include <memory>

#define DEBUG_INFO 0
#define NGaussian 4

static void red_print(std::string out)
{
    std::cout << "\033[31;1m" << out << "\033[0m" << std::endl;
}
static void green_print(std::string out)
{
    std::cout << "\033[32;1m" << out << "\033[0m" << std::endl;
}

#endif // ZHANG_UTILS_H
