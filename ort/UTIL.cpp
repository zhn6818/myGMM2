//
// Created by 张海宁 on 2020/8/19.
//


#include "UTIL.h"
//#include "TF_API.h"




bool HN_UTIL::IsSupportPos(const std::string& posfix,const std::string& support)
{
    std::string str(";");
    str.append(posfix).append(";");

    if(support.find(str)!=std::string::npos)
    {
        return true;
    }
    return false;
}
std::string HN_UTIL::GetFilePosfix(const char* path)
{
    const char* pos = strrchr(path,'.');
    if(pos)
    {
        std::string str(pos+1);
        //1.转换为小写
        //http://blog.csdn.net/infoworld/article/details/29872869
        std::transform(str.begin(),str.end(),str.begin(),::tolower);
        return str;
    }
    return std::string();
}

HN_UTIL::VecString HN_UTIL::GetListFiles(const std::string& path, const std::string& exten, bool isContainPath)
{
    std::vector<std::string> list;
    list.clear();

    DIR* dp = nullptr;
    struct dirent* dirp = nullptr;
    if ((dp = opendir(path.c_str())) == nullptr) {
        return list;
    }

    while ((dirp = readdir(dp)) != nullptr) {
        if (dirp->d_type == DT_REG) {
            if (exten.compare("*") == 0 && IsSupportPos(GetFilePosfix(dirp->d_name), HN_UTIL::POSFIX))
            {
                if(isContainPath) {
                    list.emplace_back(static_cast<std::string>(HN_UTIL::ConcatString(path, dirp->d_name)));
                }
                else {
                    list.emplace_back(static_cast<std::string>(dirp->d_name));
                }
            }
            else if (std::string(dirp->d_name).find(exten) != std::string::npos && IsSupportPos(GetFilePosfix(dirp->d_name), HN_UTIL::POSFIX))
            {
                if(isContainPath)
                {
                    list.emplace_back(static_cast<std::string>(HN_UTIL::ConcatString(path, dirp->d_name)));
                }
                else
                {
                    list.emplace_back(static_cast<std::string>(dirp->d_name));
                }
            }
        }
    }
    closedir(dp);
    return list;
}

const std::string HN_UTIL::ConcatString(const std::string &path, const std::string &file) {

    return std::string(path + prefix + file);
}

//TF_Tensor* HN_UTIL::Mat2Tensor(cv::Mat &img, float normal) {
//    const std::vector<std::int64_t> input_dims = { 1, img.size().height, img.size().width, img.channels() };
//
//    // Convert to float 32 and do normalize ops
//    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()));
//    img.convertTo(fake_mat, CV_32FC(img.channels()));
//    fake_mat *= normal;
//
//    TF_Tensor* image_input = TFUtils::CreateTensor(TF_FLOAT,
//                                                   input_dims.data(), input_dims.size(),
//                                                   fake_mat.data, (fake_mat.size().height * fake_mat.size().width * fake_mat.channels() * sizeof(float)));
//
//    return image_input;
//
//}

int HN_UTIL::ArgMax(const std::vector<float> result)
{
    float max_value = -1.0;
    int max_index = -1;
    const long count = result.size();
    for (int i = 0; i < count; ++i) {
        const float value = result[i];
        if (value > max_value) {
            max_index = i;
            max_value = value;
        }
        std::cout << "value[" << i << "] = " << value << std::endl;
    }
    return max_index;
}
