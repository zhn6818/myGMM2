//
// Created by 张海宁 on 2020/8/19.
//

#include "Onnx.h"

std::string toString(const ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    {
        return "float";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    {
        return "uint8_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    {
        return "int8_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    {
        return "uint16_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    {
        return "int16_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    {
        return "int32_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    {
        return "int64_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    {
        return "string";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    {
        return "bool";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    {
        return "float16";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    {
        return "double";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    {
        return "uint32_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    {
        return "uint64_t";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    {
        return "complex with float32 real and imaginary components";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    {
        return "complex with float64 real and imaginary components";
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    {
        return "complex with float64 real and imaginary components";
    }
    default:
        return "undefined";
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os,
                         const ONNXTensorElementDataType &type)
{
    switch (type)
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        os << "undefined";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        os << "float";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        os << "uint8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        os << "int8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        os << "uint16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        os << "int16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        os << "int32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        os << "int64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        os << "std::string";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        os << "bool";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        os << "float16";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        os << "double";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        os << "uint32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        os << "uint64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        os << "float real + float imaginary";
        break;
    case ONNXTensorElementDataType::
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        os << "double real + float imaginary";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        os << "bfloat16";
        break;
    default:
        break;
    }

    return os;
}
void MyOnnxPack::normalize(Mat &img)
{
    img.convertTo(img, CV_32F);
    int i = 0, j = 0;
    for (i = 0; i < img.rows; i++)
    {
        float *pdata = (float *)(img.data + i * img.step);
        for (j = 0; j < img.cols; j++)
        {
            pdata[0] = (pdata[0] - this->MEANS[0]) / this->STD[0];
            pdata[1] = (pdata[1] - this->MEANS[1]) / this->STD[1];
            pdata[2] = (pdata[2] - this->MEANS[2]) / this->STD[2];
            pdata += 3;
        }
    }
}

MyOnnxPack::MyOnnxPack(std::string modelpath, std::vector<std::string> &vecString)
{
    this->strModelPath = modelpath;
    this->vecImglist = vecString;
    this->out_shape = {1, HEIGHT, WIDTH, 2};
    mPredict = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));
    InitializeOnnxEnv();
    input_tensor_value = std::vector<float>(WIDTH * HEIGHT * CHANNEL);
    PrintfInputInfo();
    std::cout << "Construct Over! " << std::endl;
}

MyOnnxPack::MyOnnxPack(std::string modelpath, std::string testImg)
{
    this->strModelPath = modelpath;
    this->strTestImg = testImg;
    this->out_shape = {1, HEIGHT, WIDTH, 2};
    mPredict = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));
    InitializeOnnxEnv();
    input_tensor_value = std::vector<float>(WIDTH * HEIGHT * CHANNEL);
    PrintfInputInfo();
    std::cout << "Construct Over! " << std::endl;

    // std::cout << "inputImg size:  " << input_image.size() << std::endl;
}

void MyOnnxPack::InitializeOnnxEnv()
{
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    this->sessionOptions = new Ort::SessionOptions();
    this->sessionOptions->SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options;
    OrtTensorRTProviderOptions tensorRTOptions{};
    tensorRTOptions.trt_max_workspace_size = 2UL << 30;
    tensorRTOptions.trt_max_partition_iterations = 1000;
    tensorRTOptions.trt_min_subgraph_size = 1;
    tensorRTOptions.trt_fp16_enable =
        static_cast<int>(false);
    tensorRTOptions.trt_int8_enable =
        static_cast<int>(false);
    tensorRTOptions.trt_engine_cache_path =
        "/data1/code/";
    tensorRTOptions.trt_int8_calibration_table_name =
        "table";
    this->sessionOptions->AppendExecutionProvider_TensorRT(tensorRTOptions);
    sessionOptions->AppendExecutionProvider_CUDA(cuda_options);

    this->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*this->env, this->strModelPath.c_str(), *sessionOptions);
    this->allocator = new Ort::AllocatorWithDefaultOptions();

    Ort::AllocatorWithDefaultOptions allocator;

    numInputNodes = session->GetInputCount();
    numOutputNodes = session->GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char *inputName = session->GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char *outputName = session->GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    // ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    // std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    Ort::TypeInfo outputTypeInfo1 = session->GetOutputTypeInfo(1);
    auto outputTensorInfo1 = outputTypeInfo1.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims1 = outputTensorInfo1.GetShape();
    std::cout << "Output Dimensions: " << outputDims1 << std::endl;

    Ort::TypeInfo outputTypeInfo2 = session->GetOutputTypeInfo(2);
    auto outputTensorInfo2 = outputTypeInfo2.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims2 = outputTensorInfo2.GetShape();
    std::cout << "Output Dimensions: " << outputDims2 << std::endl;

    Ort::TypeInfo outputTypeInfo3 = session->GetOutputTypeInfo(3);
    auto outputTensorInfo3 = outputTypeInfo3.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims3 = outputTensorInfo3.GetShape();
    std::cout << "Output Dimensions: " << outputDims3 << std::endl;

    std::cout << std::endl;
}

MyOnnxPack::~MyOnnxPack()
{
    if (this->sessionOptions != nullptr)
    {
        delete (sessionOptions);
    }
    if (this->session != nullptr)
    {
        delete (session);
    }
    if (this->allocator != nullptr)
    {
        delete (allocator);
    }
    if (this->env != nullptr)
    {
        delete (env);
    }
}

void MyOnnxPack::PrintfInputInfo()
{
    size_t num_input_nodes = this->session->GetInputCount();
    //    std::cout << "Number of inputs " << num_input_nodes << std::endl;
    input_nodes_names = std::vector<const char *>(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++)
    {
        char *input_name = this->session->GetInputName(i, *allocator);
        // std::cout << "Input " << i << " : " << " name = " << input_name << std::endl;
        input_nodes_names[i] = input_name;

        Ort::TypeInfo type_info = this->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        input_node_dims = tensor_info.GetShape();

        //        for (int j = 0; j < input_node_dims.size(); j++)
        //            printf("Input %d : dim %d=%lld\n", i, j, input_node_dims[j]);
    }
}

std::vector<const char *> MyOnnxPack::GetOutPutName()
{

    size_t num_out_nodes = this->session->GetOutputCount();
    std::vector<const char *> out_vec_name(num_out_nodes);
    // std::cout << "Number of outputs " << num_out_nodes << std::endl;
    for (int i = 0; i < num_out_nodes; i++)
    {
        char *out_name = this->session->GetOutputName(i, *allocator);
        out_vec_name[i] = out_name;
    }
    return out_vec_name;
}

// cv::Mat *MyOnnxPack::GetImgFromVector(ResultType &result)
// {
//     if (result.size() <= 0)
//     {
//         return &mPredict;
//     }
//     assert(result.size() == WIDTH * HEIGHT * 2);
//     for (int i = 0; i < WIDTH * HEIGHT; i++)
//     {
//         int row = i / WIDTH;
//         int col = i % WIDTH;
//         uchar mm = saturate_cast<uchar>(result[i * 2] * 255);
//         mPredict.at<uchar>(row, col) = mm;
//     }
//     return &mPredict;
// }

void MyOnnxPack::Mat2ChannelLast(cv::Mat &src, float *p_input)
{
    assert(!src.empty());

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < CHANNEL; c++)
            {
                p_input[i * WIDTH * CHANNEL + j * CHANNEL + c] = (src.ptr<float>(i)[j * CHANNEL + c]) / 1.0;
            }
        }
    }
}

void MyOnnxPack::Mat2ChannelFirst(cv::Mat &src, float *p_input)
{
    assert(!src.empty());
    for (int c = 0; c < CHANNEL; c++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                p_input[c * src.rows * src.cols + i * src.cols + j] = (src.ptr<float>(i)[j * CHANNEL + c]) / 1.0;
            }
        }
    }
}

void MyOnnxPack::InferenceImg()
{
    assert(img.cols == WIDTH || img.rows == HEIGHT || img.channels() == CHANNEL);
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    normalize(dst);
    float *p_input = this->input_image.data();
    std::fill(input_image.begin(), input_image.end(), 1.f);
    // Mat2ChannelFirst(dst, p_input);

    output_node_names = GetOutPutName();
    // for (int i = 0; i < output_node_names.size(); i++)
    // {
    //     std::cout << i << " name: " << output_node_names[i] << std::endl;
    // }
    // assert(output_node_names.size() == 1);
    double t1 = (double)getTickCount();
    memcpy(this->input_tensor_value.data(), p_input, sizeof(float) * WIDTH * HEIGHT * CHANNEL);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_node_dims[0] = 1;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_value.data(), input_tensor_value.size(), input_node_dims.data(), 4);
    // Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, result.data(), result.size(), out_shape.data(), out_shape.size());
    assert(input_tensor.IsTensor());

    // session->Run(Ort::RunOptions(nullptr), input_nodes_names.data(), &input_tensor, 1, output_node_names.data(), &output_tensor, 1);
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, input_nodes_names.data(), &input_tensor, numInputNodes, output_node_names.data(), numOutputNodes);
    t1 = (double)getTickCount() - t1;
    std::cout << "out size: " << outputTensors.size() << std::endl;

    assert(outputTensors.size() == numOutputNodes);
    std::vector<DataOutputType> outputData;
    outputData.reserve(numOutputNodes);

    int count = 1;
    for (auto &elem : outputTensors)
    {
        std::cout << "type of input: " << count++ << " " << toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str() << "  " << elem.GetTensorTypeAndShapeInfo().GetShape() << std::endl;
        outputData.emplace_back(std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
    }
    std::cout << "value test: ";

    for (int i = 0; i < 100; i++)
    {
        std::cout << *((float *)outputData[1].first + i) << " ";
    }
    std::cout
        << "compute time :" << t1 * 1000.0 / cv::getTickFrequency() << " ms \n";

    // cv::Mat img2 = *GetImgFromVector(result);

    // std::string strPredictPath = ConstructFilePath();
    // cv::imwrite(strPredictPath, img2);
}

void MyOnnxPack::InferenceVecImg()
{
    assert(this->vecImglist.size() > 0);
    for (int i = 0; i < vecImglist.size(); i++)
    {
        ReadImg(vecImglist[i]);
        InferenceImg();
    }
}

void MyOnnxPack::ReadImg()
{
    img = cv::imread(this->strTestImg);
}

void MyOnnxPack::ReadImg(std::string path)
{
    this->strTestImg = path;
    img = cv::imread(this->strTestImg, 1);
    assert(img.channels() == CHANNEL);
}

std::string MyOnnxPack::ConstructFilePath()
{
    int pos = this->strTestImg.find_last_of(HN_UTIL::prefix);
    std::string filename = std::string(this->strTestImg.substr(pos + 1));
    std::string filepath = std::string(this->strTestImg.substr(0, pos));
    int posPoint = filename.find_last_of(".");
    std::string name = std::string(filename.substr(0, posPoint));
    std::string hPrix = std::string(filename.substr(posPoint));
    return std::string(filepath + "Result/" + name + "_predict" + hPrix);
}
