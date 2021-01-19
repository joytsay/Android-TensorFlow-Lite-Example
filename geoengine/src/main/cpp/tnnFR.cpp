#include "tnnFR.h"
#include "omp.h"
#include <fstream>
#include <android/bitmap.h>
#include <sys/time.h>

TNNFR *TNNFR::extractor = nullptr;

char *jstring2string(JNIEnv *env, jstring jstr) {
    char *rtn = nullptr;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    auto barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte *ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char *) malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}

std::string fdLoadFile(std::string path) {
    std::ifstream file(path, std::ios::in);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size = file.tellg();
        char *content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

TNNFR::TNNFR(std::string proto, std::string model, bool useGPU) {
    if (TNNFR::net == nullptr) {
//        LOGD("TNNFR init proto path: %s",proto.c_str());
//        LOGD("TNNFR init model path: %s",model.c_str());
        std::string protoContent, modelContent;
        protoContent = fdLoadFile(proto);
        modelContent = fdLoadFile(model);

        TNN_NS::Status status;
        TNN_NS::ModelConfig config;
        config.model_type = TNN_NS::MODEL_TYPE_TNN;
        config.params = {protoContent, modelContent};
        auto net = std::make_shared<TNN_NS::TNN>();
        status = net->Init(config);
        TNNFR::net = net;

        TNNFR::device_type = useGPU ? TNN_NS::DEVICE_OPENCL : TNN_NS::DEVICE_ARM;

        TNN_NS::InputShapesMap shapeMap;
        TNN_NS::NetworkConfig network_config;
        network_config.library_path = {""};
        network_config.device_type = TNNFR::device_type;
        auto ins = TNNFR::net->CreateInst(network_config, status, shapeMap);
        if (status != TNN_NS::TNN_OK || !ins) {
            LOGW("GPU initialization failed, switch to CPU");
            // 如果出现GPU加载失败，切换到CPU
            TNNFR::device_type = TNN_NS::DEVICE_ARM;
            network_config.device_type = TNN_NS::DEVICE_ARM;
            ins = TNNFR::net->CreateInst(network_config, status, shapeMap);
        }
        TNNFR::instance = ins;
        LOGD("TNNFR init model succeed return status:(%d)", (int)status);

        if (status != TNN_NS::TNN_OK) {
            LOGE("TNN init failed %d", (int) status);
            return;
        }
    }
}

TNNFR::~TNNFR() {
    TNNFR::instance = nullptr;
    TNNFR::net = nullptr;
}

std::vector<float> TNNFR::run(JNIEnv *env, jobject bitmap) {
    timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);
    LOGD("TNNFR::run start");
    std::vector<float> results;
    AndroidBitmapInfo bitmapInfo;
    LOGD("TNNFR::run 1");
    void *imageSource;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
        LOGD("TNNFR::run 1.1");
        return results;
    }
    LOGD("TNNFR::run 2");
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGD("TNNFR::run 2.1");
        return results;
    }
    LOGD("TNNFR::run 3");
    if (AndroidBitmap_lockPixels(env, bitmap, &imageSource) < 0) {
        LOGD("TNNFR::run 3.1");
        return results;
    }
    LOGD("TNNFR::run 4");
    int image_h = bitmapInfo.height;
    int image_w = bitmapInfo.width;
    LOGD("TNNFR::run 5");
    // 原始图片
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;  // 当前数据来源始终位于CPU，不需要设置成OPENCL，tnn自动复制cpu->gpu
    TNN_NS::DimsVector image_dims = {1, 4, image_h, image_w};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, image_dims, imageSource);
    LOGD("TNNFR::run 6");
    // 模型输入
    TNN_NS::DimsVector target_dims = {1, 4, net_height, net_width};
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims);
    LOGD("TNNFR::run 7");
    // OPENCL需要设置queue
//    void *command_queue = nullptr;
//    auto status = TNNFR::instance->GetCommandQueue(&command_queue);
//    if (status != TNN_NS::TNN_OK) {
//        LOGE("MatUtils::GetCommandQueue Error: %s", status.description().c_str());
//    }
//    LOGD("TNNFR::run 8");
    // 转换大小
//    TNN_NS::ResizeParam param;
//    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, command_queue);
    LOGD("TNNFR::run 9");
    // 输入数据
    TNN_NS::MatConvertParam input_cvt_param;
//    input_cvt_param.scale = {1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0};
//    input_cvt_param.bias = {0.0, 0.0, 0.0, 0.0};
    input_cvt_param.scale = { 0.0078125, 0.0078125, 0.0078125, 0 };
    input_cvt_param.bias = { -128*0.0078125, -128*0.0078125 , -128*0.0078125, 0 };
    auto status = TNNFR::instance->SetInputMat(input_mat, input_cvt_param);
    if (status != TNN_NS::TNN_OK) {
        LOGE("instance.SetInputMat Error: %s", status.description().c_str());
    }
    LOGD("TNNFR::run 10");
    // 前向
//    TNN_NS::Callback callback;
//    status = TNNFR::instance->ForwardAsync(callback);
    status = TNNFR::instance->Forward();
    if (status != TNN_NS::TNN_OK) {
        LOGE("instance.Forward Error: %s", status.description().c_str());
    }
    LOGD("TNNFR::run 11");
    // 获取数据
    std::shared_ptr<TNN_NS::Mat> output_mat;
    TNN_NS::MatConvertParam output_param;
    status = TNNFR::instance->GetOutputMat(output_mat, output_param, "embedding:0");
    if (status != TNN_NS::TNN_OK) {
        LOGE("instance.GetOutputMat Error: %s", status.description().c_str());
    }
    LOGD("===============> %d %d %d %d", output_mat->GetDims()[0], output_mat->GetDims()[1], output_mat->GetDims()[2], output_mat->GetDims()[3]);
    gettimeofday(&tv_end, NULL);
    double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
    LOGD("TNNFR::run time: %f ms", elapsed);
//    }
    // 后处理
    float * data = (float *)output_mat->GetData();

//    int c = output_mat->GetChannel();
//    int h = output_mat->GetHeight();
//    int w = output_mat->GetWidth();
//    LOGD("TNNFR::run GetChannel:%d, GetHeight:%d, GetWidth:%d", c,h,w);
//    for(int j = 0; j< c; j++){
//        for(int k = 0; k < h; k++){
//            for(int p = 0; p < w; p++){
//                LOGD("data%d:%f ",data[j*h*w + k*w + p], j*h*w + k*w + p);
//            }
//        }
//    }

    LOGD("TNNFR::run data[0,1,128,510,511]:[%f,%f,%f,%f,%f]",data[0],data[1],data[128],data[510],data[511]);
    for (int i = 0; i < 512; ++i) {
        results.push_back(data[i]);
    }
    LOGD("TNNFR::run feature[0,1,128,510,511]:[%f,%f,%f,%f,%f]",
            results.data()[0], results.data()[1], results.data()[128], results.data()[510], results.data()[511]);
    AndroidBitmap_unlockPixels(env, bitmap);
    if (status != TNN_NS::TNN_OK) {
        LOGE("get outputmat fail");
        return results;
    }
    LOGD("TNNFR::run done");
    return results;
}