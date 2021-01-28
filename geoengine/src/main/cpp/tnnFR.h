#ifndef TNNFRS_H
#define TNNFRS_H

#include <android/log.h>
#include <string>
#include <vector>
#include <jni.h>

#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"

#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#ifndef LOG_TAG
#define LOG_TAG "FR_TNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif

class TNNFR {
public:
    TNNFR(std::string proto, std::string model, bool useGPU);

    ~TNNFR();

    std::vector<float> run(JNIEnv *env, jobject bitmap);

private:
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<TNN_NS::Instance> instance;
    TNN_NS::DeviceType device_type;

    int net_width = 224;
    int net_height = 224;

public:
    static TNNFR *extractor;
};

#endif //TNNFRS_H
