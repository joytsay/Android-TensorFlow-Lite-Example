#include <jni.h>
#include <string>
#include <fstream>

#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"

#include "tnnFR.h"

#include <android/bitmap.h>
#include <android/log.h>

#ifndef LOG_TAG
#define LOG_TAG "GV_TNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
	delete TNNFR::extractor;
}

/* ======================================[ TNNFR ]======================================*/
extern "C"
JNIEXPORT void JNICALL
Java_com_wzt_tnn_model_TNNFR_init(JNIEnv *env, jclass clazz, jstring proto, jstring model, jstring path, jboolean use_gpu) {

}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_wzt_tnn_model_TNNFR_detect(JNIEnv *env, jclass clazz, jobject bitmap, jbyteArray image_bytes, jint width,
                                     jint height, jdouble threshold, jdouble nms_threshold) {

}



extern "C"
JNIEXPORT void JNICALL
Java_tw_com_geovision_geoengine_tnn_tnnFR_init(JNIEnv *env, jclass clazz, jstring proto,
                                               jstring model, jstring path, jboolean use_gpu) {
    if (TNNFR::extractor != nullptr) {
        delete TNNFR::extractor;
        TNNFR::extractor = nullptr;
    }
    if (TNNFR::extractor == nullptr) {
        std::string parentPath = env->GetStringUTFChars(path, 0);
        std::string protoPathStr = parentPath + env->GetStringUTFChars(proto, 0);
        std::string modelPathStr = parentPath + env->GetStringUTFChars(model, 0);
        TNNFR::extractor = new TNNFR(protoPathStr, modelPathStr, use_gpu);
    }
}extern "C"
JNIEXPORT jfloatArray JNICALL
Java_tw_com_geovision_geoengine_tnn_tnnFR_run(JNIEnv *env, jclass clazz, jobject bitmap,
                                              jbyteArray image_bytes, jint width, jint height) {
    LOGD("TNNFR::run 14");
    auto result = TNNFR::extractor->run(env, bitmap);
    LOGD("TNNFR::run 15");
    jfloatArray ret;
    int size = 512;
    LOGD("TNNFR::run 16");
    ret = (*env).NewFloatArray(size);
    if (ret == NULL) {
        return NULL; /* out of memory error thrown */
    }

    // fill a temp structure to use to populate the java int array
    jfloat fill[size];
    for (int i = 0; i < size; i++) {
        fill[i] = result[i]; // put whatever logic you want to populate the values here.
    }
    // move from the temp structure to the java structure
    (*env).SetFloatArrayRegion(ret, 0, size, fill);
    return ret;
    LOGD("TNNFR::run 17");
    return ret;
}