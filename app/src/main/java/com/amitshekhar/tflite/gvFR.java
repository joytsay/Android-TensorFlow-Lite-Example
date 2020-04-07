package com.amitshekhar.tflite;
import org.tensorflow.lite.Interpreter;
import java.util.List;


public class gvFR {
    private Interpreter interpreter;
    public static final int SUCCESS = 0; //执行接口返回成功
    public static final int ERROR_INVALID_PARAM = -1; //非法参数
    public static final int ERROR_TOO_MANY_REQUESTS = -2; //太多请求
    public static final int ERROR_NOT_EXIST = -3; //不存在
    public static final int ERROR_FAILURE = -4; // 执行接口返回失败

    boolean CreateFR(String modelPath) {
        // load model ..., init interpreter
        return false;
    }

    boolean SetInfo(int iCmd,int value){  //, void *pData
        // set info
        return false;
    }

    int GetFeature(Image image, float[] feature, List faceinfos, int[] res) {
        return 0;
    }

    boolean ReleaseFR() {
        // release model
        return false;
    }
}
