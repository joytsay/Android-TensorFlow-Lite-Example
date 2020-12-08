package tw.com.geovision.geoengine;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Point;
import android.os.Environment;
import android.util.Log;

import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.lang.Math;
import java.util.UUID;
import java.util.Vector;

import tw.com.geovision.geoengine.mtcnn.Box;
import tw.com.geovision.geoengine.mtcnn.MTCNN;

import static java.lang.Math.abs;

public class GVFaceRecognition {
    public static final String version = "v0.1.1";
    private static final String PATH_FACE_GV_MODEL = "model";
    private static final String FR_MODEL_NAME = "gvFR.tflite";
    private static final String LM_MODEL_NAME = "shape_predictor_5_face_landmarks.dat";
    private static final String MTCNN_MODEL_NAME = "mtcnn_freezed_model.pb";

    private static GVFaceRecognition gvFaceRecognitionInstance = null;

//    private static GpuDelegate delegate;
    private Interpreter interpreter;
    private final ArrayList<Classifier.Recognition> recognitions = new ArrayList<>();
    private FaceDet faceDet = null;
    private MTCNN mtcnn = null;

    public static final int SUCCESS = 0; //执行接口返回成功
    public static final int ERROR_INVALID_PARAM = -1; //非法参数
    public static final int ERROR_TOO_MANY_REQUESTS = -2; //太多请求
    public static final int ERROR_NOT_EXIST = -3; //不存在
    public static final int ERROR_FAILURE = -4; // 执行接口返回失败

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private static final int FEATURE_SIZE = 512;

    private boolean quant;
    private boolean bDoJitter;
    private int inputSize;
    private UUID randomUUID;
    private boolean bSaveDebugImage;
    private int nSaveDebugImageCnt = 0;

    public static GVFaceRecognition getInstance() {
        if (gvFaceRecognitionInstance == null) {
            gvFaceRecognitionInstance = new GVFaceRecognition();
        }
        return gvFaceRecognitionInstance;
    }

    final class FDmtcnn {
        private int status;
        private final android.graphics.Rect[] rect;
        private final android.graphics.Point[][] landmarks;
        public FDmtcnn(int status, android.graphics.Rect[] rect, android.graphics.Point[][] landmarks) {
            this.status = status;
            this.rect = rect;
            this.landmarks = landmarks;
        }
        public int getStatus() {
            return status;
        }
        public android.graphics.Rect[] getRect() {
            return rect;
        }
        public android.graphics.Point[][] getLandmark() {
            return landmarks;
        }
        public Mat getCrop(Mat src, FDmtcnn faceObj){
            // get 2 landmark points
            Box ret = null;
            org.opencv.core.Point LeftEyeCenter = new org.opencv.core.Point(ret.landmark[0].x, ret.landmark[0].y);
            org.opencv.core.Point RightEyeCenter = new org.opencv.core.Point(ret.landmark[1].x, ret.landmark[1].y);
            List<org.opencv.core.Point> landmarkPoints = new ArrayList<>();
            List<org.opencv.core.Point> MTCNNrectPoints = new ArrayList<>();
            Mat resultMat = null;
            landmarkPoints.add(LeftEyeCenter);
            landmarkPoints.add(RightEyeCenter);
            int rectLeft = 0;
            int rectTop = 0;
            int rectRight = 0;
            int rectBottom = 0;
            org.opencv.core.Point MTCNNrect0 = new org.opencv.core.Point(rectLeft, rectTop);
            MTCNNrectPoints.add(MTCNNrect0);
            org.opencv.core.Point MTCNNrect1 = new org.opencv.core.Point(rectRight, rectBottom);
            MTCNNrectPoints.add(MTCNNrect1);
            org.opencv.core.Point MTCNNrect2 = new org.opencv.core.Point(rectRight, rectTop);
            MTCNNrectPoints.add(MTCNNrect2);
            org.opencv.core.Point MTCNNrect3 = new org.opencv.core.Point(rectLeft, rectBottom);
            MTCNNrectPoints.add(MTCNNrect3);
            ////do face alignment
            long warpFRstartTime = new Date().getTime();
            resultMat = warp(src, MTCNNrectPoints, landmarkPoints);
            long warpFRendTime = new Date().getTime();
            int runTime = (int) (warpFRendTime - warpFRstartTime);
            Log.d("gvFR", "Alignment and crop face runTime: " + runTime + " ticks\n");
            return resultMat;
        }
    }

    public FDmtcnn faceDetect(Mat rgbaMat, int width, int height) {
        int maxFacesNum = 10;
        android.graphics.Rect[] rect = new android.graphics.Rect[maxFacesNum];
        android.graphics.Point[][] landmarks = new Point[maxFacesNum][];
        //MTCNN FD & LM & crop
        Bitmap croppedBitmap = Bitmap.createBitmap(rgbaMat.cols(),  rgbaMat.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(rgbaMat, croppedBitmap);
        Bitmap bm= tw.com.geovision.geoengine.mtcnn.Utils.copyBitmap(croppedBitmap);
        try {
            Vector<Box> boxes = mtcnn.detectFaces(bm,100);
//                //draw MTCNN
//                for (int i=0;i<boxes.size();i++){
//                    tw.com.geovision.geoengine.mtcnn.Utils.drawRect(bm,boxes.get(i).transform2Rect());
//                    tw.com.geovision.geoengine.mtcnn.Utils.drawPoints(bm,boxes.get(i).landmark);
//                }
            if(boxes.size() == 0){
                Log.e("MTCNN","[*]no face detected");
                return new FDmtcnn(ERROR_NOT_EXIST, rect, landmarks);
            }
            Box ret = boxes.get(0);
            int rectLeft = ret.left() < 0 ? 0 : ret.left();
            int rectTop = ret.top() < 0 ? 0 : ret.top();
            int rectRight = ret.right() > croppedBitmap.getWidth() ? croppedBitmap.getWidth() : ret.right();
            int rectBottom = ret.bottom() > croppedBitmap.getHeight() ? croppedBitmap.getHeight() : ret.bottom();
        }catch (Exception e){
            Log.e("MTCNN","[*]detect false:"+e);
            return new FDmtcnn(ERROR_FAILURE, rect, landmarks);
        }
        return new FDmtcnn(SUCCESS, rect, landmarks);
    }
    ////    FDmtcnn usage
    //    FDmtcnn result = faceDetect();
    //    System.out.println(result.getRect() + result.getLandmark());

    public void initFR(final Context context) throws IOException {
        FileUtils.copyAssetFile(context, "model", context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL, false);
    }

    public void CreateFR(final Context context) throws IOException {
        // load model
        long createFRstartTime = new Date().getTime();
        quant = false;
        bDoJitter = false;
        //GPU mode
//        delegate = new GpuDelegate();
//        Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
//        options.setAllowFp16PrecisionForFp32(true);
//        gvFaceRecognitionInstance.interpreter = new Interpreter(new File(context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + FR_MODEL_NAME), options);

        //CPU mode
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(4);
        gvFaceRecognitionInstance.interpreter = new Interpreter(new File(context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + FR_MODEL_NAME), tfliteOptions);
        gvFaceRecognitionInstance.inputSize = INPUT_SIZE;
        //FD and landmark via dlib
//        if(faceDet==null) {
//            faceDet = new FaceDet(context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + LM_MODEL_NAME);
//        }
        AssetManager asm=context.getAssets();
        if(mtcnn==null) {
            mtcnn = new MTCNN(asm,context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + MTCNN_MODEL_NAME);
        }
        long createFRendTime = new Date().getTime();
        int runTime = (int) (createFRendTime - createFRstartTime);
        Log.d("gvFR", "CreateFR runTime: " + runTime + " ticks\n");
    }

    public int GetFeature(Mat ImageMat, float[] feature, FaceInfo faceinfo, int[] res, boolean isSaveImage) {
        Bitmap resultBitmap = Bitmap.createBitmap(ImageMat.cols(),  ImageMat.rows(),Bitmap.Config.RGB_565);;
        Utils.matToBitmap(ImageMat, resultBitmap);
        List<VisionDetRet> results = null;
        List<org.opencv.core.Point> rectPoints = new ArrayList<>();
//        List<org.opencv.core.Point> dliblandmarkPoints = new ArrayList<>();
        bSaveDebugImage = isSaveImage;
        randomUUID = UUID.randomUUID();
        //do Face detection
        if(faceinfo==null) {
            //MTCNN FD & LM & crop
//            Bitmap bm= tw.com.geovision.geoengine.mtcnn.Utils.copyBitmap(resultBitmap);
//            try {
//                Vector<Box> boxes=mtcnn.detectFaces(bm,100);
////                //draw MTCNN
////                for (int i=0;i<boxes.size();i++){
////                    tw.com.geovision.geoengine.mtcnn.Utils.drawRect(bm,boxes.get(i).transform2Rect());
////                    tw.com.geovision.geoengine.mtcnn.Utils.drawPoints(bm,boxes.get(i).landmark);
////                }
//                Box ret = boxes.get(0);
//                int rectLeft = ret.left() < 0 ? 0 : ret.left();
//                int rectTop = ret.top() < 0 ? 0 : ret.top();
//                int rectRight = ret.right() > resultBitmap.getWidth() ? resultBitmap.getWidth() : ret.right();
//                int rectBottom = ret.bottom() > resultBitmap.getHeight() ? resultBitmap.getHeight() : ret.bottom();
//                // get 2 landmark points
//                org.opencv.core.Point LeftEyeCenter = new org.opencv.core.Point(ret.landmark[0].x, ret.landmark[0].y);
//                org.opencv.core.Point RightEyeCenter = new org.opencv.core.Point(ret.landmark[1].x, ret.landmark[1].y);
//                landmarkPoints.add(LeftEyeCenter);
//                landmarkPoints.add(RightEyeCenter);
//                org.opencv.core.Point MTCNNrect0 = new org.opencv.core.Point(rectLeft, rectTop);
//                MTCNNrectPoints.add(MTCNNrect0);
//                org.opencv.core.Point MTCNNrect1 = new org.opencv.core.Point(rectRight, rectBottom);
//                MTCNNrectPoints.add(MTCNNrect1);
//                org.opencv.core.Point MTCNNrect2 = new org.opencv.core.Point(rectRight, rectTop);
//                MTCNNrectPoints.add(MTCNNrect2);
//                org.opencv.core.Point MTCNNrect3 = new org.opencv.core.Point(rectLeft, rectBottom);
//                MTCNNrectPoints.add(MTCNNrect3);
//                ////do face alignment
//                long warpFRstartTime = new Date().getTime();
//                resultMat = warp(ImageMat, MTCNNrectPoints, landmarkPoints);
//                long warpFRendTime = new Date().getTime();
//                int runTime = (int) (warpFRendTime - warpFRstartTime);
//                Log.d("gvFR", "Alignment and crop face runTime: " + runTime + " ticks\n");
//            }catch (Exception e){
//                Log.e("MTCNN","[*]detect false:"+e);
//                return ERROR_FAILURE;
//            }

////            dlib FD & LM & eyes align crop deprecated
//            results = faceDet.detect(resultBitmap);
//            for (final VisionDetRet ret : results) {
//                int rectLeft = ret.getLeft() < 0 ? 0 : ret.getLeft();
//                int rectTop = ret.getTop() < 0 ? 0 : ret.getTop();
//                int rectRight = ret.getRight() > resultBitmap.getWidth() ? resultBitmap.getWidth() : ret.getRight();
//                int rectBottom = ret.getBottom() > resultBitmap.getHeight() ? resultBitmap.getHeight() : ret.getBottom();
//                // get 5 landmark points
//                ArrayList<Point> landmarks = ret.getFaceLandmarks();
//                for (Point point : landmarks) {
//                    int pointX = point.x;
//                    int pointY = point.y;
//                    org.opencv.core.Point landmark = new org.opencv.core.Point(pointX, pointY);
//                    dliblandmarkPoints.add(landmark);
//                }
//                org.opencv.core.Point dlibLeftEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(2).x + dliblandmarkPoints.get(3).x)*0.5), (int)((dliblandmarkPoints.get(2).y + dliblandmarkPoints.get(3).y)*0.5));
//                org.opencv.core.Point dlibRightEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(0).x + dliblandmarkPoints.get(1).x)*0.5), (int)((dliblandmarkPoints.get(0).y + dliblandmarkPoints.get(1).y)*0.5));
//                landmarkPoints.add(dlibLeftEyeCenter);
//                landmarkPoints.add(dlibRightEyeCenter);
//                org.opencv.core.Point rect0 = new org.opencv.core.Point(rectLeft, rectTop);
//                rectPoints.add(rect0);
//                org.opencv.core.Point rect1 = new org.opencv.core.Point(rectRight, rectBottom);
//                rectPoints.add(rect1);
//                org.opencv.core.Point rect2 = new org.opencv.core.Point(rectRight, rectTop);
//                rectPoints.add(rect2);
//                org.opencv.core.Point rect3 = new org.opencv.core.Point(rectLeft, rectBottom);
//                rectPoints.add(rect3);
//                ////do face alignment
//                long warpFRstartTime = new Date().getTime();
//                resultMat = warp(ImageMat, rectPoints, landmarkPoints);
//                long warpFRendTime = new Date().getTime();
//                int runTime = (int) (warpFRendTime - warpFRstartTime);
//                Log.d("gvFR", "Alignment and crop face runTime: " + runTime + " ticks\n");
//            }
//            if(results.size() == 0){ //no Face detected
//                return ERROR_FAILURE;
//            }
        }else{ //given Rect by user

            //get face_x1_sdk Rect
            org.opencv.core.Point rect0 = new org.opencv.core.Point(faceinfo.mRect.left, faceinfo.mRect.top);
            rectPoints.add(rect0);
            org.opencv.core.Point rect1 = new org.opencv.core.Point(faceinfo.mRect.right, faceinfo.mRect.bottom);
            rectPoints.add(rect1);
            org.opencv.core.Point rect2 = new org.opencv.core.Point(faceinfo.mRect.right, faceinfo.mRect.top);
            rectPoints.add(rect2);
            org.opencv.core.Point rect3 = new org.opencv.core.Point(faceinfo.mRect.left, faceinfo.mRect.bottom);
            rectPoints.add(rect3);
            Mat inputRectMat = ImageMat.clone();
            int thickness = 2;
            int lineType = 1;
            int shift = 0;
            Imgproc.rectangle(inputRectMat, rect0, rect1, new Scalar(0,255,0), thickness, lineType, shift);
            if(bSaveDebugImage) {
                Bitmap inputRectBitmap = Bitmap.createBitmap(inputRectMat.cols(),  inputRectMat.rows(),Bitmap.Config.RGB_565);;
                Utils.matToBitmap(inputRectMat, inputRectBitmap);
                SaveImage(inputRectBitmap, randomUUID);
            }
            Log.d("gvFR", "faceinfo FD_[l,r,t,b](" + faceinfo.mRect.left + "," + faceinfo.mRect.right + "," +faceinfo.mRect.top + "," + faceinfo.mRect.bottom +
                    "[w,h](" + (faceinfo.mRect.right - faceinfo.mRect.left) + ","  + (faceinfo.mRect.bottom - faceinfo.mRect.top) +")\n");

            //deprecated face_x1_sdk no landmark
//            for (int i = 0; i < 5; i++) {
//                org.opencv.core.Point landmark = new org.opencv.core.Point((int) faceinfo.mLandmark.mX[i], (int) faceinfo.mLandmark.mY[i]);
//                landmarkPoints.add(landmark);
//            }

            //need to crop input image first to decrease dlib FD loading
            int padding = (int)(abs(faceinfo.mRect.right - faceinfo.mRect.left)*0.25);
            int cropleft = faceinfo.mRect.left - padding;
            if( cropleft < 0) { cropleft = 0; }

            int cropright = faceinfo.mRect.right + padding;
            if( cropright > resultBitmap.getWidth()) { cropright = resultBitmap.getWidth(); }

            int croptop = faceinfo.mRect.top - padding;
            if( croptop < 0) { croptop = 0; }

            int cropbottom = faceinfo.mRect.bottom + padding;
            if( cropbottom > resultBitmap.getHeight()) { cropbottom = resultBitmap.getHeight(); }
            Log.d("gvFR", "padding FD_[l,r,t,b](" + cropleft + "," + cropright + "," +croptop + "," + cropbottom +
                    "[w,h](" + (cropright - cropleft) + ","  + (cropbottom - croptop) +")\n");

            Rect roi = new  Rect(cropleft,croptop,cropright-cropleft,cropbottom-croptop);
            Mat croppedMat = ImageMat.submat(roi);

            int resizeLength = 224;
            if(croppedMat.cols() > resizeLength || croppedMat.rows() > resizeLength){
                float resizeRatio = (float) croppedMat.cols()/(float)resizeLength >
                        (float) croppedMat.rows()/(float)resizeLength ? (float) croppedMat.cols()/(float)resizeLength
                        : (float) croppedMat.rows()/(float)resizeLength;
                Imgproc.resize( croppedMat, croppedMat, new Size((int)(croppedMat.cols()/resizeRatio),
                        (int)(croppedMat.rows()/resizeRatio)));
            }

            Bitmap croppedBitmap = Bitmap.createBitmap(croppedMat.cols(),  croppedMat.rows(),Bitmap.Config.RGB_565);
            Utils.matToBitmap(croppedMat, croppedBitmap);
            if(bSaveDebugImage) {
                SaveImage(croppedBitmap, randomUUID);
            }


            //do dlib FD & LM deprecated
//            long dlibStartTime = new Date().getTime();
//            results = faceDet.detect(croppedBitmap);
//            for (final VisionDetRet ret : results) {
//                int rectLeft = ret.getLeft() < 0 ? 0 : ret.getLeft();
//                int rectTop = ret.getTop() < 0 ? 0 : ret.getTop();
//                int rectRight = ret.getRight() > resultBitmap.getWidth() ? resultBitmap.getWidth() : ret.getRight();
//                int rectBottom = ret.getBottom() > resultBitmap.getHeight() ? resultBitmap.getHeight() : ret.getBottom();
//                // get 5 landmark points
//                ArrayList<Point> landmarks = ret.getFaceLandmarks();
//                for (Point point : landmarks) {
//                    int pointX = point.x;
//                    int pointY = point.y;
//                    org.opencv.core.Point landmark = new org.opencv.core.Point(pointX, pointY);
//                    dliblandmarkPoints.add(landmark);
//                }
//                org.opencv.core.Point dlibLeftEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(2).x + dliblandmarkPoints.get(3).x)*0.5), (int)((dliblandmarkPoints.get(2).y + dliblandmarkPoints.get(3).y)*0.5));
//                org.opencv.core.Point dlibRightEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(0).x + dliblandmarkPoints.get(1).x)*0.5), (int)((dliblandmarkPoints.get(0).y + dliblandmarkPoints.get(1).y)*0.5));
//                landmarkPoints.add(dlibLeftEyeCenter);
//                landmarkPoints.add(dlibRightEyeCenter);
//            }
//
//            long dlibEndTime = new Date().getTime();
//            int runTime = (int) (dlibEndTime - dlibStartTime);
//
//
//            if(results.size() > 0){ //no Face detected
//                Log.d("gvFR", "dlib FD & LM faceCount(" + results.size() +
//                        ") landmarkLeft(" + landmarkPoints.get(0).x + "," + landmarkPoints.get(0).y + ") landmarkRight(" + landmarkPoints.get(1).x + "," + landmarkPoints.get(1).y + ")"
//                        +" runTime: " + runTime + " ticks\n");
//                ////do face alignment
//                long warpFRstartTime = new Date().getTime();
//                resultMat = warp(croppedMat, rectPoints, landmarkPoints);
//                long warpFRendTime = new Date().getTime();
//                int warpTime = (int) (warpFRendTime - warpFRstartTime);
//                Log.d("gvFR", "Alignment runTime: " + warpTime + " ticks\n");
//            }else{
//                Log.d("gvFR", "dlib FD & LM no face detected\n");
//                return ERROR_FAILURE;
//            }
        }

        Bitmap output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.RGB_565);
//        Utils.matToBitmap(resultMat, output);

        //do FR
        long FRstartTime = new Date().getTime();
        float[][] embeddings = new float[1][FEATURE_SIZE];

        if(bDoJitter){ //flip jitter once
            Matrix matrix = new Matrix();
            matrix.postScale(-1, 1, (int)(output.getWidth()*0.5), (int)(output.getHeight()*0.5));
            Bitmap mirrorOutput = Bitmap.createBitmap(output, 0, 0, output.getWidth(), output.getHeight(), matrix, true);
            if(bSaveDebugImage) {
                SaveImage(mirrorOutput, randomUUID);
            }
            float[][] embeddingsOrigin = new float[1][FEATURE_SIZE];
            float[][] embeddingsMirror = new float[1][FEATURE_SIZE];
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(output);
            interpreter.run(byteBuffer, embeddingsOrigin);
            ByteBuffer byteBufferMirror = convertBitmapToByteBuffer(mirrorOutput);
            interpreter.run(byteBufferMirror, embeddingsMirror);
            for(int i=0;i<FEATURE_SIZE;i++){
                embeddings[0][i] = (float) ((embeddingsOrigin[0][i] + embeddingsMirror[0][i])*0.5);
            }
        }else{ //no jitter
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(output);
            interpreter.run(byteBuffer, embeddings);
        }
        long FRendTime = new Date().getTime();
        res[0] = (int) (FRendTime - FRstartTime);
        Log.d("gvFR", "FR feature[0,1,128,510,511] ("
                + embeddings[0][0] + ","+ embeddings[0][1] + ","+ embeddings[0][128] + ","+ embeddings[0][510] + ","+ embeddings[0][511]
                + ") runTime(" + res[0] + ") ticks\n");
        System.arraycopy(embeddings[0], 0, feature, 0, embeddings[0].length);
        return 0;
    }

    public int GetFeatureByBitmap(Bitmap resultBitmap, float[] feature, List faceinfos, int[] res) { //deprecated
        Bitmap resizeBitmap = Bitmap.createScaledBitmap(resultBitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizeBitmap);
        float[][] embeddings = new float[1][FEATURE_SIZE];
        long GetFeatureByBitmapStartTime = new Date().getTime();
        interpreter.run(byteBuffer, embeddings);
        long GetFeatureByBitmapEndTime = new Date().getTime();
        res[0] = (int) (GetFeatureByBitmapEndTime - GetFeatureByBitmapStartTime);
        System.arraycopy(embeddings[0], 0, feature, 0, embeddings[0].length);
        return 0;
    }

    public int Compare( float[] origin, float[] chose, float[] score ){
        long CompareStartTime = new Date().getTime();
        double sum = 0;
        boolean bfeatureHasZero = false;
        for(int i=0;i<FEATURE_SIZE;i++){
            double diffValue = origin[i] - chose[i];
//            sum += Math.pow(origin[i] - chose[i],2);
            sum += (diffValue * diffValue);
            if(origin[i] == 0 || chose[i] == 0){
                bfeatureHasZero = true;
                break;
            }
        }
        if(!bfeatureHasZero) {
            score[0] = (float) ((1.00 - (Math.sqrt(sum) * 0.50 - 0.20)) * 100);
        }else {
            score[0] = 0;
        }
        if(score[0]>100) score[0] = 100;
        long CompareEndTime = new Date().getTime();
        int runTime = (int) (CompareEndTime - CompareStartTime);
        Log.d("gvFR", "FR confidence ("
                + score[0] + ") runTime(" +runTime + ") ticks\n");
        return 0;
    }
    /* origin1W 是 origin 中每個 float * 10000 */
    public int CompareWithInteger( int[] origin1W, int[] chose1W)
    {
        int sum = 0;
        boolean bfeatureHasZero = false;

        for(int i=0; i < FEATURE_SIZE; i++){
            int diffValue = origin1W[i] - chose1W[i];
            sum += (diffValue * diffValue);
            /*if(origin1W[i] == 0 || chose1W[i] == 0){
                bfeatureHasZero = true;
                break;
            }*/
        }

        if(!bfeatureHasZero) {
            return sum;
        }

        return 1000000;
    }
    public boolean ReleaseFR() {
        // release model
        interpreter.close();
        interpreter = null;
//        delegate.close();
        return true;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        if(quant) {
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                if(quant){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }
        return byteBuffer;
    }

    private Mat warp(Mat originPhoto, List<org.opencv.core.Point> rect, List<org.opencv.core.Point> landmarks) {
        int resultWidth = INPUT_SIZE;
        int resultHeight = INPUT_SIZE;

        //face alignment by affine transformation via rotation and translation matrix
        Mat outputMat = originPhoto.clone();

        //leftEye rightEye nose leftMouth rightMouth
        org.opencv.core.Point leftEye = new org.opencv.core.Point(landmarks.get(0).x, landmarks.get(0).y);
        org.opencv.core.Point rightEye = new org.opencv.core.Point(landmarks.get(1).x, landmarks.get(1).y);

        //compute the angle between the eye centroids
        int dY = (int) (rightEye.y - leftEye.y);
        int dX = (int) (rightEye.x - leftEye.x);
        double angle = Math.toDegrees(Math.atan2(dY,dX));

        //compute the desired right eye x-coordinate based on the
        // desired x-coordinate of the left eye
        org.opencv.core.Point desiredLeftEye = new org.opencv.core.Point(0.30,0.30);
        double desiredRightEyeX = 1.0 - desiredLeftEye.x;
        int desiredFaceWidth = resultWidth;
        int desiredFaceHeight = resultHeight;

        //determine the scale of the new resulting image by taking
        // the ratio of the distance between eyes in the *current*
        // image to the ratio of distance between eyes in the
        // *desired* image
        double dist = Math.sqrt((Math.pow(dX, 2)) + ((Math.pow(dY, 2))));
        double desiredDist = (desiredRightEyeX - desiredLeftEye.x);
        desiredDist *= desiredFaceWidth;
        double scale = desiredDist / dist;

        //compute center (x, y)-coordinates (i.e., the median point)
        // between the two eyes in the input image
        org.opencv.core.Point eyesCenter =
                new org.opencv.core.Point((leftEye.x + rightEye.x)*0.5,(leftEye.y + rightEye.y)*0.5);

        //check input image rect and landmark
        Mat InputMat = originPhoto.clone();
        int thickness = 2;
        int lineType = 1;
        int shift = 0;
        Imgproc.circle(InputMat, leftEye, InputMat.cols()/100, new Scalar(0,0,255), thickness, lineType, shift);
        Imgproc.circle(InputMat, rightEye, InputMat.cols()/100, new Scalar(0,0,255), thickness, lineType, shift);
        Imgproc.circle(InputMat, eyesCenter, InputMat.cols()/100, new Scalar(0,255,0), thickness, lineType, shift);
        Imgproc.rectangle(InputMat, rect.get(0), rect.get(1), new Scalar(255,0,0), thickness, lineType, shift);
//        //debug save image
        Bitmap output = Bitmap.createBitmap(InputMat.cols(), InputMat.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(InputMat, output);
        if(bSaveDebugImage) {
            SaveImage(output, randomUUID);
        }
        //grab the rotation matrix for rotating and scaling the face
        Mat M = Imgproc.getRotationMatrix2D(eyesCenter,angle, scale);

        //update the translation component of the matrix
        double tX = desiredFaceWidth * 0.5;
        double tY = desiredFaceHeight * desiredLeftEye.y;
        //M[0, 2] += (tX - eyesCenter[0])
        double[] buff = M.get(0, 2);
        buff[0] += (tX - eyesCenter.x);
        M.put(0, 2, buff[0]);
        //M[1, 2] += (tY - eyesCenter[1])
        buff = M.get(1, 2);
        buff[0] += (tY - eyesCenter.y);
        M.put(1, 2, buff[0]);

        //apply the affine transformation
        Imgproc.warpAffine(originPhoto, outputMat, M, new Size(resultWidth, resultHeight), Imgproc.INTER_CUBIC);

        //check outputBitmap
        Bitmap outputBitmap = Bitmap.createBitmap(outputMat.cols(), outputMat.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputMat, outputBitmap);
        if(bSaveDebugImage) {
            SaveImage(outputBitmap, randomUUID);
        }
//deprecated Perspective transform
//        Mat startM = Converters.vector_Point2f_to_Mat(rect);
//        org.opencv.core.Point ocvPOut0 = new org.opencv.core.Point(0, 0);
//        org.opencv.core.Point ocvPOut1 = new org.opencv.core.Point(INPUT_SIZE, INPUT_SIZE);
//        org.opencv.core.Point ocvPOut2 = new org.opencv.core.Point(INPUT_SIZE, 0);
//        org.opencv.core.Point ocvPOut3 = new org.opencv.core.Point(0, INPUT_SIZE);
//        List<org.opencv.core.Point> dest = new ArrayList<>();
//        dest.add(ocvPOut0);
//        dest.add(ocvPOut1);
//        dest.add(ocvPOut2);
//        dest.add(ocvPOut3);
//        Mat endM = Converters.vector_Point2f_to_Mat(dest);
//        MatOfPoint2f matOfPoint2fStart = new MatOfPoint2f(startM);
//        MatOfPoint2f matOfPoint2fEnd = new MatOfPoint2f(endM);
//        Mat perspectiveTransform = Calib3d.findHomography(matOfPoint2fStart, matOfPoint2fEnd);
//        Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, new Size(resultWidth, resultHeight));

        return outputMat;
    }

    private void SaveImage(Bitmap finalBitmap, UUID _uuid) {
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/cropped_face");
        myDir.mkdirs();
//        Long tsLong = System.currentTimeMillis()/1000;
//        String n = tsLong.toString();
        String fname = nSaveDebugImageCnt + "-" + "Face-"+ _uuid + ".jpg";
        nSaveDebugImageCnt++;
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}


