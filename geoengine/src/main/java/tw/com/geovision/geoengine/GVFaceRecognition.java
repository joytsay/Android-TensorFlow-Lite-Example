package tw.com.geovision.geoengine;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Point;
import android.os.Environment;
import android.util.Log;
import android.widget.GridView;

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
    public static final String version = "v0.2.0";
    private static final String PATH_FACE_GV_MODEL = "model";
    private static final String FR_MODEL_NAME = "gvFR.tflite";
    private static final String MTCNN_MODEL_NAME = "mtcnn_freezed_model.pb";

    private static GVFaceRecognition gvFaceRecognitionInstance = null;

//    private static GpuDelegate delegate;
    private Interpreter interpreter;
    private final ArrayList<Classifier.Recognition> recognitions = new ArrayList<>();
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

    public Vector<Box> faceDetect(Mat rgbaMat, int minFaceSize) {
        Vector<Box> boxes = null;
        //MTCNN FD & LM & crop
        Bitmap croppedBitmap = Bitmap.createBitmap(rgbaMat.cols(),  rgbaMat.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(rgbaMat, croppedBitmap);
        Bitmap bm= tw.com.geovision.geoengine.mtcnn.Utils.copyBitmap(croppedBitmap);
        try {
            return boxes = mtcnn.detectFaces(bm, minFaceSize);
//                //draw MTCNN
//                for (int i=0;i<boxes.size();i++){
//                    tw.com.geovision.geoengine.mtcnn.Utils.drawRect(bm,boxes.get(i).transform2Rect());
//                    tw.com.geovision.geoengine.mtcnn.Utils.drawPoints(bm,boxes.get(i).landmark);
//                }
        }catch (Exception e){
            Log.e("MTCNN","[*]detect false:"+e);
            return boxes; //null
        }
    }

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
        tfliteOptions.setNumThreads(2);
        gvFaceRecognitionInstance.interpreter = new Interpreter(new File(context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + FR_MODEL_NAME), tfliteOptions);
        gvFaceRecognitionInstance.inputSize = INPUT_SIZE;

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
        List<org.opencv.core.Point> rectPoints = new ArrayList<>();
        Mat resultMat = null;
        bSaveDebugImage = isSaveImage;
        randomUUID = UUID.randomUUID();
        //do Face detection
        if(faceinfo==null) {
            //MTCNN FD & LM & crop
            Vector<Box> boxes = faceDetect(ImageMat,100);
            if (boxes.size() == 0){
                return ERROR_NOT_EXIST;
            }
            resultMat = getCrop(ImageMat, boxes.get(0)); //get first face only
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

            //need to crop and give padding to decrease FD loading
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

            //MTCNN FD & LM & crop
            Vector<Box> boxes = faceDetect(croppedMat,100);
            if (boxes.size() == 0){
                return ERROR_NOT_EXIST;
            }
            resultMat = getCrop(croppedMat, boxes.get(0)); //get first face only
        }

        //do FR
        Bitmap output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.RGB_565);
        Utils.matToBitmap(resultMat, output);
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
        return SUCCESS;
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
        return outputMat;
    }

    private Mat getCrop(Mat src, Box faceObj){
        // get 2 landmark points
        org.opencv.core.Point LeftEyeCenter = new org.opencv.core.Point(faceObj.landmark[0].x,faceObj.landmark[0].y);
        org.opencv.core.Point RightEyeCenter = new org.opencv.core.Point(faceObj.landmark[1].x,faceObj.landmark[1].y);
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

    private void SaveImage(Bitmap finalBitmap, UUID _uuid) {
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/cropped_face");
        myDir.mkdirs();
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


