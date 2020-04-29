package tw.com.geovision.geoengine;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Point;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class gvFR {
    private Interpreter interpreter;
    private int inputSize;
    public static final int SUCCESS = 0; //执行接口返回成功
    public static final int ERROR_INVALID_PARAM = -1; //非法参数
    public static final int ERROR_TOO_MANY_REQUESTS = -2; //太多请求
    public static final int ERROR_NOT_EXIST = -3; //不存在
    public static final int ERROR_FAILURE = -4; // 执行接口返回失败

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;
    private static final int INPUT_SIZE = 112;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;


    private boolean quant;
    private long startTime;
    private long endTime;
    private long frTime;
    final ArrayList<Classifier.Recognition> recognitions = new ArrayList<>();
    FaceDet faceDet = null;

    public static gvFR CreateFR(AssetManager assetManager, String modelPath) throws IOException {
        // load model
        gvFR model = new gvFR();
        model.interpreter = new Interpreter(model.loadModelFile( assetManager, modelPath), new Interpreter.Options());
        model.inputSize = INPUT_SIZE;
        return model;
    }

    boolean SetInfo(int iCmd,int value){  //, void *pData
        // set info
        return false;
    }

    public int GetFeature(Mat ImageMat, float[] feature, FaceInfo faceinfo, int[] res) {
        Bitmap resultBitmap = Bitmap.createBitmap(ImageMat.cols(),  ImageMat.rows(),Bitmap.Config.ARGB_8888);;
        Utils.matToBitmap(ImageMat, resultBitmap);
        List<VisionDetRet> results = null;
        List<org.opencv.core.Point> rectPoints = new ArrayList<>();
        List<org.opencv.core.Point> landmarkPoints = new ArrayList<>();

        //do Face detection
        if(faceinfo==null) {
            if(faceDet==null) {
                faceDet = new FaceDet(Constants.getFaceShapeModelPath());
            }
            results = faceDet.detect(resultBitmap);
            for (final VisionDetRet ret : results) {
                int rectLeft = ret.getLeft() < 0 ? 0 : ret.getLeft();
                int rectTop = ret.getTop() < 0 ? 0 : ret.getTop();
                int rectRight = ret.getRight() > resultBitmap.getWidth() ? resultBitmap.getWidth() : ret.getRight();
                int rectBottom = ret.getBottom() > resultBitmap.getHeight() ? resultBitmap.getHeight() : ret.getBottom();
                // get 5 landmark points
                ArrayList<Point> landmarks = ret.getFaceLandmarks();
                for (Point point : landmarks) {
                    int pointX = point.x;
                    int pointY = point.y;
                    org.opencv.core.Point landmark = new org.opencv.core.Point(pointX, pointY);
                    landmarkPoints.add(landmark);
                }
                org.opencv.core.Point rect0 = new org.opencv.core.Point(rectLeft, rectTop);
                rectPoints.add(rect0);
                org.opencv.core.Point rect1 = new org.opencv.core.Point(rectRight, rectBottom);
                rectPoints.add(rect1);
                org.opencv.core.Point rect2 = new org.opencv.core.Point(rectRight, rectTop);
                rectPoints.add(rect2);
                org.opencv.core.Point rect3 = new org.opencv.core.Point(rectLeft, rectBottom);
                rectPoints.add(rect3);
            }
            if(results.size() == 0){ //no Face detected
                return ERROR_FAILURE;
            }
        }else{ //given Rect by user
            org.opencv.core.Point rect0 = new org.opencv.core.Point(faceinfo.mRect.left, faceinfo.mRect.top);
            rectPoints.add(rect0);
            org.opencv.core.Point rect1 = new org.opencv.core.Point(faceinfo.mRect.right, faceinfo.mRect.bottom);
            rectPoints.add(rect1);
            org.opencv.core.Point rect2 = new org.opencv.core.Point(faceinfo.mRect.right, faceinfo.mRect.top);
            rectPoints.add(rect2);
            org.opencv.core.Point rect3 = new org.opencv.core.Point(faceinfo.mRect.left, faceinfo.mRect.bottom);
            rectPoints.add(rect3);
        }

        ////do face alignment
        resultBitmap = warp(resultBitmap, rectPoints, landmarkPoints);
        Bitmap resizeBitmap = Bitmap.createScaledBitmap(resultBitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizeBitmap);
        float[][] embeddings = new float[1][512];
        startTime = new Date().getTime();
        interpreter.run(byteBuffer, embeddings);
        endTime = new Date().getTime();
        res[0] = (int) (endTime - startTime);
        System.arraycopy(embeddings[0], 0, feature, 0, embeddings[0].length);
        return 0;
    }

    public int GetFeatureByBitmap(Bitmap resultBitmap, float[] feature, List faceinfos, int[] res) { //deprecated
        Bitmap resizeBitmap = Bitmap.createScaledBitmap(resultBitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizeBitmap);
        float[][] embeddings = new float[1][512];
        startTime = new Date().getTime();
        interpreter.run(byteBuffer, embeddings);
        endTime = new Date().getTime();
        res[0] = (int) (endTime - startTime);
        System.arraycopy(embeddings[0], 0, feature, 0, embeddings[0].length);
        return 0;
    }

    public int Compare( float[] origin, float[] chose, float[] score ){
        double sum = 0;
        boolean bfeatureHasZero = false;
        for(int i=0;i<512;i++){
            sum += Math.pow(origin[i] - chose[i],2);
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
        return 0;
    }

    public boolean ReleaseFR() {
        // release model
        interpreter.close();
        interpreter = null;
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

    public static Bitmap warp(Bitmap originPhoto, List<org.opencv.core.Point> rect, List<org.opencv.core.Point> landmarks) {
        int resultWidth = INPUT_SIZE;
        int resultHeight = INPUT_SIZE;

        //det rect quad homography transform
        Mat inputMat = new Mat(originPhoto.getHeight(), originPhoto.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(originPhoto, inputMat);
        Mat outputMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC1);
        Mat rotateMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC1);
        Mat startM = Converters.vector_Point2f_to_Mat(rect);
        org.opencv.core.Point ocvPOut0 = new org.opencv.core.Point(0, 0);
        org.opencv.core.Point ocvPOut1 = new org.opencv.core.Point(INPUT_SIZE, INPUT_SIZE);
        org.opencv.core.Point ocvPOut2 = new org.opencv.core.Point(INPUT_SIZE, 0);
        org.opencv.core.Point ocvPOut3 = new org.opencv.core.Point(0, INPUT_SIZE);
        List<org.opencv.core.Point> dest = new ArrayList<>();
        dest.add(ocvPOut0);
        dest.add(ocvPOut1);
        dest.add(ocvPOut2);
        dest.add(ocvPOut3);
        Mat endM = Converters.vector_Point2f_to_Mat(dest);
        MatOfPoint2f matOfPoint2fStart = new MatOfPoint2f(startM);
        MatOfPoint2f matOfPoint2fEnd = new MatOfPoint2f(endM);
        Mat perspectiveTransform = Calib3d.findHomography(matOfPoint2fStart, matOfPoint2fEnd);


        Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, new Size(resultWidth, resultHeight));

        Bitmap output = Bitmap.createBitmap(resultWidth, resultHeight, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(outputMat, output);

        return output;
    }
}
