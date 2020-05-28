package tw.com.geovision.geoengine;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.os.Environment;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

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

public class gvFR {
    private static final String PATH_FACE_GV_MODEL = "model";
    private static final String MODEL_NAME = "gvFR.tflite";

    private static GpuDelegate delegate;
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
    private FaceDet faceDet = null;

    public static gvFR initFR(final Context context) throws IOException {
        gvFR model = new gvFR();
        FileUtils.copyAssetFile(context, "model", context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL, false);
        return model;
    }

    public static gvFR CreateFR(gvFR model, final Context context) throws IOException {
        // load model
        delegate = new GpuDelegate();
        Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
//        options.setAllowFp16PrecisionForFp32(true);
        model.interpreter = new Interpreter(new File(context.getFilesDir().getAbsolutePath() + File.separator + PATH_FACE_GV_MODEL + File.separator + MODEL_NAME), options);
//        model.interpreter = new Interpreter(model.loadModelFile( assetManager, modelPath), new Interpreter.Options());
        model.inputSize = INPUT_SIZE;
        return model;
    }

    boolean SetInfo(int iCmd,int value){  //, void *pData
        // set info
        return false;
    }

    public int GetFeature(Mat ImageMat, float[] feature, FaceInfo faceinfo, int[] res, boolean isSaveImage) {
        Bitmap resultBitmap = Bitmap.createBitmap(ImageMat.cols(),  ImageMat.rows(),Bitmap.Config.RGB_565);;
        Utils.matToBitmap(ImageMat, resultBitmap);
        List<VisionDetRet> results = null;
        List<org.opencv.core.Point> rectPoints = new ArrayList<>();
        List<org.opencv.core.Point> dliblandmarkPoints = new ArrayList<>();
        List<org.opencv.core.Point> landmarkPoints = new ArrayList<>();

        //do Face detection
        if(faceinfo==null) {
            //FD and landmark via dlib
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
                    dliblandmarkPoints.add(landmark);
                }
                org.opencv.core.Point dlibLeftEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(2).x + dliblandmarkPoints.get(3).x)*0.5), (int)((dliblandmarkPoints.get(2).y + dliblandmarkPoints.get(3).y)*0.5));
                org.opencv.core.Point dlibRightEyeCenter = new org.opencv.core.Point((int)((dliblandmarkPoints.get(0).x + dliblandmarkPoints.get(1).x)*0.5), (int)((dliblandmarkPoints.get(0).y + dliblandmarkPoints.get(1).y)*0.5));
                landmarkPoints.add(dlibLeftEyeCenter);
                landmarkPoints.add(dlibRightEyeCenter);
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
            for (int i = 0; i < 5; i++) {
                org.opencv.core.Point landmark = new org.opencv.core.Point((int) faceinfo.mLandmark.mX[i], (int) faceinfo.mLandmark.mY[i]);
                landmarkPoints.add(landmark);
            }
        }

        ////do face alignment
        Mat resultMat = warp(ImageMat, rectPoints, landmarkPoints);
        Bitmap output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.RGB_565);
        Utils.matToBitmap(resultMat, output);
//        Bitmap resizeBitmap = Bitmap.createScaledBitmap(resultBitmap, INPUT_SIZE, INPUT_SIZE, false);
        if (isSaveImage) {
            SaveImage(output);
        }
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(output);
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
        return 0;
    }

    public boolean ReleaseFR() {
        // release model
        interpreter.close();
        interpreter = null;
        delegate.close();
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

    public static Mat warp(Mat originPhoto, List<org.opencv.core.Point> rect, List<org.opencv.core.Point> landmarks) {
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
        org.opencv.core.Point desiredLeftEye = new org.opencv.core.Point(0.25,0.25);
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
//        Mat InputMat = originPhoto.clone();
//        int thickness = 2;
//        int lineType = 1;
//        int shift = 0;
//        Imgproc.circle(InputMat, leftEye, InputMat.cols()/100, new Scalar(0,0,255), thickness, lineType, shift);
//        Imgproc.circle(InputMat, rightEye, InputMat.cols()/100, new Scalar(0,0,255), thickness, lineType, shift);
//        Imgproc.circle(InputMat, eyesCenter, InputMat.cols()/100, new Scalar(0,255,0), thickness, lineType, shift);
//        Imgproc.rectangle(InputMat, rect.get(0), rect.get(1), new Scalar(255,0,0), thickness, lineType, shift);
//        Bitmap output = Bitmap.createBitmap(InputMat.cols(), InputMat.rows(), Bitmap.Config.RGB_565);
//        Utils.matToBitmap(InputMat, output);

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
//        Bitmap outputBitmap = Bitmap.createBitmap(outputMat.cols(), outputMat.rows(), Bitmap.Config.RGB_565);
//        Utils.matToBitmap(outputMat, outputBitmap);

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

    private void SaveImage(Bitmap finalBitmap) {
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/cropped_face");
        myDir.mkdirs();
        Long tsLong = System.currentTimeMillis()/1000;
        String n = tsLong.toString();
        String fname = "Face-"+ n +".jpg";
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


