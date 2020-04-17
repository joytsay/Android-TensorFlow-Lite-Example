package com.amitshekhar.tflite;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v4.content.ContextCompat;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;
import com.wonderkiln.camerakit.CameraView;


import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


import tw.com.geovision.geoengine.Classifier;
import tw.com.geovision.geoengine.FaceInfo;
import tw.com.geovision.geoengine.FileUtils;
import tw.com.geovision.geoengine.Image;
import tw.com.geovision.geoengine.TensorFlowImageClassifier;
import tw.com.geovision.geoengine.gvFR;

public class MainActivity extends AppCompatActivity {

    //interface for face_x1_sdk
    private gvFR face;
    private static final int REQUEST_CODE_PERMISSION = 2;

    private static final String MODEL_PATH = "gvFR.tflite";
    private static final boolean QUANT = true;
    private static final int INPUT_SIZE = 224;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult, textViewScore, textViewSDK01, textViewSDK02, textViewSDKScore;
    private Button btnDetectObject, btnToggleCamera, butttonFR1, buttonFR2, buttonDoFR, buttonDoSDK;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private static int RESULT_LOAD_IMAGE01 = 0;
    private static int RESULT_LOAD_IMAGE02 = 1;
    private static final int MY_PERMISSION_READ_FILES = 100;
    Bitmap imgBitmapFR01;
    Bitmap imgBitmapFR02;
    Bitmap imgBitmapAlign01;
    Bitmap imgBitmapAlign02;
    private FaceDet mFaceDet;

    // Storage Permissions
    private static String[] PERMISSIONS_REQ = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
    };
    private static boolean verifyPermissions(Activity activity) {
        // Check if we have write permission
        int write_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int read_persmission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int camera_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.CAMERA);

        if (write_permission != PackageManager.PERMISSION_GRANTED ||
                read_persmission != PackageManager.PERMISSION_GRANTED ||
                camera_permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_REQ,
                    REQUEST_CODE_PERMISSION
            );
            return false;
        } else {
            return true;
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        List<Classifier.Recognition> results;
        if(requestCode == RESULT_LOAD_IMAGE01 && resultCode == RESULT_OK && null != data){
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };
            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn,null,null,null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();
            ImageView imageView = (ImageView) findViewById(R.id.imageViewFace01);
            imgBitmapFR01 = BitmapFactory.decodeFile(picturePath);
            imageView.setImageBitmap(imgBitmapFR01);
            //Resize image and prepare tensor pixel normalized to [-1 ~ +1]
            Bitmap resizeBitmap = Bitmap.createScaledBitmap(imgBitmapFR01, INPUT_SIZE, INPUT_SIZE, false);
            //Inference Face Recognition
            results = classifier.recognizeImage(resizeBitmap, "0");
            textViewResult.setText(results.toString());
        }

        if(requestCode == RESULT_LOAD_IMAGE02 && resultCode == RESULT_OK && null != data){
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };
            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn,null,null,null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();
            ImageView imageView = (ImageView) findViewById(R.id.imageViewFace02);
            imgBitmapFR02 = BitmapFactory.decodeFile(picturePath);
            imageView.setImageBitmap(imgBitmapFR02);
            //Resize image and prepare tensor pixel normalized to [-1 ~ +1]
            Bitmap resizeBitmap = Bitmap.createScaledBitmap(imgBitmapFR02, INPUT_SIZE, INPUT_SIZE, false);
            //Inference Face Recognition
            results = classifier.recognizeImage(resizeBitmap, "1");
            textViewResult.setText(results.toString());
        }
    }//onActivityResult

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textViewResult = findViewById(R.id.textViewFRResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());
        textViewScore = findViewById(R.id.textViewFRScore);
        textViewSDK01 = findViewById(R.id.textViewSDKResult01);
        textViewSDK02 = findViewById(R.id.textViewSDKResult02);
        textViewSDKScore = findViewById(R.id.textViewSDKScore);


        //Loads and initializes OpenCV library (system.loadLibrary("opencv_java"))
        OpenCVLoader.initDebug();

        //check files permission
        if( ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},MY_PERMISSION_READ_FILES);
        }
        // For API 23+ you need to request the read/write permissions even if they are already in your manifest.
        int currentapiVersion = android.os.Build.VERSION.SDK_INT;

        if (currentapiVersion >= Build.VERSION_CODES.M) {
            verifyPermissions(this);
        }

        //create TensorFlow Lite model
        initTensorFlowAndLoadModel();

        //load img for face 01
        butttonFR1 = findViewById(R.id.btnLoadImg01);
        butttonFR1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i,RESULT_LOAD_IMAGE01);
            }
        });

        //load img for face 02
        butttonFR1 = findViewById(R.id.btnLoadImg02);
        butttonFR1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i,RESULT_LOAD_IMAGE02);
            }
        });

        //calculate Face similarity
        buttonDoFR = findViewById(R.id.btnRunFR);
        buttonDoFR.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                double FRscore = classifier.getFRscore();
                String strFRscore= "FR Score: " + FRscore;
                textViewScore.setText(strFRscore);
            }
        });


        //run interface for face_x1_sdk
        buttonDoSDK = findViewById(R.id.btnRunSDK);
        buttonDoSDK.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                //copy landmark file to sdcard
                final String targetPath = Constants.getFaceShapeModelPath();
                if (!new File(targetPath).exists()) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Copy landmark model to " + targetPath, Toast.LENGTH_SHORT).show();
                        }
                    });
                    FileUtils.copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_5_face_landmarks, targetPath);
                }

                //get imgBitmapFR to Mat
                Mat mat01 = new Mat();
                Utils.bitmapToMat(imgBitmapFR01, mat01);

                Mat mat02 = new Mat();
                Utils.bitmapToMat(imgBitmapFR02, mat02);

                //convert OpenCV Mat to SDK Image
                Image image01 = new Image();
                long getNativeObjAddr01 = mat01.getNativeObjAddr();
                image01.matAddrframe = getNativeObjAddr01;
                image01.height = mat01.height();
                image01.width = mat01.width();

                Image image02 = new Image();
                long getNativeObjAddr02 = mat02.getNativeObjAddr();
                image02.matAddrframe = getNativeObjAddr02;
                image02.height = mat02.height();
                image02.width = mat02.width();

                //new SDK face object
                float[] feature01 = new float[512];
                Arrays.fill(feature01, 0.0f);
                float[] feature02 = new float[512];
                Arrays.fill(feature02, 0.0f);

                List<FaceInfo> tmpPos = null;
                int[] res = new int[1];
                Arrays.fill(res, 0);

                //create FR model
                try {
                    face = gvFR.CreateFR(getAssets(),MODEL_PATH);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //extract feature via FR from image
                int retGetFeature01 = face.GetFeature( mat01, feature01, tmpPos, res );
                if( retGetFeature01 == gvFR.SUCCESS ){
                    String strSDK= "[SDK results: feature01[0,1,128,510,511] \n("
                            + feature01[0] + ","+ feature01[1] + ","+ feature01[128] + ","+ feature01[510] + ","+ feature01[511]
                            + ")\n FR time: [" + res[0] + "] ticks]\n";
                    textViewSDK01.setText(strSDK);
                    Log.d("MainActivity","retGetFeature01" + strSDK);
                }


                int retGetFeature02 = face.GetFeature( mat02, feature02, tmpPos, res );
                if( retGetFeature02 == gvFR.SUCCESS ){
                    String strSDK = "[SDK results: feature02[0,1,128,510,511] \n("
                            + feature02[0] + ","+ feature02[1] + ","+ feature02[128] + ","+ feature02[510] + ","+ feature02[511]
                            + ")\n FR time: [" + res[0] + "] ticks]\n";
                    textViewSDK02.setText(strSDK);
                    Log.d("MainActivity","retGetFeature02" + strSDK);
                }


                long  startTime = new Date().getTime();

                //compare features
                float[] compareScore = new float[1];
                Arrays.fill(compareScore, 0);
                int retCompare = face.Compare(feature01,feature02,compareScore);
                if( retCompare == gvFR.SUCCESS ){
                    String strSDKscore= "SDK FR score: [" + compareScore[0] + "]\n" + "Compare time: " + (new Date().getTime() - startTime)+ "\n";
                    textViewSDKScore.setText(strSDKscore);
                    Log.d("MainActivity","compareScore: " + strSDKscore);
                }
            }
        });
    }//onCreate

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {

                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            INPUT_SIZE,
                            QUANT);
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });//thread
    }//initTensorFlowAndLoadModel
}
