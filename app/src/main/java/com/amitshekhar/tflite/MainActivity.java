package com.amitshekhar.tflite;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_PATH = "gvFR.tflite";
    private static final boolean QUANT = true;
    private static final String LABEL_PATH = "labels.txt";
    private static final int INPUT_SIZE = 224;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera, butttonFR1, buttonFR2, buttonDoFR;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private static int RESULT_LOAD_IMAGE = 1;
    private static final int MY_PERMISSION_READ_FILES = 100;
//    private Bitmap imgBitmapFR01, imgBitmapFR02;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data){
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };
            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn,null,null,null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();
            ImageView imageView = (ImageView) findViewById(R.id.imageViewFace01);
            Bitmap imgBitmapFR01 = BitmapFactory.decodeFile(picturePath);
            imageView.setImageBitmap(imgBitmapFR01);
            //Resize image and prepare tensor pixel normalized to [-1 ~ +1]
            Bitmap resizeBitmap = Bitmap.createScaledBitmap(imgBitmapFR01, INPUT_SIZE, INPUT_SIZE, false);
            //Inference Face Recognition
            final List<Classifier.Recognition> results = classifier.recognizeImage(resizeBitmap);
            textViewResult.setText(results.toString());
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textViewResult = findViewById(R.id.textViewFRResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        //check files permission
        if( ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},MY_PERMISSION_READ_FILES);
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
                startActivityForResult(i,RESULT_LOAD_IMAGE);
            }
        });





        //Deprecated code
//        cameraView = findViewById(R.id.cameraView);
//        imageViewResult = findViewById(R.id.imageViewResult);
//        textViewResult = findViewById(R.id.textViewResult);
//        textViewResult.setMovementMethod(new ScrollingMovementMethod());
//
//        btnToggleCamera = findViewById(R.id.btnToggleCamera);
//        btnDetectObject = findViewById(R.id.btnDetectObject);
//
//        cameraView.addCameraKitListener(new CameraKitEventListener() {
//            @Override
//            public void onEvent(CameraKitEvent cameraKitEvent) {
//
//            }
//
//            @Override
//            public void onError(CameraKitError cameraKitError) {
//
//            }
//
//            @Override
//            public void onImage(CameraKitImage cameraKitImage) {
//
//                Bitmap bitmap = cameraKitImage.getBitmap();
//
//                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
//
//                imageViewResult.setImageBitmap(bitmap);
//
//                final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);
//
//                textViewResult.setText(results.toString());
//
//            }
//
//            @Override
//            public void onVideo(CameraKitVideo cameraKitVideo) {
//
//            }
//        });
//
//        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                cameraView.toggleFacing();
//            }
//        });
//
//        btnDetectObject.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                cameraView.captureImage();
//            }
//        });
//
//        initTensorFlowAndLoadModel();
//    }
//
//    @Override
//    protected void onResume() {
//        super.onResume();
//        cameraView.start();
//    }
//
//    @Override
//    protected void onPause() {
//        cameraView.stop();
//        super.onPause();
//    }
//
//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        executor.execute(new Runnable() {
//            @Override
//            public void run() {
//                classifier.close();
//            }
//        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {

                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            LABEL_PATH,
                            INPUT_SIZE,
                            QUANT);
//                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    //Deprecated code
//    private void makeButtonVisible() {
//        runOnUiThread(new Runnable() {
//            @Override
//            public void run() {
//                btnDetectObject.setVisibility(View.VISIBLE);
//            }
//        });
//    }
}
