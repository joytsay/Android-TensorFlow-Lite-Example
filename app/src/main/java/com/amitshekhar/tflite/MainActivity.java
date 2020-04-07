package com.amitshekhar.tflite;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v4.content.ContextCompat;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.wonderkiln.camerakit.CameraView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    //interface for face_x1_sdk
    private gvFR face;

    private static final String MODEL_PATH = "gvFR.tflite";
    private static final boolean QUANT = true;
    private static final String LABEL_PATH = "labels.txt";
    private static final int INPUT_SIZE = 224;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult, textViewScore;
    private Button btnDetectObject, btnToggleCamera, butttonFR1, buttonFR2, buttonDoFR;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private static int RESULT_LOAD_IMAGE01 = 0;
    private static int RESULT_LOAD_IMAGE02 = 1;
    private static final int MY_PERMISSION_READ_FILES = 100;

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
            Bitmap imgBitmapFR01 = BitmapFactory.decodeFile(picturePath);
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
            Bitmap imgBitmapFR02 = BitmapFactory.decodeFile(picturePath);
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

        //check files permission
        if( ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},MY_PERMISSION_READ_FILES);
        }

        //create TensorFlow Lite model
        initTensorFlowAndLoadModel();

        //interface for face_x1_sdk
        gvFR face = new gvFR();
        float[] feature = new float[512];
        List<FaceInfo> tmpPos = null;
        int[] res = new int[0];
        Image image = null;
        int ret = face.GetFeature( image, feature, tmpPos, res );
        if( ret == gvFR.SUCCESS ){
            //to do 成功获取到人脸特征值
        }

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
    }//onCreate

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
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });//thread
    }//initTensorFlowAndLoadModel

}
