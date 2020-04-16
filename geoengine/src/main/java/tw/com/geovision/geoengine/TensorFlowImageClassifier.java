package tw.com.geovision.geoengine;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.lang.Math;
import java.util.PriorityQueue;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TensorFlowImageClassifier implements Classifier {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private static final int IMAGE_MEAN = 256;
    private static final float IMAGE_STD = 256.0f;

    private Interpreter interpreter;
    private int inputSize;
    private boolean quant;
    private long startTime;
    private long endTime;
    private long frTime;
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    private final float[][] embeddings01 = new float[1][512];
    private final float[][] embeddings02 = new float[1][512];

    private TensorFlowImageClassifier() {

    }

    public static Classifier create(AssetManager assetManager,
                             String modelPath,
                             int inputSize,
                             boolean quant) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        GpuDelegate delegate = new GpuDelegate();
        Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());
        classifier.inputSize = inputSize;
        return classifier;
    }




    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap, String strID) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] embeddings = new float[1][512];
        startTime = new Date().getTime();
        interpreter.run(byteBuffer, embeddings);
        endTime = new Date().getTime();
        frTime = endTime - startTime;
        int nId = Integer.parseInt(strID);
        if(nId==0) {
            System.arraycopy(embeddings, 0, this.embeddings01, 0, this.embeddings01.length);
        }else {
            System.arraycopy(embeddings, 0, this.embeddings02, 0, this.embeddings02.length);
        }
        return getResultEmbeddings(embeddings, strID);

    }

    @Override
    public double getFRscore() {
        double sum = 0;
        for(int i=0;i<512;i++){
            sum += Math.pow(embeddings01[0][i] - embeddings02[0][i],2);
        }
        double score = (1.00 - (Math.sqrt(sum)*0.50 - 0.20))*100;
        if(score>100) score = 100;
        return score;
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
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

    @SuppressLint("DefaultLocale")
    private List<Recognition> getResultEmbeddings(float[][] embeddingArray, String strId) {
        float confidence = 100;
        Recognition result = new Recognition(strId,"unknown",confidence,quant,embeddingArray,frTime);
        recognitions.add(result);
        return recognitions;
    }
}
