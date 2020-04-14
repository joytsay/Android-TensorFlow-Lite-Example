package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * Whether or not the model features quantized or float weights.
         */
        private final boolean quant;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /**
         * Extracted 512D embeddings for face recognition.
         */
        private final float[][] embeddings01 = new float[1][512];
        private final float[][] embeddings02 = new float[1][512];

        /**
         * Execute time for face recognition.
         */
        private final long frTime[] = new long[2];
        /**
         * Output String for Textview.
         */
        private String resultString = "";

        public Recognition(
                final String id, final String title, final Float confidence, final boolean quant, float[][] embeddings, final long time) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.quant = quant;
            int nId = Integer.parseInt(id);
            if(nId==0) {
                System.arraycopy(embeddings, 0, this.embeddings01, 0, this.embeddings01.length);
            }else {
                System.arraycopy(embeddings, 0, this.embeddings02, 0, this.embeddings02.length);
            }
            this.frTime[nId] = time;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public double getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            if (frTime[0] > 0){
                resultString += String.format("\nFace One feature: [0,1,128,510,511](%.3f,%.3f,%.3f,%.3f,%.3f)",
                        embeddings01[0][0], embeddings01[0][1], embeddings01[0][128], embeddings01[0][510], embeddings01[0][511]);
                resultString += "\nFR time: [" + frTime[0] + "] ticks\n";
                frTime[0] = 0;
            }
            if (frTime[1] > 0) {
                resultString += String.format("\nFace Two feature: [0,1,128,510,511](%.3f,%.3f,%.3f,%.3f,%.3f)",
                        embeddings02[0][0], embeddings02[0][1], embeddings02[0][128], embeddings02[0][510], embeddings02[0][511]);
                resultString += "\nFR time: [" + frTime[1] + "] ticks\n";
                frTime[1] = 0;
            }
//            if (id != null) {
//                resultString += "[" + id + "] ";
//            }
//
//            if (title != null) {
//                resultString += title + " ";
//            }
//
//            if (confidence != null) {
//                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
//            }

            return resultString.trim();
        }
    }


    List<Recognition> recognizeImage(Bitmap bitmap, String strId);

    double getFRscore();

    void close();
}
