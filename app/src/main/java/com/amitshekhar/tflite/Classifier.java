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
        private final float[][] embeddings = new float[1][512];

        /**
         * Execute time for face recognition.
         */
        private final long frTime;

        public Recognition(
                final String id, final String title, final Float confidence, final boolean quant, float[][] embeddings, final long time) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.quant = quant;
            System.arraycopy(embeddings, 0, this.embeddings, 0,this.embeddings.length);
            this.frTime = time;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (embeddings[0] != null) {
                resultString += String.format("feature[0,1,128,510,511](%.3f,%.3f,%.3f,%.3f,%.3f) ",
                        embeddings[0][0], embeddings[0][1], embeddings[0][128], embeddings[0][510], embeddings[0][511]);
            }
            if (frTime > 0) {
                resultString += "FR time [" + frTime + "] ticks ";
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


    List<Recognition> recognizeImage(Bitmap bitmap);

    void close();
}
