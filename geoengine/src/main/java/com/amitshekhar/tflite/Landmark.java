package com.amitshekhar.tflite;
import java.util.Arrays;

public class Landmark {
    public float[] lmX = new float[106];
    public float[] lmY = new float[106];
    public float[] mX = new float[5];
    public float[] mY = new float[5];

    public Landmark() {
    }

    public Landmark(float[] x, float[] y, float[] lm_x, float[] lm_y) {
        int i;
        for(i = 0; i < Math.min(x.length, this.mX.length); ++i) {
            this.mX[i] = x[i];
        }

        for(i = 0; i < Math.min(y.length, this.mY.length); ++i) {
            this.mY[i] = y[i];
        }

        for(i = 0; i < Math.min(lm_x.length, this.lmX.length); ++i) {
            this.lmX[i] = lm_x[i];
        }

        for(i = 0; i < Math.min(lm_y.length, this.lmY.length); ++i) {
            this.lmY[i] = lm_y[i];
        }

    }

    public Landmark(Landmark l) {
        int i;
        for(i = 0; i < 5; ++i) {
            this.mX[i] = l.mX[i];
        }

        for(i = 0; i < 5; ++i) {
            this.mY[i] = l.mY[i];
        }

        for(i = 0; i < 106; ++i) {
            this.lmX[i] = l.lmX[i];
        }

        for(i = 0; i < 106; ++i) {
            this.lmY[i] = l.lmY[i];
        }

    }

    public String toString() {
        return "Landmark(" + Arrays.toString(this.mX) + " && " + Arrays.toString(this.mY) + ")\nnew Landmark:" + Arrays.toString(this.lmX) + " && " + Arrays.toString(this.lmY);
    }
}