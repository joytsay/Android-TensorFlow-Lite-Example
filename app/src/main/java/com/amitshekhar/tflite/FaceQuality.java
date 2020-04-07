package com.amitshekhar.tflite;

public class FaceQuality {
    public float mScore;
    public float mLeftRight;
    public float mUpDown;
    public float mHorizontal;
    public float mClarity;
    public float mBright;

    public FaceQuality() {
        this.mScore = 0.0F;
        this.mLeftRight = 0.0F;
        this.mUpDown = 0.0F;
        this.mHorizontal = 0.0F;
        this.mClarity = 0.0F;
        this.mBright = 0.0F;
    }

    public FaceQuality(float score, float leftRight, float upDown, float horizontal, float clarity) {
        this.mScore = score;
        this.mLeftRight = leftRight;
        this.mUpDown = upDown;
        this.mHorizontal = horizontal;
        this.mClarity = clarity;
    }

    public FaceQuality(FaceQuality q) {
        this.mScore = q.mScore;
        this.mLeftRight = q.mLeftRight;
        this.mUpDown = q.mUpDown;
        this.mHorizontal = q.mHorizontal;
        this.mClarity = q.mClarity;
        this.mBright = q.mBright;
    }

    public String toString() {
        return "Quality(" + Float.toString(this.mScore) + "," + Float.toString(this.mLeftRight) + "," + Float.toString(this.mUpDown) + "," + Float.toString(this.mHorizontal) + "," + Float.toString(this.mClarity) + ")";
    }
}
