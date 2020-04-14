package com.amitshekhar.tflite;

public class FaceAttribute {
    public static final int GENDER_MALE = 0;
    public static final int GENDER_FEMALE = 1;
    public static final int EMOTION_CALM = 0;
    public static final int EMOTION_HAPPY = 1;
    public int mGender;
    public int mEmotion;
    public int mAge;

    public FaceAttribute() {
        this.mGender = 0;
        this.mEmotion = 0;
        this.mAge = 0;
    }

    public FaceAttribute(int gender, int emotion, int age) {
        this.mGender = gender;
        this.mEmotion = emotion;
        this.mAge = age;
    }

    public FaceAttribute(FaceAttribute a) {
        this.mGender = a.mGender;
        this.mEmotion = a.mEmotion;
        this.mAge = a.mAge;
    }

    public String toString() {
        return "Attr(" + Integer.toString(this.mGender) + "," + Integer.toString(this.mEmotion) + "," + Integer.toString(this.mAge) + ")";
    }
}
