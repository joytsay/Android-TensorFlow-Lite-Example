package tw.com.geovision.geoengine;
import android.graphics.Rect;

public class FaceInfo {
    public Rect mRect;
    public FaceAttribute mAttr;
    public FaceQuality mQuality;
    public Landmark mLandmark;

    public FaceInfo() {
        this.mRect = new Rect();
        this.mAttr = new FaceAttribute();
        this.mQuality = new FaceQuality();
        this.mLandmark = new Landmark();
    }

    public FaceInfo(FaceInfo info) {
        this.mRect = new Rect(info.mRect);
        this.mAttr = new FaceAttribute(info.mAttr);
        this.mQuality = new FaceQuality(info.mQuality);
        this.mLandmark = new Landmark(info.mLandmark);
    }

    public FaceInfo(Rect rect, FaceAttribute attr, FaceQuality quality, Landmark landmark) {
        this.mRect = new Rect(rect);
        this.mAttr = new FaceAttribute(attr);
        this.mQuality = new FaceQuality(quality);
        this.mLandmark = new Landmark(landmark);
    }

    public FaceInfo(Rect rect) {
        this.mRect = new Rect(rect);
        this.mAttr = new FaceAttribute();
        this.mQuality = new FaceQuality();
        this.mLandmark = new Landmark();
    }

    public String toString() {
        return this.mRect.toString() + "," + this.mAttr.toString() + "," + this.mQuality.toString() + "," + this.mLandmark.toString();
    }
}