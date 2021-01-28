package tw.com.geovision.geoengine.tnn;

import android.graphics.Bitmap;

public class tnnFR {
    static {
        System.loadLibrary("tnnFR");
    }

    public static native void init(String proto, String model, String path, boolean useGPU);
    public static native float[] run(Bitmap bitmap);

}
