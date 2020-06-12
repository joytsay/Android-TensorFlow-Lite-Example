/*
 * Copyright 2016-present Tzutalin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tw.com.geovision.geoengine;

import android.content.Context;
import android.support.annotation.NonNull;
import android.support.annotation.RawRes;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by Tzutalin on 2016/3/30.
 */
public class FileUtils {
    @NonNull
    public static final void copyFileFromRawToOthers(@NonNull final Context context, @RawRes int id, @NonNull final String targetPath) {
        InputStream in = context.getResources().openRawResource(id);
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(targetPath);
            byte[] buff = new byte[1024];
            int read = 0;
            while ((read = in.read(buff)) > 0) {
                out.write(buff, 0, read);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (in != null) {
                    in.close();
                }
                if (out != null) {
                    out.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 将asset中文件copy到系统中
     * @param context
     * @param assetPath
     * @param saveFilePath
     * @param need_overwrite 文件存在时是否覆盖写入
     */
    public static boolean copyAssetFile(Context context, String assetPath, String saveFilePath, boolean need_overwrite) {
        try {
            String fileNames[] = context.getAssets().list(assetPath);//获取assets目录下的所有文件及目录名
            if (fileNames.length > 0) {//如果是目录
                File file = new File(saveFilePath);
                file.mkdirs();//如果文件夹不存在，则递归
                for (String fileName : fileNames) {
                    copyAssetFile(context, assetPath + "/" + fileName, saveFilePath + "/" + fileName,need_overwrite);
                }
            } else {//如果是文件
                File configFile = new File(saveFilePath);
                if (need_overwrite){
                    saveFile(context, assetPath, configFile);
                }else {//如果copy为false，但是系统中又没有configFile文件，则强制copy
                    if (!configFile.exists()){
                        saveFile(context, assetPath, configFile);
                    }
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
            //如果捕捉到错误则通知UI线程
            return false;
        }
        return true;
    }


    /**
     * 将assets中文件copy到系统中
     * @param context
     * @param assetPath
     * @param saveFile
     * @throws
     */
    private static void saveFile(Context context, String assetPath, File saveFile) throws Exception {
        InputStream is = context.getAssets().open(assetPath);
        FileOutputStream fos = new FileOutputStream(saveFile);
        byte[] buffer = new byte[1024];
        int byteCount = 0;
        while ((byteCount = is.read(buffer)) != -1) {//循环从输入流读取 buffer字节
            fos.write(buffer, 0, byteCount);//将读取的输入流写入到输出流
        }
        fos.flush();//刷新缓冲区
        is.close();
        fos.close();
    }
}
