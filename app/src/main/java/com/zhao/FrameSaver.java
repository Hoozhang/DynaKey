package com.zhao;

import android.graphics.Bitmap;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class FrameSaver implements Runnable {

    private Bitmap bm;
    private String bmName;

    public FrameSaver(String bmName, Bitmap bm) {
        this.bmName = bmName;
        this.bm = bm;
    }

    @Override
    public void run() {
        String storePath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "DynaKey";
        File sdCardDir = new File(storePath);
        if (!sdCardDir.exists())
            sdCardDir.mkdir();
        File file = new File(sdCardDir, bmName + ".jpg");
        try {
            FileOutputStream fos = new FileOutputStream(file);
            bm.compress(Bitmap.CompressFormat.JPEG, 80, fos);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
