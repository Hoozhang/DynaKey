package com.zhao;

import android.os.Environment;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class DataSaver implements Runnable {

    private String fileName;
    private long timeStamp;
    private float[] sensorData;

    public DataSaver(String fn, long ts, float[] data) {
        fileName = fn;
        timeStamp = ts;
        sensorData = data;
    }

    @Override
    public void run() {
        File file = new File(Environment.getExternalStorageDirectory().getPath(), fileName + ".dat");
        try {
            FileWriter writer = new FileWriter(file, true);
            writer.write(timeStamp + " ");
            for(float f :sensorData)
                writer.write(f + " ");
            writer.write("\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
