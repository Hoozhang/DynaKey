package com.zhao;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import static com.zhao.MainActivity.movingGyroValue;

public class SensorActivity implements SensorEventListener {
    private static final String TAG = "SensorActivity";

    private final SensorManager sensorManager;
    private final Sensor sensorGyro;

    public static double movingGyroAngle = 0.0;
    public static boolean isFirstValue = true;
    private double lastGyroMagnitude = 0.0;
    private long lastTimeStamp = 0;

    public SensorActivity(SensorManager sensorManager) {
        this.sensorManager = sensorManager;
        sensorGyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
    }

    public void registerListener() {
        // 监听传感器
        sensorManager.registerListener(this, sensorGyro, SensorManager.SENSOR_DELAY_FASTEST);
    }

    public void unregisterListener() {
        // 取消监听传感器
        sensorManager.unregisterListener(this);
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        int sensorType = sensorEvent.sensor.getType();
        if (sensorType == Sensor.TYPE_GYROSCOPE) {
            float[] gyroData = sensorEvent.values.clone();
            double gyroMagnitude = Math.sqrt(gyroData[0]*gyroData[0]+gyroData[1]*gyroData[1]+gyroData[2]*gyroData[2]);
            movingGyroValue.pushValue((float)gyroMagnitude);
            //long formatTimeStamp = (new Date()).getTime() + (sensorEvent.timestamp-System.nanoTime())/1000000L;
            //integralGyroAngle(gyroMagnitude, formatTimeStamp/1000L);
            //new Thread(new SensorDataSaver("sensorGyro2", System.currentTimeMillis(), gyroData)).start();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void integralGyroAngle(double gyroMagnitude, long timeStamp) {
        if (isFirstValue) {
            isFirstValue = false;
            lastGyroMagnitude = gyroMagnitude;
            lastTimeStamp = timeStamp;
        } else {
            double duration = timeStamp-lastTimeStamp;
            Log.i(TAG, "integralGyroAngle: timastamp " + timeStamp);
            movingGyroAngle += (gyroMagnitude+lastGyroMagnitude)/2 * duration;
            Log.i(TAG, "integralGyroAngle: " + movingGyroAngle);
            lastGyroMagnitude = gyroMagnitude;
            lastTimeStamp = timeStamp;
        }

    }

}
