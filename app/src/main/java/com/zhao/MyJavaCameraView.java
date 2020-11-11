package com.zhao;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;

import org.opencv.android.JavaCameraView;

public class MyJavaCameraView extends JavaCameraView {
    private static final String TAG = "MyJavaCameraView";

    public MyJavaCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void setAbsolution(int width , int height) {
        Log.i(TAG, "setAbsolution: " + width + " " + height);
        disconnectCamera();
        mMaxWidth = width;
        mMaxHeight = height;
        connectCamera(getWidth(), getHeight());
    }

}
