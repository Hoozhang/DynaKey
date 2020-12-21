package com.zhao;

import android.app.IntentService;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

import androidx.annotation.Nullable;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.List;

import static com.zhao.MainActivity.movingGyroValue;
import static com.zhao.MainActivity.frameBeforeMoving;
import static com.zhao.MainActivity.frameAfterMoving;
import static com.zhao.MainActivity.fingertips;

public class KeystrokeService extends IntentService {
    private static final String TAG = "KeyboardService";

    public static Intent newIntent(Context context) {
        return new Intent(context, KeystrokeService.class);
    }

    public KeystrokeService() {
        super("KeyboardService");
    }

    // 处理具体的逻辑(内部已经新开了一个线程)
    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        assert intent != null;
        // 取出 frameCounter 和 frame
        int frameCounter = intent.getIntExtra("frameCounter", -1);
        float startProcessingTime = intent.getFloatExtra("startProcessingTime", -1);
        BitmapApplication bitmapApp = (BitmapApplication)getApplicationContext();
        Bitmap inputBitmap = bitmapApp.getOneBitmap();
        Mat inputFrame = new Mat();
        Utils.bitmapToMat(inputBitmap, inputFrame);

        KeystrokeDetector keystrokeTask = new KeystrokeDetector(frameCounter, inputFrame, startProcessingTime);
        if(MainActivity.isFirstFrame) {
            // First frame for Key Extraction
            MainActivity.isFirstFrame = false;
            new KeyExtractor(inputFrame).KeyExtraction();
            frameBeforeMoving = inputBitmap.copy(Bitmap.Config.RGB_565, true);
        } else if(frameCounter % 5 == 0) {
            // Subsequent frames for tracking, keystroke detection and localization
            // Gyro value for PersTrans
            /*if(movingGyroValue.getValue() > 0.02) {
                frameAfterMoving = inputBitmap.copy(Bitmap.Config.RGB_565, true);
                new KeyTracking().PersTrans();
                frameBeforeMoving = frameAfterMoving.copy(Bitmap.Config.RGB_565, true);
            }*/
            // Fingertip Detection
            FingertipDetector tipDetector = new FingertipDetector(inputFrame);
            Mat handFrame = tipDetector.HandSegmentation();
            fingertips = tipDetector.TipDetection(handFrame);
            // Keystroke detection
            if (!fingertips.isEmpty()) {
                keystrokeTask.KeystrokeDetection(fingertips);
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

}
