package com.zhao;

import android.app.IntentService;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

import androidx.annotation.Nullable;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.List;

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
        BitmapApplication bitmapApp = (BitmapApplication)getApplicationContext();
        Bitmap inputBitmap = bitmapApp.getOneBitmap();
        Mat inputFrame = new Mat();
        Utils.bitmapToMat(inputBitmap, inputFrame);
        int frameWidth = inputFrame.width();
        int frameHeight = inputFrame.height();

        KeystrokeTask keystrokeTask = new KeystrokeTask(frameCounter, inputFrame);
        if(MainActivity.isFirstFrame) {
            // First frame for Key Exaction
            MainActivity.isFirstFrame = false;
            keystrokeTask.KeyExtraction();
            //frameBeforeMoving = inputBitmap.copy(Bitmap.Config.RGB_565, true);
        } else if(frameCounter % 5 == 0) {
            // Subsequent frames for tracking, keystroke detection and localization
            /*
            // 当Gyro检测运动 or 检测到键盘转换出错时，就继续透视变换
            if(movingGyroValue.getValue() > 0.02 || isWrongTransformation) {
            //if (movingGyroAngle > 0.1 || isWrongTransformation) {
                //MainActivity.UPDATE_KEYMAP = isWrongTransformation? 7 : MainActivity.UPDATE_KEYMAP;
                Log.i(TAG, "onHandleIntent: 开始透视变换 " + System.currentTimeMillis());
                frameAfterMoving = inputBitmap.copy(Bitmap.Config.RGB_565, true);
                keyboardTask.UpdateKeyMap();
                frameBeforeMoving = frameAfterMoving.copy(Bitmap.Config.RGB_565, true);
                //movingGyroAngle = 0;
                //isFirstValue = true;
            } else {
                Log.i(TAG, "onHandleIntent: 停止透视变换 " + System.currentTimeMillis());
            }

            float[] data = {avg(preKeyMap, keyMap), avgIoU(preKeyMap, keyMap)};
            new Thread(new SensorDataSaver("offset", System.currentTimeMillis(), data)).start();
             */

            // 若导出的帧图像水平方向，需旋转90度：转置+水平翻转
            //Mat transposeFrame = new Mat(), flipFrame = new Mat();
            //transpose(oneFrame, transposeFrame);
            //flip(transposeFrame, flipFrame, 1);


            Mat handFrame = keystrokeTask.HandSegmentation();
            List<TipObject> fingertips = keystrokeTask.TipDetection(handFrame);

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
