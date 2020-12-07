package com.zhao;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.zhao.MainActivity.MESSAGE_TOAST;
import static com.zhao.MainActivity.STROKE_KEY;
import static com.zhao.MainActivity.STROKE_SUC;
import static com.zhao.MainActivity.keyMap;
import static com.zhao.MainActivity.keyboardLeftDown;
import static com.zhao.MainActivity.keyboardLeftUp;
import static com.zhao.MainActivity.keyboardRightDown;
import static com.zhao.MainActivity.keyboardRightUp;
import static com.zhao.MainActivity.prevFingertips;
import static com.zhao.MainActivity.prevStrokeFrame;
import static com.zhao.MainActivity.prevStrokeKey;
import static java.lang.Math.PI;
import static java.lang.Math.max;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class KeystrokeTask {
    private static final String TAG = "KeystrokeTask";

    // 帧的计数
    private final int frameCounter;
    // 当前帧
    private final Mat inputFrame;
    // 帧的宽度和高度
    private final int frameWidth, frameHeight;

    private float startProcessingTime;

    public KeystrokeTask(int count, Mat oneFrame, float startProcessingTime) {
        frameCounter = count;
        frameWidth = oneFrame.width();
        frameHeight = oneFrame.height();
        inputFrame = oneFrame.clone();
        this.startProcessingTime = startProcessingTime;
    }

    /**
     * Keystroke Detection and Localization,
     *
     */

    public void KeystrokeDetection(List<TipObject> fingertips) {
        if (prevFingertips.isEmpty()) {
            // 初始情况下记录上一帧的指尖和距离
            prevFingertips = fingertips;
        } else {
            // 判断两帧之间的指尖距离
            int isStill = hasStillFingertips(fingertips);
            // 判断指尖到重心的距离
            int hasTyping = hasTypingFingertips(fingertips, isStill);
            // 指尖几乎静止 & 有指尖弯曲
            if (isStill != 0 && hasTyping != -1) {
                int frameInterval = frameCounter - prevStrokeFrame;
                String key = keyLocation(fingertips, hasTyping);
                // 判断是否是重复按键
                if (prevStrokeKey.equals("NullKey") || !prevStrokeKey.equals(key) || frameInterval >= 30) {
                    System.out.println("---------------------------------  Keystroke: " + key + " " + frameCounter);
                    MESSAGE_TOAST = STROKE_SUC;
                    STROKE_KEY = key;
                    prevStrokeFrame = frameCounter;
                    prevStrokeKey = key;
                    float stopProcessingTime = System.nanoTime()/1000000;
                    float[] time = {stopProcessingTime - startProcessingTime};
                    new Thread(new DataSaver("KeyStrokeDelay", frameCounter, time)).start();
                }
            }
            // 记录上一帧的指尖
            for (int i = 0; i < fingertips.size(); i++) {
                prevFingertips.get(i).setTip(fingertips.get(i).getTip());
                prevFingertips.get(i).setCenter(fingertips.get(i).getCenter());
                if (fingertips.get(i).getDistance() > prevFingertips.get(i).getDistance())
                    prevFingertips.get(i).setDistance(fingertips.get(i).getDistance());
            }
        }
    }

    private int hasStillFingertips(List<TipObject> fingertips) {
        boolean stillLeft = true;
        for (int i = 0; i < fingertips.size()/2; i++) {
            int distance = (int)distanceBetween(fingertips.get(i).getTip(), prevFingertips.get(i).getTip());
            if (distance > 15) {
                stillLeft = false;
                break;
            }
        }
        boolean stillRight = true;
        for (int i = fingertips.size()/2; i < fingertips.size(); i++) {
            int distance = (int) distanceBetween(fingertips.get(i).getTip(), prevFingertips.get(i).getTip());
            if (distance > 15) {
                stillRight = false;
                break;
            }
        }
        if (stillLeft && stillRight) {
            return 3;
        } else if (stillRight) {
            return 2;
        } else if (stillLeft) {
            return 1;
        } else {
            return 0;
        }
    }

    private int hasTypingFingertips(List<TipObject> fingertips, int isStill) {
        if (isStill == 0) {
            return -1;
        }
        // 判断是左手高还是右手高
        int leftSum = 0, rightSum = 0;
        for (int i = 0; i < fingertips.size(); i++) {
            if (i < 4) {
                leftSum += (int)fingertips.get(i).getTip().y;
            } else {
                rightSum += (int)fingertips.get(i).getTip().y;
            }
        }
        int beginIdx, endIdx;
        if (leftSum < rightSum) {
            if (isStill == 2) {
                return -1;
            }
            beginIdx = 0;
            endIdx = fingertips.size()/2;
        } else {
            if (isStill == 1) {
                return -1;
            }
            beginIdx = fingertips.size()/2;
            endIdx = fingertips.size();
        }

        int tipIdx = -1;
        int maxOffset = Integer.MIN_VALUE;
        for (int i = beginIdx; i < endIdx; i++) {
            int offset1 = prevFingertips.get(i).getDistance();
            int offset2 = fingertips.get(i).getDistance();
            int offset = Math.abs(offset1 - offset2);
            if (offset > maxOffset && KeyExtractor.inKeyboardArea("keyDet", fingertips.get(i).getTip())) {
                tipIdx = i;
                maxOffset = offset;
            }
        }
        if (tipIdx == -1) {
            // 没找到手指
            return -1;
        } else {
            return (maxOffset > prevFingertips.get(tipIdx).getDistance() / 10) ? tipIdx : -1;
        }

    }

    private String keyLocation(List<TipObject> fingertips, int typingIdx) {
        Point tip = fingertips.get(typingIdx).getTip();
        double minDis = Double.POSITIVE_INFINITY;
        String minDisKey = "NullKey";
        // 找与按键的指尖距离最近的key
        for(String key : keyMap.keySet()) {
            Point coord4Key = keyMap.get(key);
            assert(coord4Key != null); // assert后的布尔表达式为true时继续执行
            double dis = distanceBetween(tip, coord4Key);
            if(dis < minDis) {
                minDis = dis;
                minDisKey = key;
            }
        }
        return minDisKey;
    }

    public static double distanceBetween(Point p1, Point p2) {
        return sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
    }

    private void saveFrame(String fileName, Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(mat, bitmap); // OpenCV的Mat2Bitmap
        new Thread(new FrameSaver(fileName, bitmap)).start(); // 保存图片
    }

}
