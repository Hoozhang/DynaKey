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

import static com.zhao.MainActivity.keyMap;
import static com.zhao.MainActivity.keyboardLeftDown;
import static com.zhao.MainActivity.keyboardLeftUp;
import static com.zhao.MainActivity.keyboardRightDown;
import static com.zhao.MainActivity.keyboardRightUp;
import static com.zhao.MainActivity.prevFingertips;
import static com.zhao.MainActivity.prevStrokeFrame;
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

    public KeystrokeTask(int count, Mat oneFrame) {
        frameCounter = count;
        frameWidth = oneFrame.width();
        frameHeight = oneFrame.height();
        inputFrame = oneFrame.clone();
    }

    /**
     * Key Extraction,
     * 预处理帧, 返回Canny边缘检测后的二进制图
     * 找轮廓, 面积最大的视为键盘边缘轮廓
     * 根据夹角找键盘的四个角点
     * 根据轮廓重心和面积找key的重心
     * 映射键盘布局和key的坐标
     */
    public void KeyExtraction() {
        Mat frame = inputFrame.clone();
        // 预处理图像: 阈值化&边缘检测，返回cannyImage
        Mat cannyFrame = preProcessingFrame(frame);

        // 找键盘轮廓
        List<MatOfPoint> keyboardContours = new ArrayList<>();
        // 输出轮廓的拓扑结构信息
        Mat hierarchy = new Mat();
        // 参数一的输入必须为单通道二进制图像
        Imgproc.findContours(cannyFrame, keyboardContours, hierarchy, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_NONE);
        //Imgproc.drawContours(frame, keyboardContours, -1,  new Scalar(255, 0, 0));
        int contourSize = keyboardContours.size();
        Log.i(TAG, " 键盘轮廓原始数量 = " + contourSize);
        // 得到最大轮廓
        MatOfPoint maxContour = keyboardContours.get(findMaxContour(keyboardContours));
        Point[] maxContourArray = maxContour.toArray();
        double keyboardArea = Imgproc.contourArea(maxContour);
        // 根据键盘边缘轮廓找到四个角点，用于过滤false轮廓
        findCornerPoints(maxContourArray);
        Imgproc.circle(frame, keyboardLeftUp, 5, new Scalar(255, 0, 0), -1);
        Imgproc.circle(frame, keyboardRightUp, 5, new Scalar(255, 0, 0), -1);
        Imgproc.circle(frame, keyboardLeftDown, 5, new Scalar(255, 0, 0), -1);
        Imgproc.circle(frame, keyboardRightDown, 5, new Scalar(255, 0, 0), -1);

        // 过滤false轮廓，确定各个按键的重心
        ArrayList<Point> keyCenterList = new ArrayList<>();
        int momentCount = 0;
        for (int i = 0; i < contourSize; i++) {
            MatOfPoint contour = keyboardContours.get(i);
            // 条件一: 过滤重心不在键盘范围内的轮廓
            Moments moment = Imgproc.moments(contour, true);
            Point center = new Point(moment.m10/moment.m00, moment.m01/moment.m00);
            if (!inKeyboardArea("KeyExtraction", center))
                continue;
            // 条件二: 过滤面积过小和过大的轮廓
            double area = Imgproc.contourArea(contour);
            if (area < keyboardArea/80 || area > keyboardArea/30)
                continue;
            // 条件三: 根据两个重心很靠近的特征过滤两个近似重复的轮廓
            double threshold4NewKey = Math.min(keyboardRightDown.x-keyboardLeftUp.x,
                    keyboardRightDown.y- keyboardLeftUp.y)/8;
            if (keyCenterList.isEmpty() || isNewKey(keyCenterList, center, threshold4NewKey)) {
                keyCenterList.add(center);
                momentCount++;
                Imgproc.circle(frame, center, 5, new Scalar(255, 0, 0), -1);
            }
        }
        saveFrame("Corner_Keys", frame);
        Log.i(TAG, " 键盘轮廓最后数量 = " + momentCount + " (应等于40)");
        mapCoord2Key(keyCenterList); // 坐标和按键映射
    }

    // 预处理帧, 返回边缘检测后的二进制单通道图
    private Mat preProcessingFrame(Mat srcFrame) {
        Mat grayFrame = new Mat(), thresholdFrame = new Mat(),
                blurFrame = new Mat(), cannyFrame = new Mat();
        // 原图转换为灰度图
        Imgproc.cvtColor(srcFrame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // 灰度图做阈值化, THRESH_BINARY: 大于阈值时变为maxVal(白色)，否则变为0(黑色)
        Imgproc.threshold(grayFrame, thresholdFrame, 100, 255, Imgproc.THRESH_BINARY);
        // 二进制图做高斯滤波
        Imgproc.GaussianBlur(thresholdFrame, blurFrame, new Size(3, 3), 0);
        // Canny 边缘检测
        Imgproc.Canny(blurFrame, cannyFrame, 40, 100);
        // 形态学膨胀, dilate in place, 返回canny边缘检测后的单通道二进制图像
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(cannyFrame, cannyFrame, element);
        return cannyFrame;
    }

    // 从键盘轮廓中找键盘边缘的轮廓
    private int findMaxContour(List<MatOfPoint> contours) {
        int maxContourIdx = 0;
        double maxContourArea = 0;
        int contoursSize = contours.size();
        for(int i = 0; i < contoursSize; i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if(area > maxContourArea) {
                maxContourIdx = i;
                maxContourArea = area;
            }
        }
        return maxContourIdx;
    }

    // 找键盘边缘的四个角点
    private void findCornerPoints(Point[] pointArray) {
        List<Point> contour = new ArrayList<>(Arrays.asList(pointArray));
        // contour后面再添加前面的100个点，方便循环检测一遍
        contour.addAll(Arrays.asList(pointArray).subList(0,100));
        List<Point> cornerList = new ArrayList<>();
        // 寻找四个角点
        int preTheta = -1;
        int pre2Theta = -1;
        for(int i = 50; i < contour.size()-50; i++) {
            Point current = contour.get(i);
            int theta = calcTheta(contour, i, 50); //点乘和模得夹角
            if(theta == 0)
                theta += 180;
            // 寻找角度值的极小值点
            if ( (cornerList.size() == 0 || distanceBetween(current, cornerList.get(cornerList.size()-1)) > 100)
                    &&  theta >= preTheta && pre2Theta >= preTheta && theta < 120 ) {
                cornerList.add(current);
            }
            pre2Theta = preTheta;
            preTheta = theta;
        }
        // 检测到的角点映射到键盘四个角点
        if (cornerList.size() != 4)
            Log.i(TAG, "findCornerSelf: 检测到不止4个角点");
        sortByOneAxis(cornerList, 'y');
        for(int i = 0; i <= cornerList.size()-2; i = i+2)
            sortByOneAxis(cornerList.subList(i,i+2), 'x');
        keyboardLeftUp = cornerList.get(0);
        keyboardRightUp = cornerList.get(1);
        keyboardLeftDown = cornerList.get(2);
        keyboardRightDown = cornerList.get(3);
    }

    private int calcTheta(List<Point> hullList, int i, int gap) {
        int size = hullList.size();
        Point current = hullList.get(i);
        Point pre, next;
        pre = hullList.get((i-gap+size)%size);
        next = hullList.get((i+gap)%size);

        double dot = (pre.x-current.x )*(next.x-current.x) + (pre.y-current.y )*(next.y-current.y);
        double mo1 = Math.sqrt( pow(pre.x-current.x, 2) + pow(pre.y-current.y, 2));
        double mo2 = Math.sqrt( pow(next.x-current.x, 2) + pow(next.y-current.y, 2));
        return (int)(Math.acos(dot / (mo1 * mo2))*180/ PI); //点乘和模得夹角
    }

    private void sortByOneAxis(List<Point> List, char c) {
        for(int i = 0; i < List.size(); i++) {
            int minIdx = i;
            for(int j = i; j < List.size(); j++) {
                switch(c) {
                    case 'x':
                        if(List.get(j).x < List.get(minIdx).x)
                            minIdx = j;
                        break;
                    case 'y':
                        if(List.get(j).y < List.get(minIdx).y)
                            minIdx = j;
                        break;
                    default:
                        break;
                }
            }
            Point temp = List.get(minIdx);
            List.set(minIdx, List.get(i));
            List.set(i, temp);
        }
    }

    private boolean inKeyboardArea(String signal, Point point) {
        Point AB = new Point(keyboardRightUp.x-keyboardLeftUp.x, keyboardRightUp.y-keyboardLeftUp.y);
        Point AP = new Point(point.x-keyboardLeftUp.x, point.y-keyboardLeftUp.y);
        Point BC = new Point(keyboardRightDown.x-keyboardRightUp.x, keyboardRightDown.y-keyboardRightUp.y);
        Point BP = new Point(point.x-keyboardRightUp.x, point.y-keyboardRightUp.y);
        Point CD = new Point(keyboardLeftDown.x-keyboardRightDown.x, keyboardLeftDown.y-keyboardRightDown.y);
        Point CP = new Point(point.x-keyboardRightDown.x, point.y-keyboardRightDown.y);
        Point DA = new Point(keyboardLeftUp.x-keyboardLeftDown.x, keyboardLeftUp.y-keyboardLeftDown.y);
        Point DP = new Point(point.x-keyboardLeftDown.x, point.y-keyboardLeftDown.y);

        double AB_X_AP = AB.x * AP.y - AP.x * AB.y;
        double BC_X_BP = BC.x * BP.y - BP.x * BC.y;
        double CD_X_CP = CD.x * CP.y - CP.x * CD.y;
        double DA_X_DP = DA.x * DP.y - DP.x * DA.y;

        if ( AB_X_AP >= 0 && BC_X_BP >= 0 && CD_X_CP >= 0 && DA_X_DP >= 0) {
            Log.i(TAG, signal + " inKeyboardArea: true");
            return true;
        } else {
            Log.i(TAG, signal + " inKeyboardArea: false");
            return false;
        }
    }

    private boolean isNewKey(ArrayList<Point> list, Point p, double threshold) {
        boolean notNearPoint= true;
        for(Point point : list) {
            double dis = distanceBetween(p, point);
            if(dis < threshold) {
                notNearPoint = false;
                break;
            }
        }
        return notNearPoint;
    }

    private void mapCoord2Key(ArrayList<Point> keyList) {
        // 先按照纵坐标值排序，然后每行按横坐标值排序
        sortByOneAxis(keyList, 'y');
        for(int i = 0; i <= keyList.size()-10; i = i+10) {
            sortByOneAxis(keyList.subList(i,i+10), 'x');
        }
        // Java参数地址传递后，momentList内部已排序好
        String[] charArr = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
                "a", "s", "d", "f", "g", "h", "j", "k", "l", "Ent",
                "Sp", "z", "x", "c", "v", "b", "n", "m", ",", "."};
        for(int i = 0; i < 40; i++) {
            keyMap.put(charArr[i], keyList.get(i));
            Log.i(TAG, "keyMap: " + charArr[i] + ", " + keyList.get(i));
        }
    }

    private void saveFrame(String fileName, Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(mat, bitmap); // OpenCV的Mat2Bitmap
        new Thread(new FrameSaver(fileName, bitmap)).start(); // 保存图片
    }

    /**
     * Fingertip Detection,
     * 阈值化+Ostu方法分割出手部的轮廓
     * 找出面积最大的两个轮廓, 视为两只手的轮廓
     * 对每个轮廓, 计算手部重心到轮廓点的距离
     * 根据Prominence和水平间距过滤出波峰, 即为指尖
     * 同时计算重心到指尖的距离, 方便后续的按键动作检测
     */
    public Mat HandSegmentation() {
        Mat YCrCbFrame = new Mat();
        Imgproc.cvtColor(inputFrame, YCrCbFrame, Imgproc.COLOR_BGR2YCrCb);
        // Cr + Otsu 方法，提取Cr分量 & Ostu分量阈值化
        Mat CrFrame = new Mat();
        // 分离通道，提取Cr分量
        Core.extractChannel(YCrCbFrame, CrFrame, 1);
        // 阈值化操作
        Imgproc.threshold(CrFrame, CrFrame, 0, 255, Imgproc.THRESH_BINARY|Imgproc.THRESH_OTSU);
        // 形态学腐蚀和膨胀
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.erode(CrFrame, CrFrame, element);
        Imgproc.dilate(CrFrame, CrFrame, element);
        // OpenCV4PC手是白色，OpenCV4Android手是黑色；如果不反过来，后面指尖提取会出问题
        Imgproc.threshold(CrFrame, CrFrame, 100, 255, Imgproc.THRESH_BINARY_INV);
        return CrFrame;
    }

    // 根据手的轮廓，检测按键的手指指尖
    public List<TipObject> TipDetection(Mat CrFrame) {
        List<TipObject> finalTipsObj = new ArrayList<>();
        List<Point> finalTips = new ArrayList<>();
        // 手的图像
        Mat handFrame = new Mat();
        inputFrame.copyTo(handFrame, CrFrame);
        // 存储轮廓结果
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat(); //存储轮廓的拓扑结构
        Imgproc.findContours(CrFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        Log.i(TAG, "TipDetection 手的图片轮廓数量 = " + contours.size());
        if(contours.size() > 5) { //轮廓效果不理想or画面中无手，直接返回(0,0)
            Log.i(TAG, "TipDetection: 轮廓数量太多 or 画面中无手!");
            return finalTipsObj;
        }
        // 面积最大的两个轮廓，即手掌所在的轮廓
        List<MatOfPoint> handContours = new ArrayList<>();
        int maxContourIdx = findMaxContour(contours);
        handContours.add(contours.get(maxContourIdx));
        contours.remove(maxContourIdx);
        handContours.add(contours.get(findMaxContour(contours)));
        // 两只手的重心
        Point[] centers = new Point[2];
        int centerIdx = 0;

        // 每个手掌轮廓分别找指尖
        for (MatOfPoint handContour : handContours) {
            // 手的重心
            Moments moment = Imgproc.moments(handContour, true);
            Point center = new Point(moment.m10/moment.m00, moment.m01/moment.m00);
            centers[centerIdx++] = center;
            Imgproc.circle(inputFrame, center, 5, new Scalar(0, 0, 255), -1);
            // 提取高于手重心的轮廓点, 以及与重心的距离
            List<Point> handPoints = handContour.toList().stream().filter(point ->
                    point.y < (center.y+inputFrame.height())/2).collect(Collectors.toList());
            List<Integer> distances = handPoints.stream().map(point ->
                    (int)distanceBetween(point, center)).collect(Collectors.toList());
            // 前100个点移到后面, 方便检测到五个波峰
            handPoints.addAll(handPoints.subList(0, 100));
            handPoints = handPoints.subList(100, handPoints.size());
            distances.addAll(distances.subList(0, 100));
            distances = distances.subList(100, distances.size());
            // 检测距离的波峰和prominence
            List<Integer> peakIdxList = findPeaks(distances);
            List<Integer> prominence = calcProminence(peakIdxList, distances);
            // 根据prominence筛选波峰, 也即指尖
            int tipCount = 0;
            while (tipCount < 5) {
                int maxProminenceIdx = findMaxElementIdx(prominence);
                prominence.set(maxProminenceIdx, 0);
                Point candidateTip = handPoints.get(peakIdxList.get(maxProminenceIdx));
                if (finalTips.isEmpty() || isNewTip(candidateTip, finalTips)) {
                    finalTips.add(handPoints.get(peakIdxList.get(maxProminenceIdx)));
                    Log.i(TAG, "" + handPoints.get(peakIdxList.get(maxProminenceIdx)));
                    Imgproc.circle(inputFrame, handPoints.get(peakIdxList.get(maxProminenceIdx)), 5, new Scalar(0, 0, 255), -1);
                    tipCount++;
                }
            }
        }
        // 指尖从左到右排列, 并计算重心到之间的距离
        sortByOneAxis(finalTips, 'x');
        for (int i = 0; i < finalTips.size(); i++) {
            int distance = (int)distanceBetween(finalTips.get(i), centers[i/5]);
            finalTipsObj.add(new TipObject(finalTips.get(i), distance, centers[i/5]));
        }
        Imgcodecs.imwrite("tipdet/handFrame.jpg", inputFrame);

        return finalTipsObj;
    }

    private List<Integer> findPeaks(List<Integer> list) {
        List<Integer> result = new ArrayList<>();
        for (int i = 15; i < list.size()-15; i++) {
            int prev = list.get(i-15);
            int curr = list.get(i);
            int next = list.get(i+15);
            if (curr >= prev && curr >= next) {
                result.add(i);
            }
        }
        return result;
    }

    private List<Integer> calcProminence(List<Integer> peaksIdx, List<Integer> list) {
        List<Integer> result = new ArrayList<>();
        for (int idx = 0; idx < peaksIdx.size(); idx++) {
            // 峰值的下标
            int peakIdx = peaksIdx.get(idx);
            // 峰值
            int peakValue = list.get(peakIdx);
            // left
            int i = idx-1;
            for (; i >= 0; i--) {
                if (list.get(peaksIdx.get(i)) > peakValue) {
                    break;
                }
            }
            int leftValley =  (i == -1) ? findMinValue(list, 0, peakIdx) : findMinValue(list, peaksIdx.get(i), peakIdx);
            // right
            int j = idx+1;
            for (; j < peaksIdx.size(); j++) {
                if (list.get(peaksIdx.get(j)) > peakValue) {
                    break;
                }
            }
            int rightValley = (j == peaksIdx.size()) ? findMinValue(list, peakIdx, list.size()) : findMinValue(list, peakIdx, peaksIdx.get(j));
            result.add(peakValue - max(leftValley, rightValley));
        }
        return result;
    }

    private int findMinValue(List<Integer> list, int begin, int end) {
        int valley = list.get(begin);
        while (begin < end) {
            if (list.get(begin) < valley) {
                valley = list.get(begin);
            }
            begin++;
        }
        return valley;
    }

    private int findMaxElementIdx(List<Integer> list) {
        int maxElemIdx = 0;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > list.get(maxElemIdx)) {
                maxElemIdx = i;
            }
        }
        return maxElemIdx;
    }

    private boolean isNewTip(Point point, List<Point> list) {
        for (Point pt : list) {
            if (distanceBetween(point, pt) < 30) {
                return false;
            }
        }
        return true;
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
            // 判断两帧之间的指尖距离, 静止
            boolean isStill = isStillFingertips(fingertips);
            // 判断指尖到重心的距离, 按键动作
            int hasTyping = hasTypingFingertips(fingertips);
            // 与上一次keystroke相隔的帧
            int frameInterval = frameCounter - prevStrokeFrame;
            if (isStill && hasTyping != -1 && frameInterval > 10) {
                String key = keyLocation(fingertips, hasTyping);
                prevStrokeFrame = frameCounter;
            }
            // 记录上一帧的指尖
            for (int i = 0; i < fingertips.size(); i++) {
                prevFingertips.get(i).setTip(fingertips.get(i).getTip());
                prevFingertips.get(i).setCenter(fingertips.get(i).getCenter());
            }
        }
    }

    private boolean isStillFingertips(List<TipObject> fingertips) {
        for (int i = 0; i < fingertips.size(); i++) {
            int distance = (int)distanceBetween(fingertips.get(i).getTip(), prevFingertips.get(i).getTip());
            if ( distance > 15) {
                return false;
            }
        }
        return true;
    }

    private int hasTypingFingertips(List<TipObject> fingertips) {
        int tipIdx = -1;
        int maxOffset = Integer.MIN_VALUE;
        for (int i = 0; i < fingertips.size(); i++) {
            int offset = prevFingertips.get(i).getDistance() - fingertips.get(i).getDistance();
            if (offset > maxOffset) {
                tipIdx = i;
                maxOffset = offset;
            }
        }
        return (maxOffset > prevFingertips.get(tipIdx).getDistance() / 5) ? tipIdx : -1;
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

}
