package com.zhao;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.zhao.KeystrokeDetector.distanceBetween;
import static com.zhao.MainActivity.keyMap;
import static com.zhao.MainActivity.keyboardLeftDown;
import static com.zhao.MainActivity.keyboardLeftUp;
import static com.zhao.MainActivity.keyboardRightDown;
import static com.zhao.MainActivity.keyboardRightUp;
import static java.lang.Math.PI;
import static java.lang.Math.pow;

public class KeyExtractor {

    /**
     * 预处理帧, 返回Canny边缘检测后的二进制图
     * 找轮廓, 面积最大的视为键盘边缘轮廓
     * 根据夹角找键盘的四个角点
     * 根据轮廓重心和面积找key的重心
     * 映射键盘布局和key的坐标
     */

    private final Mat inputFrame;

    public KeyExtractor(Mat frame) {
        inputFrame = frame;
    }

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
        //Log.i(TAG, " 键盘轮廓原始数量 = " + contourSize);
        System.out.println(" 键盘轮廓原始数量 = " + contourSize);
        // 得到最大轮廓
        MatOfPoint maxContour = keyboardContours.get(findMaxContour(keyboardContours));
        Point[] maxContourArray = maxContour.toArray();
        double keyboardArea = Imgproc.contourArea(maxContour);
        // 根据键盘边缘轮廓找到四个角点，用于过滤false轮廓
        List<Point> cornerList = findCornerPoints(maxContourArray);
        keyboardLeftUp = cornerList.get(0);
        keyboardRightUp = cornerList.get(1);
        keyboardLeftDown = cornerList.get(2);
        keyboardRightDown = cornerList.get(3);
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
            double threshold4NewKey = Math.min(keyboardRightDown.x- keyboardLeftUp.x,
                    keyboardRightDown.y- keyboardLeftUp.y)/8;
            if (keyCenterList.isEmpty() || isNewKey(keyCenterList, center, threshold4NewKey)) {
                keyCenterList.add(center);
                momentCount++;
                Imgproc.circle(frame, center, 5, new Scalar(255, 0, 0), -1);
            }
        }
        //saveFrame("Corner_Keys", frame);
        Imgcodecs.imwrite("v2f/Corner_Keys.jpg", frame);
        //Log.i(TAG, " 键盘轮廓最后数量 = " + momentCount + " (应等于40)");
        System.out.println(" 键盘轮廓最后数量 = " + momentCount + " (应等于40)");
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
    private List<Point> findCornerPoints(Point[] pointArray) {
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
            //Log.i(TAG, "findCornerSelf: 检测到不止4个角点");
            System.out.println("findCornerSelf: 检测到不止4个角点");
        sortByOneAxis(cornerList, 'y');
        for(int i = 0; i <= cornerList.size()-2; i = i+2)
            sortByOneAxis(cornerList.subList(i,i+2), 'x');
        return cornerList;
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

    public static boolean inKeyboardArea(String signal, Point point) {
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
            //Log.i(TAG, signal + " inKeyboardArea: true");
            return true;
        } else {
            //Log.i(TAG, signal + " inKeyboardArea: false");
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
                "Sp", "z", "x", "c", "_", "b", "n", "m", ",", "."};
        for(int i = 0; i < 40; i++) {
            keyMap.put(charArr[i], keyList.get(i));
            //Log.i(TAG, "keyMap: " + charArr[i] + ", " + keyList.get(i));
            System.out.println("keyMap: " + charArr[i] + ", " + keyList.get(i));
        }
    }
}
