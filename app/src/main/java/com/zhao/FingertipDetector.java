package com.zhao;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static java.lang.Math.*;
import static java.lang.Math.pow;

public class FingertipDetector {

    /**
     * 阈值化方法分割出手部的轮廓
     * 找出面积最大的两个轮廓, 视为两只手的轮廓
     * 对每个轮廓, 计算手部重心到轮廓点的距离
     * 根据Prominence和水平间距过滤出波峰, 即为指尖
     * 同时计算重心到指尖的距离, 方便后续的按键动作检测
     */

    Mat inputFrame;

    public FingertipDetector(Mat frame) {
        this.inputFrame = frame;
    }

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
        //Log.i(TAG, "TipDetection 手的图片轮廓数量 = " + contours.size());
        System.out.println("TipDetection 手的图片轮廓数量 = " + contours.size());
        if(contours.size() != 2) {
            //Log.i(TAG, "TipDetection: 轮廓数量太多 or 画面中无手!");
            System.out.println("TipDetection: 轮廓数量太多");
            return finalTipsObj;
        }

        // 面积最大的两个轮廓，即手掌所在的轮廓
        List<MatOfPoint> handContours = new ArrayList<>();
        int maxContourIdx = findMaxContour(contours);
        handContours.add(contours.get(maxContourIdx));
        contours.remove(maxContourIdx);
        handContours.add(contours.get(findMaxContour(contours)));
        // 当手刚进入视野时，可能检测到轮廓很小
        System.out.println(handContours.get(0).toArray().length + "===" + handContours.get(1).toArray().length);
        if (handContours.get(0).toArray().length < 1000 || handContours.get(1).toArray().length < 1000) {
            System.out.println("TipDetection: 轮廓点少!");
            return finalTipsObj;
        }
        // 两只手的重心
        List<Point> handCenters = new ArrayList<>();

        // 每个手掌轮廓分别找指尖
        for (MatOfPoint handContour : handContours) {
            // 手的重心
            Moments moment = Imgproc.moments(handContour, true);
            Point center = new Point(moment.m10/moment.m00, moment.m01/moment.m00);
            handCenters.add(center);
            Imgproc.circle(inputFrame, center, 5, new Scalar(0, 0, 255), -1);
            Point minYPt = findMinYPt(handContour);
            // 提取低于手重心的轮廓点, 以及与重心的距离
            List<Point> handPoints = handContour.toList().stream().filter(point ->
                    point.y < minYPt.y+200).collect(Collectors.toList());
            List<Integer> distances = handPoints.stream().map(point ->
                    (int)distanceBetween(point, center)).collect(Collectors.toList());
            // 前100个点移到后面, 方便检测到五个波峰
            handPoints.addAll(handPoints.subList(0, 100));
            handPoints = handPoints.subList(100, handPoints.size());
            distances.addAll(distances.subList(0, 100));
            distances = distances.subList(100, distances.size());
            /*for (Point pt : handPoints) {
                System.out.println(pt);
            }
            for (int dis : distances) {
                System.out.println(dis);
            }*/
            // 检测距离的波峰和prominence
            List<Integer> peakIdxList = findPeaks(distances);
            List<Integer> prominence = calcProminence(peakIdxList, distances);
            // 根据prominence筛选波峰, 也即指尖
            int tipCount = 0;
            List<Point> oneHandTips = new ArrayList<>();
            while (tipCount < 4) {
                int maxProminenceIdx = findMaxElementIdx(prominence);
                if (prominence.get(maxProminenceIdx) < 3) {

                }
                prominence.set(maxProminenceIdx, -1);
                Point candidateTip = handPoints.get(peakIdxList.get(maxProminenceIdx));
                // 过滤波峰
                //if (candidateTip.y > center.y) {
                //    continue;
                //}
                if (oneHandTips.isEmpty() || isNewTip(candidateTip, oneHandTips)) {
                    oneHandTips.add(handPoints.get(peakIdxList.get(maxProminenceIdx)));
                    //Log.i(TAG, "" + handPoints.get(peakIdxList.get(maxProminenceIdx)));
                    //System.out.println("" + handPoints.get(peakIdxList.get(maxProminenceIdx)));
                    tipCount++;
                }
            }
            finalTips.addAll(oneHandTips);
        }
        // 指尖从左到右排列, 并计算重心到之间的距离
        sortByOneAxis(finalTips, 'x');
        //finalTips.remove(findThumb(finalTips));
        //finalTips.remove(findThumb(finalTips));
        sortByOneAxis(handCenters, 'x');
        for (int i = 0; i < finalTips.size(); i++) {
            int distance = (int)distanceBetween(finalTips.get(i), handCenters.get(i/4));
            finalTipsObj.add(new TipObject(finalTips.get(i), distance, handCenters.get(i/4)));
            Imgproc.circle(inputFrame, finalTips.get(i), 5, new Scalar(0, 0, 255), -1);
        }
        Imgcodecs.imwrite("tipdet/tip.jpg", inputFrame);

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

    public static double distanceBetween(Point p1, Point p2) {
        return sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
    }

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

    private Point findMinYPt(MatOfPoint contour) {
        Point result = contour.toArray()[0];
        for (Point pt : contour.toArray()) {
            if (pt.y < result.y) {
                result = pt;
            }
        }
        return result;
    }
}
