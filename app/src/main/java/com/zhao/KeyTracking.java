package com.zhao;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static com.zhao.KeystrokeDetector.distanceBetween;
import static com.zhao.MainActivity.frameAfterMoving;
import static com.zhao.MainActivity.frameBeforeMoving;
import static com.zhao.MainActivity.keyMap;
import static java.lang.Math.*;

public class KeyTracking {

    private Mat srcFrame;

    private Mat dstFrame;

    private int frameCounter;

    public KeyTracking() {
        srcFrame = new Mat();
        Utils.bitmapToMat(frameBeforeMoving, srcFrame);
        dstFrame = new Mat();
        Utils.bitmapToMat(frameAfterMoving, dstFrame);
    }

    public KeyTracking(Mat src, Mat dst, int count) {
        this.srcFrame = src;
        this.dstFrame = dst;
        frameCounter = count;
    }

    public void PersTrans() {
        KeystrokeDetector.saveFrame("before", srcFrame);
        KeystrokeDetector.saveFrame("after", dstFrame);
        MatOfPoint2f srcQuad = new MatOfPoint2f();
        List<Point> src4Points = keypointSelection(srcFrame);
        srcQuad.fromList(src4Points);
        System.out.println(src4Points.size());

        MatOfPoint2f dstQuad = new MatOfPoint2f();
        List<Point> dst4Points = keypointSelection(dstFrame);
        dstQuad.fromList(dst4Points);
        System.out.println(dst4Points.size());

        Mat warpMat = Imgproc.getPerspectiveTransform(srcQuad, dstQuad);
        // keyMap透视变换
        for (String key : keyMap.keySet()) {
            Point[] valArray = new Point[1];
            valArray[0] = keyMap.get(key);
            MatOfPoint2f unWarpedValue = new MatOfPoint2f();
            unWarpedValue.fromArray(valArray);
            MatOfPoint2f warpedValue = new MatOfPoint2f();
            // Perform Perspective Transformation
            Core.perspectiveTransform(unWarpedValue, warpedValue, warpMat);
            // 更新转换后的按键坐标
            Point[] points = warpedValue.toArray();
            keyMap.put(key, points[0]);
        }

    }

    public List<Point> keypointSelection2(Mat oneFrame) {
        // 检测去手
        Mat handCrMat = new FingertipDetector(oneFrame).HandSegmentation();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
        Imgproc.dilate(handCrMat, handCrMat, element);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat(); //存储轮廓的拓扑结构
        Imgproc.findContours(handCrMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        if (contours.size() <= 2) {
            oneFrame.setTo(new Scalar(255,255,255), handCrMat);
        }
        Imgcodecs.imwrite("./tracking2/1-hand-" + frameCounter + ".jpg", oneFrame);

        // 键盘轮廓
        Mat cannyFrame = preProcessingFrame(oneFrame);
        KeystrokeDetector.saveFrame("canny", oneFrame);
        List<MatOfPoint> keyboardContours = new ArrayList<>();
        // 输出轮廓的拓扑结构信息
        hierarchy = new Mat();
        Imgproc.findContours(cannyFrame, keyboardContours, hierarchy, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint maxContour = keyboardContours.get(findMaxContour(keyboardContours));
        Point[] maxContourArray = maxContour.toArray();
        // 根据键盘边缘轮廓找到四个角点，用于过滤false轮廓
        List<Point> cornerList = findCornerPoints(maxContourArray);
        return cornerList;
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

    public List<Point> keypointSelection(Mat oneFrame) {
        // 检测去手
        Mat handCrMat = new FingertipDetector(oneFrame).HandSegmentation();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
        Imgproc.dilate(handCrMat, handCrMat, element);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat(); //存储轮廓的拓扑结构
        Imgproc.findContours(handCrMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        // 通过手的轮廓数量判断是否有手
        if (contours.size() <= 2) {
            oneFrame.setTo(new Scalar(255,255,255), handCrMat);
        }
        Imgcodecs.imwrite("./tracking2/1-hand-" + frameCounter + ".jpg", oneFrame);

        // 边缘检测
        Mat grayFrame = new Mat(), cannyFrame = new Mat();
        Imgproc.cvtColor(oneFrame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(grayFrame, cannyFrame, 40, 100);
        Imgcodecs.imwrite("./tracking2/2-canny-" + frameCounter + ".jpg", cannyFrame);

        // 霍夫变换
        Mat lines = new Mat();
        Imgproc.HoughLines(cannyFrame, lines, 1, PI/180, 100);
        Mat houghFrame = oneFrame.clone();
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0], theta = lines.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a*rho, y0 = b*rho;
            // 直线上的两个点
            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(houghFrame, pt1, pt2, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA, 0);
        }
        Imgcodecs.imwrite("./tracking2/3-hough-" + frameCounter + ".jpg", houghFrame);

        // 拟合直线
        List<Point> lineList = LinesFilter(lines);
        Mat lineFrame = oneFrame.clone();
        for (int i = 0; i < lineList.size(); i++) {
            double rho = lineList.get(i).x, theta = lineList.get(i).y;
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(lineFrame, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
            //Imgcodecs.imwrite("./input/kps_houghFrame" + i + ".jpg", cannyDstFrame);
            //System.out.println(i + "  rho = " + rho + "  theta = " + theta);
        }
        Imgcodecs.imwrite("./tracking2/4-line-" + frameCounter + ".jpg", lineFrame);

        // 四条直线
        Point y1 = new Point(0, 0), y2 = new Point(0, 0);
        Point x1 = new Point(0, 0), x2 = new Point(0, 0);
        for (int i = 0; i < lineList.size(); i++) {
            double rho = lineList.get(i).x, theta = lineList.get(i).y;
            if (1 < theta && theta < 2) {
                if (y1.x == 0 && y1.y == 0) y1 = lineList.get(i);
                if (y2.x == 0 && y2.y == 0) y2 = lineList.get(i);
                // y1是最上边的线，找距离最短的
                if (abs(rho) < abs(y1.x)) {
                    y2 = y1;
                    y1 = lineList.get(i);
                } else if (abs(rho) < abs(y2.x)) {
                    y2 = lineList.get(i);
                }
            } else {
                if (x1.x == 0 && x1.y == 0) x1 = lineList.get(i);
                if (x2.x == 0 && x2.y == 0) x2 = lineList.get(i);
                if (abs(rho) < abs(x1.x)) x1 = lineList.get(i);
                if (abs(rho) > abs(x2.x)) x2 = lineList.get(i);
            }
        }

        // 四个交点
        List<Point> keypoints = new ArrayList<>();
        keypoints.add(getCrossPoint(get2Point(y1)[0], get2Point(y1)[1], get2Point(x1)[0], get2Point(x1)[1]));
        keypoints.add(getCrossPoint(get2Point(y1)[0], get2Point(y1)[1], get2Point(x2)[0], get2Point(x2)[1]));
        keypoints.add(getCrossPoint(get2Point(y2)[0], get2Point(y2)[1], get2Point(x1)[0], get2Point(x1)[1]));
        keypoints.add(getCrossPoint(get2Point(y2)[0], get2Point(y2)[1], get2Point(x2)[0], get2Point(x2)[1]));
        Mat crossFrame = oneFrame.clone();
        for (Point point : keypoints) {
            Imgproc.circle(crossFrame, point, 5, new Scalar(0, 0, 255), -1);
        }
        Imgcodecs.imwrite("./tracking2/5-cross-" + frameCounter + ".jpg", crossFrame);
        frameCounter++;

        return keypoints;
    }

    private List<Point> LinesFilter(Mat lines) {
        List<Point> newLineList = new LinkedList<>();
        for (int i = 0; i < lines.rows(); i++) {
            double rho = lines.get(i, 0)[0], theta = lines.get(i, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double k = -a/b, b1 = rho/b, absk = abs(k);
            if ((absk > 0.1 && absk < 5) || theta == 0)
                continue;
            if (newLineList.isEmpty() || isNewLine(rho, theta, newLineList)) {
                newLineList.add(new Point(rho, theta));
            }
        }
        return newLineList;
    }

    private boolean isNewLine(double rho, double theta, List<Point> newLineList) {
        theta = (double) Math.round(theta * 1000) / 1000;
        for(int i = 0; i < newLineList.size(); i++) {
            double rhoTmp = newLineList.get(i).x, thetaTmp = newLineList.get(i).y;
            // rho和theta都很近
            double rhoDis = abs(rho - rhoTmp);
            double thetaDis = abs(theta-thetaTmp);
            if (thetaDis > 1.57) thetaDis = 3.14 - thetaDis;
            if (rhoDis < 30 && thetaDis < 1)
                return false;

            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho, x1 = x0 + 1000 * (-b), y1 = y0 + 1000 * (a);
            a = Math.cos(thetaTmp); b = Math.sin(thetaTmp);
            double x2 = a * rhoTmp, y2 = b * rhoTmp, x3 = x2 + 1000 * (-b), y3 = y2 + 1000 * (a);
            // 计算交点
            Point crossPoint = getCrossPoint(new Point(x0, y0), new Point(x1, y1), new Point(x2, y2), new Point(x3, y3));
            // 计算夹角
            double dot = (x1-x0)*(x3-x2) + (y1-y0)*(y3-y2);
            double mo1 = sqrt( pow(x1-x0, 2) + pow(y1-y0, 2));
            double mo2 = sqrt( pow(x3-x2, 2) + pow(y3-y2, 2));
            double angle = acos(dot / (mo1 * mo2))*180/PI;
            if ((angle < 50 || angle > 130) && (crossPoint.x > 0 && crossPoint.x < 800 && crossPoint.y > 0 && crossPoint.y < 480))
                return false;
        }
        return true;
    }

    // 得到一条直线上的两个点
    private Point[] get2Point(Point line) {
        double rho = line.x, theta = line.y;
        double a = Math.cos(theta), b = Math.sin(theta);
        double x0 = a * rho, y0 = b * rho;
        double x1 = x0 - 1000 * (-b), y1 = y0 - 1000 * (a);
        double x2 = x0 + 1000 * (-b), y2 = y0 + 1000 * (a);
        return new Point[]{new Point(x1, y1), new Point(x2, y2)};
    }

    private Point getCrossPoint(Point p1, Point p2, Point p3, Point p4) {
        Point crossPoint = new Point();
        if (p1.x == p2.x) {
            // Line1无斜率
            crossPoint.x = p1.x;
            crossPoint.y = (p3.y == p4.y)? p3.y : (p3.y-p4.y)*(crossPoint.x-p3.x)/(p3.x-p4.x)+p3.y;
        } else if (p3.x == p4.x) {
            // Line2 无斜率
            crossPoint.x = p3.x;
            crossPoint.y = (p1.y == p2.y)? p1.y : (p1.y-p2.y)*(crossPoint.x-p1.x)/(p1.x-p2.x)+p1.y;
        } else {
            // 都有斜率
            double k1 = (p2.y-p1.y)/(p2.x-p1.x);
            double b1 = p1.y-k1*p1.x;
            double k2 = (p4.y-p3.y)/(p4.x-p3.x);
            double b2 = p3.y-k2*p3.x;
            // k1x+b1 = k2x+b2, x=(b2-b1)/(k1-k2)
            crossPoint.x = (b2-b1)/(k1-k2);
            crossPoint.y = k1*crossPoint.x + b1;
        }
        return crossPoint;
    }
}
