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

import static com.example.hao.ocvcamera.MainActivity.isWrongTransformation;
import static com.example.hao.ocvcamera.MainActivity.keyMap;
import static com.example.hao.ocvcamera.MainActivity.keyboardArea;
import static com.example.hao.ocvcamera.MainActivity.keyboardLeftDown;
import static com.example.hao.ocvcamera.MainActivity.keyboardLeftUp;
import static com.example.hao.ocvcamera.MainActivity.keyboardRightDown;
import static com.example.hao.ocvcamera.MainActivity.keyboardRightUp;
import static com.example.hao.ocvcamera.MainActivity.lastFingerTip;
import static com.example.hao.ocvcamera.MainActivity.lastStatus;
import static com.example.hao.ocvcamera.MainActivity.preKeystrokeFingerTip;
import static java.lang.Math.PI;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class KeystrokeTask {
    private static final String TAG = "KeystrokeTask";

    // 帧的计数
    private int frameCounter;
    // 当前帧
    private Mat inputFrame;
    // 帧的宽度和高度
    private int frameWidth, frameHeight;

    // 提取的"按键-坐标"映射
    public static Map<String, Point> keyMap = new HashMap<>();

    //键盘的四个角点
    public static Point keyboardLeftUp = new Point(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
    public static Point keyboardRightUp = new Point(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    public static Point keyboardLeftDown = new Point(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
    public static Point keyboardRightDown = new Point(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

    public KeystrokeTask(int count, Mat oneFrame) {
        frameCounter = count;
        frameWidth = oneFrame.width();
        frameHeight = oneFrame.height();
        inputFrame = oneFrame.clone();
    }

    // 通过透视变换更新keyMap
    public void UpdateKeyMap() {
        Mat mat = new Mat();
        Mat mat2 = new Mat();

        //saveFrame("移动前的帧", mat);
        //saveFrame("移动后的帧", mat2);
        Utils.bitmapToMat(MainActivity.frameBeforeMoving, mat);
        Utils.bitmapToMat(MainActivity.frameAfterMoving, mat2);

        long startTracking = System.nanoTime(); // 为了测试tracking delay

        // 获取移动前后的帧的对应四个点，得到变换矩阵
        MatOfPoint2f srcQuad = new MatOfPoint2f();
        Point[] src4Points = get4Points(mat);
        srcQuad.fromArray(src4Points);
        MatOfPoint2f dstQuad = new MatOfPoint2f();
        Point[] dst4Points = get4Points(mat2);
        dstQuad.fromArray(dst4Points);
        Mat warpMat = Imgproc.getPerspectiveTransform(srcQuad, dstQuad);

        // keyMap透视变换
        for (String key : keyMap.keySet()) {
            Point[] valArray = new Point[1];
            valArray[0] = keyMap.get(key);
            MatOfPoint2f unWarpedValue = new MatOfPoint2f();
            unWarpedValue.fromArray(valArray);
            MatOfPoint2f warpedValue = new MatOfPoint2f();
            Core.perspectiveTransform(unWarpedValue, warpedValue, warpMat); // 执行透视变换
            keyMap.put(key, warpedValue.toArray()[0]);
            Imgproc.circle(mat2, warpedValue.toArray()[0], 5, new Scalar(0, 0, 255), -1);
        }

        long stopTracking = System.nanoTime(); // 为了测试tracking delay
        // 变换上一次Keystroke的指尖，为了解决重复检测
        if (preKeystrokeFingerTip.x != 0 && preKeystrokeFingerTip.y != 0) {
            MatOfPoint2f unWarpedFingerTip = new MatOfPoint2f();
            unWarpedFingerTip.fromArray(preKeystrokeFingerTip);
            MatOfPoint2f warpedFingerTip = new MatOfPoint2f();
            Core.perspectiveTransform(unWarpedFingerTip, warpedFingerTip, warpMat); // 对指尖透视变换
            preKeystrokeFingerTip = warpedFingerTip.toArray()[0];
        }
        Log.i(TAG, "KeyStrokeDetection: 变换之后的lastFingerTip = " + preKeystrokeFingerTip);

        // 更新键盘四个角点
        keyboardLeftUp = dst4Points[0];
        keyboardRightUp = dst4Points[1];
        keyboardLeftDown = dst4Points[2];
        keyboardRightDown = dst4Points[3];

        float[] delay = {(stopTracking-startTracking)/1000000}; // 为了测试tracking delay
        //new Thread(new SensorDataSaver("ZKeyTrackDelay", System.currentTimeMillis(), delay)).start();

        IsWrongTransformation();

        // 为了测试tracking的accuracy
        /*
        for (String key: keyMap.keySet())
            preKeyMap.put(key, keyMap.get(key));
        KeyExtraction();
        if (!isWrongTransformation) {
            float[] data = {avgKeyOffset(preKeyMap, keyMap), avgKeyOffset(preKeyMap, keyMap)/12};
            new Thread(new SensorDataSaver("KeyTrackAcc", System.currentTimeMillis(), data)).start();
        }*/


        //saveFrame("透视变换后的键盘坐标" + System.currentTimeMillis(), mat2);
    }

    // 获取透视变换所需的四个点
    private Point[] get4Points(Mat oneFrame) {

        Mat digitImage = oneFrame.clone();
        // 预处理图像:阈值化&边缘检测，返回cannyImage
        Mat cannyImage = preProcessingMat(digitImage);

        // 找键盘轮廓
        List<MatOfPoint> keyboardContours = new ArrayList<>();
        Mat hierarchy = new Mat(); // 输出轮廓的拓扑结构信息
        Imgproc.findContours(cannyImage, keyboardContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE); // param1须为单通道二进制图像
        //Imgproc.drawContours(digitImage, keyboardContours, -1,  new Scalar(255, 0, 0));
        int contoursSize = keyboardContours.size();
        System.out.println(TAG + " 键盘轮廓原始数量 = " + contoursSize);
        MatOfPoint maxContour = keyboardContours.get(findMaxContour(keyboardContours)); // 得到最大轮廓
        Point[] maxContourArray = maxContour.toArray();
        double keyboardArea = Imgproc.contourArea(maxContour);

        //根据键盘边缘轮廓找到四个角点，用于过滤false轮廓
        return findCornerSelf(maxContourArray);
    }

    // 检测移除手，找剩余键盘的轮廓和凸包
    // 通过一系列过滤条件得到五个凸包点，相交得第四个角点
    // return: 交点+其他已知的三个点
    private Point[] get4PointsV3(Mat oneFrame) {
        Log.i(TAG, "UpdateKeyMap get4PointsV3: Begin!");

        // 检测手的轮廓和凸包
        Mat handMat = HandSegmentation(oneFrame);
        //saveFrame(System.currentTimeMillis()+"检测到的手", handMat);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(25, 25));
        Imgproc.dilate(handMat, handMat, element);

        List<MatOfPoint> handContours = new ArrayList<>();
        Mat handHierarchy = new Mat();
        Imgproc.findContours(handMat, handContours, handHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint handContourArray = handContours.get(findMaxContour(handContours)); // 找最大轮廓
        MatOfInt handHull = new MatOfInt(); // handHull保存凸包点在轮廓点集的下标
        Imgproc.convexHull(handContourArray, handHull); // 找凸包
        List<Point> polyPoint = new ArrayList<>(); // 保存检测到的凸包点
        for(int i : handHull.toArray())
            polyPoint.add(handContourArray.toArray()[i]);

        MatOfPoint x = new MatOfPoint();
        x.fromList(polyPoint);
        Mat handMat2 = handMat.clone();
        Imgproc.fillConvexPoly(handMat2, x, new Scalar(255,255,255));
        //saveFrame(System.currentTimeMillis()+"检测到的手凸包", handMat2);

        double percentArea = Imgproc.contourArea(handContourArray)/(frameWidth*frameHeight);
        if (percentArea > 0.9) {
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 最大轮廓的面积占比：" + percentArea + "  没有手！");
        } else {
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 最大轮廓的面积占比：" + percentArea + "  检测有手！");
        }
        if( percentArea < 0.4 ) { // 以轮廓面积判断是否检测到有手
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 手部凸包已移除");
            oneFrame.setTo(new Scalar(255, 255, 255), handMat2); // 移除手的凸包
        }
        saveFrame(System.currentTimeMillis()+"移除手凸包", oneFrame);

        Mat cannyImage = preProcessingMat(oneFrame);
        List<MatOfPoint> contours = new ArrayList<>(); // 输出的轮廓结果
        Mat hierarchy = new Mat(); // 输出的轮廓的拓扑结构信息
        Imgproc.findContours(cannyImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        int maxContourIdx = findMaxContour(contours);
        Imgproc.drawContours(oneFrame, contours, maxContourIdx, new Scalar(255, 0,0), 3);
        MatOfPoint maxContour = contours.get(maxContourIdx);
        Point[] maxContourArray = maxContour.toArray();

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(maxContour, hull);
        int[] hullArray = hull.toArray();
        for( int i : hullArray) {
            Imgproc.circle(oneFrame, maxContourArray[i], 5, new Scalar(0, 0, 255));
        }
        Imgcodecs.imwrite("/Users/zhanghao/Downloads/原始凸包点.jpg", oneFrame);
        saveFrame(System.currentTimeMillis()+"原始凸包点", oneFrame);

        // 条件一: 过滤附近点
        int nearThreshold = 10;
        List<Point> hullList = new ArrayList<>();
        hullList.add(maxContourArray[hullArray[0]]);
        for (int i = 1; i < hullArray.length; i++) {
            int idx = hullArray[i];
            if (distanceBetween(maxContourArray[idx], maxContourArray[hullArray[i-1]]) > nearThreshold)
                hullList.add(maxContourArray[idx]);
        }
        Mat nearPoints = oneFrame.clone();
        for(Point i : hullList)
            Imgproc.circle(nearPoints, i, 5, new Scalar(0, 0, 255), -1);
        Imgcodecs.imwrite("/Users/zhanghao/Downloads/过滤附近点.jpg", nearPoints);
        saveFrame("过滤附近点", nearPoints);

        // 条件二: 过滤平角的点，得到5个凸包点
        int pingThreshold = 170;
        List<Point> tempList= new ArrayList<>();
        for(int i = 0; i < hullList.size(); i++) {
            int theta = calcTheta(hullList, i, 1); //点乘和模得夹角
            if(theta < pingThreshold)
                tempList.add(hullList.get(i));
        }
        hullList = tempList;

        int filterThreshold = 50;
        List<Point> tempList2= new ArrayList<>();
        for(int i = 0; i < hullList.size(); i++) {
            if(distanceBetween(hullList.get(i), hullList.get((i+1)%hullList.size())) > filterThreshold)
                tempList2.add(hullList.get(i));
        }
        hullList = tempList2;

        Mat fiveHull = oneFrame.clone();
        for(Point i : hullList)
            Imgproc.circle(fiveHull, i, 5, new Scalar(0, 0, 255), -1);
        Imgcodecs.imwrite("/Users/zhanghao/Downloads/five hull.jpg", fiveHull);
        saveFrame("四个or五个凸包点", fiveHull);


        // 条件三: 找到两条线，相交得到第四个点
        int crossThreshold = 105;
        Point forthPoint = new Point();
        int hullListSize = hullList.size();
        for(int idx = 0; idx < hullListSize; idx++) {
            int theta = calcTheta(hullList, idx, 1); //点乘和模得夹角
            if(theta > crossThreshold) {
                int thetaPre = calcTheta(hullList, (idx-1+hullListSize)%hullListSize, 1);
                if(thetaPre > crossThreshold) {
                    //求交点
                    Point pointPre = hullList.get((idx-1+hullListSize)%hullListSize);
                    Point point = hullList.get(idx);
                    forthPoint = getCrossPoint(hullList.get((idx-2+hullListSize)%hullListSize),pointPre,point,hullList.get((idx+1)%hullListSize));
                    hullList.remove(pointPre);
                    hullList.remove(point);
                    hullList.add(forthPoint);
                    Imgproc.circle(oneFrame, point, 5, new Scalar(0, 255, 0), -1);
                    Imgproc.circle(oneFrame, pointPre, 5, new Scalar(0, 255, 0), -1);
                    break;
                } else {
                    Point pointNext = hullList.get((idx+1)%hullListSize);
                    Point point = hullList.get(idx);
                    forthPoint = getCrossPoint(hullList.get((idx-1+hullListSize)%hullListSize),point,pointNext,hullList.get((idx+2)%hullListSize));
                    hullList.remove(pointNext);
                    hullList.remove(point);
                    hullList.add(forthPoint);
                    Imgproc.circle(oneFrame, point, 5, new Scalar(0, 255, 0), -1);
                    Imgproc.circle(oneFrame, pointNext, 5, new Scalar(0, 255, 0), -1);
                    break;
                }
            }
        }
        Mat finalHull = oneFrame.clone();
        for(Point i : hullList)
            Imgproc.circle(finalHull, i, 10, new Scalar(0, 255, 255), -1);
        Imgproc.circle(finalHull, forthPoint, 10, new Scalar(0, 255, 255), -1);
        Imgcodecs.imwrite("/Users/zhanghao/Downloads/final hull.jpg", finalHull);
        saveFrame("最后得到的凸包点", finalHull);


        // 此时hullList中应该是四个点，按 左上/右上/左下/右下 排序
        if (hullList.size() != 4)
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 没有检测到四个点 " + hullList.size());
        sortByOneAxis(hullList, 'y');
        for(int i = 0; i <= hullList.size()-2; i = i+2) {
            sortByOneAxis(hullList.subList(i,i+2), 'x');
        }

        Point[] fourPoints = new Point[4];
        int idx = 0;
        for(int i = 0; i < 4; i++) { // 将hullList转换为Point[]的fourPoints
            Point p = hullList.get(i);
            fourPoints[idx] = p;
            Imgproc.circle(oneFrame, p, 5, new Scalar(0, 255, 0), -1);
            Log.i(TAG, "UpdateKeyMap get4PointsV3: " + p);
            idx++;
        }
        // 用四个点更新keyboard的四个角点
        keyboardLeftUp = hullList.get(0);
        keyboardRightUp = hullList.get(1);
        keyboardLeftDown = hullList.get(2);
        keyboardRightDown = hullList.get(3);
        Log.i(TAG, "UpdateKeyMap get4PointsV3: End！");
        return fourPoints;
    }

    private Point[] get4PointsV4(Mat oneFrame) {
        Log.i(TAG, "UpdateKeyMap get4PointsV4: Begin!");

        Mat handMat = HandSegmentation(oneFrame); // 检测手
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(handMat, handMat, element);
        saveFrame(System.currentTimeMillis()+"检测到的手", handMat);
        List<MatOfPoint> handContours = new ArrayList<>();
        Mat handHierarchy = new Mat();
        Imgproc.findContours(handMat, handContours, handHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint handContourArray = handContours.get(findMaxContour(handContours)); // 找最大轮廓
        MatOfInt handHull = new MatOfInt(); // handHull保存凸包点在轮廓点集的下标
        Imgproc.convexHull(handContourArray, handHull); // 找凸包
        List<Point> handHullPoint = new ArrayList<>(); // 保存检测到的凸包点
        for(int i : handHull.toArray())
            handHullPoint.add(handContourArray.toArray()[i]);

        MatOfPoint handHullMat = new MatOfPoint();
        handHullMat.fromList(handHullPoint);
        Mat handMat2 = handMat.clone();
        Imgproc.fillConvexPoly(handMat2, handHullMat, new Scalar(255,255,255)); // 凸包内填充为白色
        //saveFrame(System.currentTimeMillis()+"检测到的手凸包", handMat2);

        double percentArea = Imgproc.contourArea(handContourArray)/(frameWidth*frameHeight);
        if (percentArea > 0.9) {
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 最大轮廓的面积占比：" + percentArea + "  没有手！");
        } else {
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 最大轮廓的面积占比：" + percentArea + "  检测有手！");
        }
        if( percentArea < 0.4 ) { // 以轮廓面积判断是否检测到有手
            Log.i(TAG, "UpdateKeyMap get4PointsV3: 手部凸包已移除");
            oneFrame.setTo(new Scalar(255, 255, 255), handMat2); // 移除手的凸包
        }
        saveFrame(System.currentTimeMillis()+"移除手凸包", oneFrame);

        Mat cannyImage = preProcessingMat(oneFrame);
        //blurImage.convertTo(oneFrame, CvType.CV_8UC3);
        List<MatOfPoint> contours = new ArrayList<>(); // 输出的轮廓结果
        Mat hierarchy = new Mat(); // 输出的轮廓的拓扑结构信息
        Imgproc.findContours(cannyImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        int maxContourIdx = findMaxContour(contours);
        Imgproc.drawContours(oneFrame, contours, maxContourIdx, new Scalar(255, 0,0), 3);
        MatOfPoint maxContour = contours.get(maxContourIdx);
        Point[] maxContourArray = maxContour.toArray();

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(maxContour, hull);
        int[] hullArray = hull.toArray();
        List<Point> restKbdHullPoint = new ArrayList<>();
        for( int i : hullArray) {
            restKbdHullPoint.add(maxContourArray[i]);
            Imgproc.circle(oneFrame, maxContourArray[i], 5, new Scalar(0, 0, 255));
        }
        Imgcodecs.imwrite("/Users/zhanghao/Downloads/原始凸包点.jpg", oneFrame);
        saveFrame(System.currentTimeMillis()+"原始凸包点", oneFrame);

        MatOfPoint restKbdHullMat = new MatOfPoint();
        restKbdHullMat.fromList(restKbdHullPoint);
        //Imgproc.fillConvexPoly(oneFrame, restKbdHullMat, new Scalar(255,255,255)); // 凸包内填充为白色
        Mat tmpMat = new Mat(frameHeight, frameWidth, CvType.CV_8UC3, new Scalar(255,255,255));
        for (int i = 1 ; i < restKbdHullPoint.size(); i++) {
            Imgproc.line(tmpMat, restKbdHullPoint.get(i), restKbdHullPoint.get(i-1), new Scalar(0, 0, 0), 1);
        }
        Imgproc.line(tmpMat, restKbdHullPoint.get(0), restKbdHullPoint.get(restKbdHullPoint.size()-1), new Scalar(0, 0, 0), 1);
        saveFrame(System.currentTimeMillis()+"填充键盘后的图", tmpMat);

        cannyImage = preProcessingMat(tmpMat);
        saveFrame(System.currentTimeMillis()+"填充键盘后的图canny", cannyImage);
        // 霍夫变换检测直线
        Mat lines = new Mat();
        Imgproc.HoughLines(cannyImage, lines, 1, PI/180, 50);
        //Log.i(TAG, "get4PointsV4: 霍夫变换检测到的直线" + lines.rows() + " " + lines.cols());
        List<MyLineClass> linesList = new ArrayList<>();
        for(int i = 0; i < lines.rows(); i++) {
            double[] vec = lines.get(i,0);
            double rho = vec[0], theta = vec[1];
            Point pt1 = new Point(), pt2 = new Point();
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = Math.round(x0 + 1000 * (-b));
            pt1.y = Math.round(y0 + 1000 * (a));
            pt2.x = Math.round(x0 - 1000 * (-b));
            pt2.y = Math.round(y0 - 1000 * (a));
            double k = (pt2.y-pt1.y)/(pt2.x-pt1.x);
            Log.i(TAG, "get4PointsV4: 斜率 " + k + " rho " + rho + " theta " + theta);
            Imgproc.line(tmpMat, pt1, pt2, new Scalar(0,0,255));
            MyLineClass lineFormat = new MyLineClass(k, rho, theta, pt1, pt2);

            if (k == Double.POSITIVE_INFINITY || k == Double.NEGATIVE_INFINITY)
                continue;
            // 根据向量的夹角判断是否属于同一簇
            if (linesList.isEmpty()) {
                linesList.add(lineFormat);
            } else {
                // 和list中的其他线求夹角
                int nearLineIdx = containsNearLine(lineFormat, linesList);
                if (nearLineIdx != -1) {
                    // 求平均
                    MyLineClass elem = linesList.remove(nearLineIdx); // List.get()得到的是值，而非地址
                    //elem.k = (k+elem.k)/2;
                    //elem.rho = (rho+elem.rho)/2;
                    //elem.theta = (theta+elem.theta)/2;
                    //elem.beginPoint = new Point((pt1.x+elem.beginPoint.x)/2, (pt1.y+elem.beginPoint.y)/2);
                    //elem.endPoint = new Point((pt2.x+elem.endPoint.x)/2, (pt2.y+elem.endPoint.y)/2);
                    linesList.add(elem);
                } else {
                    linesList.add(lineFormat);
                }
            }
        }

        saveFrame(System.currentTimeMillis()+"填充键盘后的图Hough检测1", tmpMat);
        for (MyLineClass line : linesList) {
            Log.i(TAG, "get4PointsV4: 五条线的斜率为" + line.k);
            if (line.k == 0)
                Imgproc.line(tmpMat, line.beginPoint, line.endPoint, new Scalar(255, 0, 0), 3);
            else if (abs(line.k) > 10)
                Imgproc.line(tmpMat, line.beginPoint, line.endPoint, new Scalar(255, 0, 0), 3);
            else
                Imgproc.line(tmpMat, line.beginPoint, line.endPoint, new Scalar(255, 0, 255), 3);
        }
        saveFrame(System.currentTimeMillis()+"填充键盘后的图Hough检测2", tmpMat);

        int linesSize = linesList.size();
        for(int i = 0; i < linesSize; i++) {
            int minIdx = i;
            for(int j = i+1; j < linesSize; j++) {
                if ( abs(linesList.get(j).k) < abs(linesList.get(minIdx).k))
                    minIdx = j;
            }
            // swap
            MyLineClass tmpElem = linesList.get(minIdx);
            linesList.set(minIdx, linesList.get(i));
            linesList.set(i, tmpElem);
        }
        Log.i(TAG, "get4PointsV4: linesSize = " + linesSize);
        if (linesSize == 5)
            linesList.remove(2);


        List<Point> fourPointsList = new ArrayList<>();
        fourPointsList.add( getCrossPoint(linesList.get(0).beginPoint, linesList.get(0).endPoint, linesList.get(2).beginPoint, linesList.get(2).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(0).beginPoint, linesList.get(0).endPoint, linesList.get(3).beginPoint, linesList.get(3).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(1).beginPoint, linesList.get(1).endPoint, linesList.get(2).beginPoint, linesList.get(2).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(1).beginPoint, linesList.get(1).endPoint, linesList.get(3).beginPoint, linesList.get(3).endPoint));

        sortByOneAxis(fourPointsList, 'y');
        for(int i = 0; i <= fourPointsList.size()-2; i = i+2) {
            sortByOneAxis(fourPointsList.subList(i,i+2), 'x');
        }
        Point[] fourPoints = new Point[4];
        int idx = 0;
        for(int i = 0; i < 4; i++) { // 将hullList转换为Point[]的fourPoints
            Point p = fourPointsList.get(i);
            fourPoints[idx] = p;
            Imgproc.circle(tmpMat, p, 10, new Scalar(0, 255, 0 ), -1);
            Log.i(TAG, "UpdateKeyMap get4PointsV4: " + p);
            idx++;
        }
        saveFrame(System.currentTimeMillis()+"填充键盘后最后确定的点", tmpMat);
        return fourPoints;
    }

    // 膨胀、凸包、线长和角度
    private Point[] get4PointsV5(Mat oneFrame) {
        timeStamp = System.currentTimeMillis();
        Log.i(TAG, timeStamp+"get4PointsV5: Begin!");

        // 检测手并膨胀
        saveFrame(timeStamp + "get4PointsV5 原始图", inputFrame);
        Mat handCrMat = HandSegmentation(oneFrame);
        Mat handMat = oneFrame.clone();
        handMat.setTo(new Scalar(255,255,255), handCrMat);
        saveFrame(timeStamp + "get4PointsV5 检测的手", handMat);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
        Imgproc.dilate(handCrMat, handCrMat, element);
        handMat.setTo(new Scalar(255,255,255), handCrMat);
        saveFrame(timeStamp + "get4PointsV5 手膨胀效果", handMat);

        // 找手的轮廓和凸包
        List<MatOfPoint> handContours = new ArrayList<>();
        Mat handHierarchy = new Mat();
        Imgproc.findContours(handCrMat, handContours, handHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint handContourArray = handContours.get(findMaxContour(handContours)); // 找最大轮廓
        MatOfInt handHull = new MatOfInt(); // handHull保存凸包点在轮廓点集的下标
        Imgproc.convexHull(handContourArray, handHull); // 找凸包
        List<Point> handHullPoint = new ArrayList<>(); // 保存检测到的凸包点
        for(int i : handHull.toArray()) {
            handHullPoint.add(handContourArray.toArray()[i]);
            //Imgproc.circle(oneFrame, handContourArray.toArray()[i], 5, new Scalar(255, 0 ,0), -1);
        }
        // 手的凸包内填充为白色
        MatOfPoint handHullMat = new MatOfPoint();
        handHullMat.fromList(handHullPoint);
        Mat handMat2 = handCrMat.clone();
        Imgproc.fillConvexPoly(handMat2, handHullMat, new Scalar(255,255,255));
        //saveFrame(System.currentTimeMillis()+"检测到的手凸包", handMat2);

        double percentArea = Imgproc.contourArea(handContourArray)/(frameWidth*frameHeight);
        if( percentArea < 0.25 ) { // 以轮廓面积判断是否检测到有手
            Log.i(TAG, timeStamp + "get4PointsV5: 手面积占比：" + percentArea + "  检测有手！手部凸包已移除");
            oneFrame.setTo(new Scalar(255, 255, 255), handMat2); // 移除手的凸包
        } else {
            Log.i(TAG, timeStamp + "get4PointsV5: 手的面积占比：" + percentArea + "  没有手！");
        }
        saveFrame(timeStamp + "get4PointsV5 移除手凸包，接下来真正处理", oneFrame);

        // 此处得到的是已经去除手、真正需要处理的图片
        Mat cannyImage = preProcessingMat(oneFrame);
        // 找剩余键盘的轮廓
        List<MatOfPoint> contours = new ArrayList<>(); // 输出的轮廓结果
        Mat hierarchy = new Mat(); // 输出的轮廓的拓扑结构信息
        Imgproc.findContours(cannyImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        int maxContourIdx = findMaxContour(contours);
        //Imgproc.drawContours(oneFrame, contours, maxContourIdx, new Scalar(255, 0,0), 3);
        MatOfPoint maxContour = contours.get(maxContourIdx);
        Point[] maxContourArray = maxContour.toArray();
        // 找剩余键盘轮廓的凸包
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(maxContour, hull);
        int[] hullArray = hull.toArray();
        List<Point> restKbdHullPoint = new ArrayList<>();
        for( int i : hullArray) {
            restKbdHullPoint.add(maxContourArray[i]);
            //Imgproc.circle(oneFrame, maxContourArray[i], 3, new Scalar(0, 0, 255), -1);
        }
        saveFrame(timeStamp + "getPointsV5原始凸包点", oneFrame);

        // 此处得到所有的凸包点，从这些凸包点中去粗存精得到真正的4个or5个点
        Point beginPoint = restKbdHullPoint.get(0);
        Point endPoint = restKbdHullPoint.get(1);
        int j = 2;
        while ( lineLength(beginPoint, endPoint) < 10) {
            //beginPoint = new Point( (beginPoint.x+endPoint.x)/2, (beginPoint.y+endPoint.y)/2 );
            endPoint = restKbdHullPoint.get(j);
            j++;
        }
        //Imgproc.line(oneFrame, beginPoint, endPoint, new Scalar(255,0,0),2);
        List<Point> trueKbdHullPoint = new ArrayList<>();
        for ( ; j < restKbdHullPoint.size(); j++) {
            Point point = restKbdHullPoint.get(j);
            double lineAngle = lineAngle(beginPoint, endPoint, endPoint, point);
            double lineLength = lineLength(point, endPoint);
            Log.i(TAG, timeStamp + "get4PointsV5: lineAngle " + lineAngle);
            Log.i(TAG, timeStamp + "get4PointsV5: lineLength " + lineLength);
            if (lineAngle < 8) {
                //endPoint = new Point( (endPoint.x+point.x)/2, (endPoint.y+point.y)/2);
                endPoint = point;
                Log.i(TAG, timeStamp+" get4PointsV5: 角度<15，" + lineAngle + " endPoint延长到" + endPoint);
            } else if (lineLength < 20) {
                //endPoint = new Point( (endPoint.x+point.x)/2, (endPoint.y+point.y)/2);
                Log.i(TAG, timeStamp+" get4PointsV5: 距离<20 " + lineLength(point, endPoint));
            } else {
                trueKbdHullPoint.add(beginPoint);
                beginPoint = endPoint;
                endPoint = point;
                Log.i(TAG, timeStamp+" get4PointsV5: beginPoint存入，beginPoint为" + beginPoint + " endPoint为" + endPoint);
            }
        }
        Point lastPoint = trueKbdHullPoint.get(trueKbdHullPoint.size()-1);
        if (lastPoint.x == beginPoint.x && lastPoint.y == beginPoint.y) {
            Log.i(TAG, timeStamp+"get4PointsV5: 已加入beginPoint " + beginPoint + " 现在加入endPoint：" + endPoint);
            trueKbdHullPoint.add(endPoint);
        } else {
            Log.i(TAG, timeStamp+" get4PointsV5: 没有加入beginPoint " + beginPoint);
            trueKbdHullPoint.add(beginPoint);
        }

        while (trueKbdHullPoint.size() > 5) {
            removeMaxTheta(trueKbdHullPoint);
        }

        int trueKbdHullPointSize = trueKbdHullPoint.size();
        Log.i(TAG, timeStamp+"get4PointsV5: 凸包点的个数" + trueKbdHullPointSize);
        for( Point p : trueKbdHullPoint) {
            //Imgproc.circle(oneFrame, p, 5, new Scalar(255, 0, 255), -1);
        }
        saveFrame(timeStamp+"最后凸包点getPointsV5", oneFrame);

        List<MyLineClass> linesList = new ArrayList<>();
        for (int i = 0; i < trueKbdHullPointSize; i++) {
            Point start = trueKbdHullPoint.get(i);
            Point stop = trueKbdHullPoint.get((i + 1) % trueKbdHullPointSize);
            double slope = (stop.y - start.y) / (stop.x - start.x);
            linesList.add(new MyLineClass(slope, start, stop));
        }

        int linesSize = linesList.size();
        for (int i = 0; i < linesSize; i++) {
            int minIdx = i;
            for (int t = i + 1; t < linesSize; t++) {
                if (abs(linesList.get(t).k) < abs(linesList.get(minIdx).k))
                    minIdx = t;
            }

            // swap
            MyLineClass tmpElem = linesList.get(minIdx);
            linesList.set(minIdx, linesList.get(i));
            linesList.set(i, tmpElem);
        }
        if (linesSize == 5)
            linesList.remove(2);

        List<Point> fourPointsList = new ArrayList<>();
        fourPointsList.add( getCrossPoint(linesList.get(0).beginPoint, linesList.get(0).endPoint, linesList.get(2).beginPoint, linesList.get(2).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(0).beginPoint, linesList.get(0).endPoint, linesList.get(3).beginPoint, linesList.get(3).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(1).beginPoint, linesList.get(1).endPoint, linesList.get(2).beginPoint, linesList.get(2).endPoint));
        fourPointsList.add( getCrossPoint(linesList.get(1).beginPoint, linesList.get(1).endPoint, linesList.get(3).beginPoint, linesList.get(3).endPoint));

        sortByOneAxis(fourPointsList, 'y');
        for(int i = 0; i <= fourPointsList.size()-2; i = i+2) {
            sortByOneAxis(fourPointsList.subList(i,i+2), 'x');
        }

        Point[] fourPoints = new Point[4];
        int idx = 0;
        for(int i = 0; i < 4; i++) { // 将hullList转换为Point[]的fourPoints
            Point p = fourPointsList.get(i);
            fourPoints[idx] = p;
            Imgproc.circle(oneFrame, p, 5, new Scalar(255, 0, 0 ), -1);
            Log.i(TAG, timeStamp+"get4PointsV5: 找到的四个点" + p);
            idx++;
        }
        for (Point p : keyMap.values())
            Imgproc.circle(oneFrame, p, 5, new Scalar(0, 0, 255), -1);
        //Imgproc.circle(oneFrame, maxContourArray[hullArray[0]], 5, new Scalar(0, 255, 255), -1); // 墨绿色
        //Imgproc.circle(oneFrame, maxContourArray[hullArray[1]], 5, new Scalar(255, 255, 0), -1); // 黄色
        saveFrame(timeStamp+"最后四个点getPointsV5", oneFrame);
        return fourPoints;
    }

    private void removeMaxTheta(List<Point> list) {

        int maxThetaIdx = 0;
        int maxTheta = -1;
        for(int i = 1; i < list.size()-1; i++) {
            int theta = calcTheta(list, i, 1);
            if (theta > maxTheta) {
                maxTheta = theta;
                maxThetaIdx = i;
            }
        }
        list.remove(maxThetaIdx);
    }

    private int containsNearLine(MyLineClass lineFormat, List<MyLineClass> linesList) {
        int nearLineIdx = -1;
        for (MyLineClass line : linesList) {
            Point vector1 = new Point(lineFormat.endPoint.x-lineFormat.beginPoint.x, lineFormat.endPoint.y-lineFormat.beginPoint.y);
            Point vector2 = new Point(line.endPoint.x-line.beginPoint.x, line.endPoint.y-line.beginPoint.y);
            double dot = vector1.x*vector2.x + vector1.y*vector2.y;
            double mo1 = Math.sqrt( pow(vector1.x, 2) + pow(vector1.y, 2));
            double mo2 = Math.sqrt( pow(vector2.x, 2) + pow(vector2.y, 2));
            double angle = Math.acos(dot / (mo1 * mo2))*180/ PI;

            Log.i(TAG, "get4PointsV4 : 夹角 " + angle);
            Log.i(TAG, "get4PointsV4 : rho差距 " + abs(abs(lineFormat.rho)- abs(line.rho)));
            // 角度符合，并且rho值也很靠近
            if ( (angle < 15 || angle > 165) && abs(abs(lineFormat.rho)- abs(line.rho)) < 100 ) {
                //点乘和模得夹角
                nearLineIdx = linesList.indexOf(line);
                break;
            }
        }
        for (MyLineClass line : linesList)
            Log.i(TAG, "get4PointsV4 containsNearLine: 当前的list中" + line.k + " " + line.rho + " " + line.theta);
        Log.i(TAG, "get4PointsV4: nearLineIdx " + nearLineIdx);
        return nearLineIdx;
    }

    private Point getCrossPoint(Point p1, Point p2, Point p3, Point p4) {
        Log.i(TAG, "getCrossPoint:  Line1: " + p1 + "-" + p2 + "  Line2: " + p3 + "-"+ p4);
        Point crossPoint = new Point();
        if (p1.x == p2.x) { // Line1无斜率
            crossPoint.x = p1.x;
            crossPoint.y = (p3.y == p4.y)? p3.y :
                    (p3.y-p4.y)*(crossPoint.x-p3.x)/(p3.x-p4.x)+p3.y;
        } else if (p3.x == p4.x) { // Line2 无斜率
            crossPoint.x = p3.x;
            crossPoint.y = (p1.y == p2.y)? p1.y :
                    (p1.y-p2.y)*(crossPoint.x-p1.x)/(p1.x-p2.x)+p1.y;
        } else { // 都有斜率
            double k1 = (p2.y-p1.y)/(p2.x-p1.x);
            double b1 = p1.y-k1*p1.x;
            double k2 = (p4.y-p3.y)/(p4.x-p3.x);
            double b2 = p3.y-k2*p3.x;
            crossPoint.x = (b2-b1)/(k1-k2); // k1x+b1 = k2x+b2, x=(b2-b1)/(k1-k2)
            crossPoint.y = k1*crossPoint.x + b1;
        }
        Log.i(TAG, "getCrossPoint: 交点为" + crossPoint);
        return crossPoint;
    }

    private double lineLength(Point p1, Point p2) {
        return sqrt( pow(p2.x-p1.x, 2)+pow(p2.y-p1.y, 2) );
    }

    private double lineAngle(Point p1, Point p2, Point p3, Point p4) {
        Point vector1 = new Point(p2.x-p1.x, p2.y-p1.y);
        Point vector2 = new Point(p4.x-p3.x, p4.y-p3.y);
        double dot = vector1.x*vector2.x + vector1.y*vector2.y;
        double mo1 = Math.sqrt( pow(vector1.x, 2) + pow(vector1.y, 2));
        double mo2 = Math.sqrt( pow(vector2.x, 2) + pow(vector2.y, 2));
        return Math.acos(dot / (mo1 * mo2))*180/ PI;
    }

    private void IsWrongTransformation() {
        List<Point> pointArr = new ArrayList<>();
        pointArr.add(keyboardLeftUp);
        pointArr.add(keyboardRightUp);
        pointArr.add(keyboardRightDown);
        pointArr.add(keyboardLeftDown);
        MatOfPoint pointMat = new MatOfPoint();
        pointMat.fromList(pointArr);
        double currentKeyboardArea = Imgproc.contourArea(pointMat); // 根据变换得到的四个角点，算键盘的面积
        Log.w(TAG, timeStamp+"IsWrongTransformation: 原始键盘面积=" + keyboardArea);
        Log.w(TAG, timeStamp+"IsWrongTransformation: 当前键盘面积=" + currentKeyboardArea + "所占比=" + currentKeyboardArea/keyboardArea);

        double dot = (keyboardRightUp.x-keyboardLeftUp.x )*(keyboardRightDown.x-keyboardLeftDown.x)
                + (keyboardRightUp.y-keyboardLeftUp.y )*(keyboardRightDown.y-keyboardLeftDown.y);
        double mo1 = Math.sqrt( pow(keyboardRightUp.x-keyboardLeftUp.x, 2) + pow(keyboardRightUp.y-keyboardLeftUp.y, 2));
        double mo2 = Math.sqrt( pow(keyboardRightDown.x-keyboardLeftDown.x, 2) + pow(keyboardRightDown.y-keyboardLeftDown.y, 2));
        double horizontalAngle = Math.acos(dot / (mo1 * mo2))*180/ PI; //点乘和模得夹角
        dot = (keyboardLeftDown.x-keyboardLeftUp.x )*(keyboardRightDown.x-keyboardRightUp.x)
                + (keyboardLeftDown.y-keyboardLeftUp.y )*(keyboardRightDown.y-keyboardRightUp.y);
        mo1 = Math.sqrt( pow(keyboardLeftDown.x-keyboardLeftUp.x, 2) + pow(keyboardLeftDown.y-keyboardLeftUp.y, 2));
        mo2 = Math.sqrt( pow(keyboardRightDown.x-keyboardRightUp.x, 2) + pow(keyboardRightDown.y-keyboardRightUp.y, 2));
        double verticalAngle = Math.acos(dot / (mo1 * mo2))*180/ PI; //点乘和模得夹角
        Log.w(TAG, timeStamp+"IsWrongTransformation: 水平和垂直夹角=" + horizontalAngle + " " + verticalAngle);

        // 根据变换后键盘的面积和对边的夹角判断
        isWrongTransformation =  (currentKeyboardArea > 1.4*keyboardArea || currentKeyboardArea < 0.6*keyboardArea
                || horizontalAngle > 10 || verticalAngle > 10);
        if (isWrongTransformation)
            Log.e(TAG, timeStamp+"IsWrongTransformation: 检测到错误");
    }

    private float avgKeyOffset(Map<String, Point> m1, Map<String, Point> m2) {
        double sumDis = 0;
        for (String key: m1.keySet())
            sumDis += KeystrokeTask.distanceBetween(m1.get(key), m2.get(key));
        return (float)sumDis;
    }

    /** 以下是 Key Exaction 部分 */

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

    /** 以下是 Fingertip Detection 部分 */

    // 从帧画面中分割手部区域
    private Mat HandSegmentation(Mat oneFrame) {
        Mat YCrCbMat = new Mat();
        Imgproc.cvtColor(oneFrame, YCrCbMat, Imgproc.COLOR_BGR2YCrCb);
        //法一：通过YCrCb阈值检测肤色
        /*for (int i = 0; i < YCrCbMat.height(); i++) {
            for (int j = 0; j < YCrCbMat.width(); j++) {
                double[] color = YCrCbMat.get(i, j);
                if (color[1] > 133 && color[1] < 173 && color[2] > 77 && color[2] < 127)
                    Imgproc.circle(handImage, new Point(j, i), 1, new Scalar(0, 0, 255), -1);
            }
        }*/
        //法二：Cr + Otsu 方法，提取Cr分量 & Ostu分量阈值化
        Mat CrMat = new Mat();
        // 分离通道，提取Cr分量
        Core.extractChannel(YCrCbMat, CrMat, 1);
        //阈值化操作
        Imgproc.threshold(CrMat, CrMat, 0, 255, Imgproc.THRESH_BINARY|Imgproc.THRESH_OTSU);
        // 形态学腐蚀和膨胀
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
        Imgproc.erode(CrMat, CrMat, element);
        Imgproc.dilate(CrMat, CrMat, element);
        // OpenCV4PC手是白色，OpenCV4Android手是黑色；如果不反过来，后面指尖提取会出问题
        Imgproc.threshold(CrMat, CrMat, 100, 255, Imgproc.THRESH_BINARY_INV);
        return CrMat;
    }

    // 根据手的轮廓，检测按键的手指指尖
    private Point TipDetection(Mat CrMat) {
        Point finalTip = new Point(0,0); //最后确定的指尖
        Mat handImage = new Mat();
        inputFrame.copyTo(handImage, CrMat); //手的图像

        List<MatOfPoint> contours = new ArrayList<>(); // 存储轮廓结果
        Mat hierarchy = new Mat(); //存储轮廓的拓扑结构
        Imgproc.findContours(CrMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        Log.i(TAG, "TipDiscovery 手的图片轮廓数量 = " + contours.size());
        if(contours.size() > 5) { //轮廓效果不理想or画面中无手，直接返回(0,0)
            Log.i(TAG, "TipDiscovery: 轮廓数量太多 or 画面中无手!");
            return finalTip;
        }

        // moments函数找重心
        Moments moment = Imgproc.moments(CrMat, true);
        Point center = new Point(moment.m10/moment.m00, moment.m01/moment.m00);
        Imgproc.circle(handImage, center, 5, new Scalar(0, 0, 255), -1);
        // 找指尖
        Point[] contour = contours.get(findMaxContour(contours)).toArray(); // 找面积最大的轮廓，即手掌所在的轮廓
        if (contour.length < 100) {
            Log.i(TAG, "TipDiscovery: 手的轮廓点数小于100!");
            return finalTip;
        }
        ArrayList<Point> contourFix = new ArrayList<>(Arrays.asList(contour));
        contourFix.addAll(Arrays.asList(contour).subList(0, 100)); // contourFix后面再添加前面的100个点，方便循环检测一遍
        Point pre, current, next;
        // 检测每个点与其前后50个像素点的夹角
        List<Point> candidateTips = new ArrayList<>();
        for(int i = 50; i < contourFix.size()-50; i++) {
            pre = contourFix.get(i-50);
            current = contourFix.get(i);
            next = contourFix.get(i+50);
            int theta = calcTheta(contourFix, i, 50); //点乘和模得夹角
            // 条件一：点乘逆运算得夹角的阈值
            if (theta > 60 && theta < 130) {
                // 条件二：cross叉乘用来当作指尖or两指之间的凹槽的判据
                int cross = (int)((pre.x - current.x ) * (next.y - current.y) - (next.x - current.x ) * (pre.y - current.y));
                // 条件三：一般符合夹角阈值的是一个邻近区域，我们只选择第一个
                Point lastElem;
                double dis2lastElem = 0;
                if(candidateTips.size() > 0) {
                    lastElem = candidateTips.get(candidateTips.size() - 1);
                    dis2lastElem = distanceBetween(current, lastElem);
                }
                // 条件四：候选的指尖点不大可能太靠近左右边缘，滤除这部分点
                if (cross > 0 && (dis2lastElem == 0 ||dis2lastElem > 100) && // 分辨率为1920*1080时dis2lastElem>300, 1280*720时>150
                        (current.x > frameWidth/20 && frameWidth-current.x > frameWidth/20) &&
                        (current.y > frameHeight/20 && frameHeight-current.y > frameHeight/20) ) {
                    candidateTips.add(contourFix.get(i+15)); // 为了指尖更准确
                    if(candidateTips.size() == 1)
                        Log.i(TAG, "TipDiscovery: 候选指尖坐标：");
                    Log.i(TAG, "TipDiscovery: " + current);
                    Imgproc.circle(handImage, current, 5 ,new Scalar(255, 0, 0), -1);
                    Imgproc.line(handImage, center, current, new Scalar(255, 0, 0), 2);
                }
            }
        }
        // 条件五：指尖离重心最远的特征，求出final fingerTip
        double maxDis2Center = 0;
        for(Point p : candidateTips) {
            double dis2Center = distanceBetween(p, center);
            if(dis2Center > maxDis2Center) {
                finalTip = p;
                maxDis2Center = dis2Center;
            }
        }
        Log.i(TAG, "TipDiscovery 最后确定的指尖坐标：" + finalTip);
        Imgproc.circle(handImage, finalTip, 5 ,new Scalar(0, 255, 0), -1);
        saveFrame("frame-"+frameCounter+"-tip", handImage);
        // MainActivity中的onCameraFrame方法中检测到为2，利用handler发送消息，Toast显示检测到指尖
        //MainActivity.TIPDET_SUCCESS = 2;
        return finalTip;
    }

    /** 以下是 Keystroke Detection and Localization 部分 */

    // 检测某一帧画面的按键动作
    public void KeyStrokeDetection() {
        Log.i(TAG, "KeyStrokeDetection: 检测第 " + frameCounter + " 帧");
        Mat inputHandFrame = HandSegmentation(inputFrame);
        Point fingerTip = TipDetection(inputHandFrame);

        //float stopProcessingTime = System.nanoTime()/1000000;
        //float[] time = {stopProcessingTime - startProcessingTime};
        //Log.i(TAG, "start ProcessingTime: " + frameCounter + "  " + startProcessingTime);
        //Log.i(TAG, "stop ProcessingTime: " + frameCounter + "  " + stopProcessingTime + "finalKey: " + key);
        //new Thread(new SensorDataSaver("KeyStrokeDelay", frameCounter, time)).start();

        boolean currentStatus;
        if(frameCounter == 0) {// 记录第一帧的指尖，用于之后的指尖距离计算
            MainActivity.lastFingerTip = fingerTip;
        } else {
            double disTwoTips = distanceBetween(fingerTip, lastFingerTip); //计算两指尖距离
            Log.i(TAG, "KeyStrokeDetection: 与前一帧指尖的距离为" + disTwoTips);
            if(disTwoTips < 15 && (fingerTip.x != 0 && fingerTip.y != 0) ) {
                //两指尖的距离很小表示staying，指尖坐标非(0,0)即画面有指尖
                currentStatus = true;
                double distanceBetweenMoments = distanceBetween(preKeystrokeFingerTip, fingerTip);
                Log.i(TAG, "KeyStrokeDetection: 重复检测两指尖之间的距离为" + distanceBetweenMoments);
                if( !lastStatus && inKeyboardArea("KeyStrokeDetection", fingerTip)) {
                    String key = KeyLocation(fingerTip);
                    if(distanceBetweenMoments > 15) {
                        // 计算KeyStroke的结束时间
                        float stopProcessingTime = System.nanoTime()/1000000;
                        float[] time = {stopProcessingTime - startProcessingTime};
                        //Log.i(TAG, "start ProcessingTime: " + frameCounter + "  " + startProcessingTime);
                        //Log.i(TAG, "stop ProcessingTime: " + frameCounter + "  " + stopProcessingTime + "finalKey: " + key);
                        //new Thread(new SensorDataSaver("ZKeyStrokeDelay", frameCounter, time)).start();

                        Log.i(TAG, "---KeyStrokeDetection!!!---帧数: " + frameCounter + "  final key: " + key);
                        // MainActivity中的onCameraFrame方法中检测到为3，利用handler发送消息，Toast显示检测定位到的按键
                        MainActivity.KEYSTROKE_SUCCESS = 3;
                        MainActivity.KEYDET_RESULT = key;
                        // 若在lastFingerTip处实时更新，距离是每两帧之间的，就失去了距离比较的意义
                        preKeystrokeFingerTip = fingerTip;
                        Log.i(TAG, "KeyStrokeDetection: 变换之前的lastFingerTip = " + preKeystrokeFingerTip);
                    }
                }
            } else {
                currentStatus = false;
            }
            lastFingerTip = fingerTip;
            lastStatus = currentStatus;
        }
    }

    // 已检测到按键动作，定位按键
    private String KeyLocation(Point tip) {
        double minDis = Double.POSITIVE_INFINITY;
        String minDisKey = "NullKey";
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
        return Math.sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
    }

}
