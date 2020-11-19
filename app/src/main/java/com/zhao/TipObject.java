package com.zhao;

import org.opencv.core.Point;

public class TipObject {

    // 指尖坐标
    private Point tip;

    // 重心到指尖的距离
    private int distance;

    // 手的重心
    private Point center;

    public TipObject(Point tip, int dis, Point center) {
        this.tip = tip;
        this.distance = dis;
        this.center = center;
    }

    public int getDistance() {
        return distance;
    }

    public Point getTip() {
        return tip;
    }

    public Point getCenter() {
        return center;
    }

    public void setDistance(int distance) {
        this.distance = distance;
    }

    public void setTip(Point tip) {
        this.tip = tip;
    }

    public void setCenter(Point center) {
        this.center = center;
    }
}
