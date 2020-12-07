package com.zhao;

/**
 * Created by HaoZ on 2018/3/20.
 */
public class MovingAverage {
    private static final String TAG = "MovingAverage";

    private final float[] circularBuffer; // 存放k个数值的buffer数组
    private float avg; // 计算的平均值
    private int bufferIndex; // buffer数组的index下标
    private int count; // 已经应用Moving Average的value数目

    public MovingAverage(int k) {
        circularBuffer = new float[k];
        avg = 0;
        bufferIndex = 0;
        count = 0;
    }

    // get current moving average
    public float getValue() {
        return avg;
    }

    // 将x加入到buffer数组
    public void pushValue(float x) {
        if(count++ == 0) { // 先判断count == 0，再执行count++
            primeBuffer(x);
        }
        float lastValue = circularBuffer[bufferIndex];
        // 无需重新求和平均，只需加上avg的增加量，即(x-lastValue)/circularBuffer.length
        avg = avg + (x - lastValue)/circularBuffer.length;
        circularBuffer[bufferIndex] = x;
        bufferIndex = getNextIndex(bufferIndex);
    }

    public long getCount() {
        return count;
    }

    // 起初状态下，用第一个元素填充数组
    private void primeBuffer(float val) {
        for(int i = 0; i < circularBuffer.length; i++) {
            circularBuffer[i] = val;
        }
        avg = val;
    }

    // 获取下一个下标，分为数组未满和数组已满两种情况
    private int getNextIndex(int currentIndex) {
        if(currentIndex + 1 >= circularBuffer.length) {
            return 0;
        }
        return currentIndex + 1;
    }

}
