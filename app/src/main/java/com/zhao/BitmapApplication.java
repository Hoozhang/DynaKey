package com.zhao;

import android.app.Application;
import android.graphics.Bitmap;

/**
 * 不同的Activity之间传递图片
 * 法一：设置static变量，供全局set和get
 * 法二：基于Android Context，自定义MyApplication类, 在manifest里指定这个类
 * https://blog.csdn.net/kavensu/article/details/8262187
 */

public class BitmapApplication extends Application {
    private static final String TAG = "MyApplication";

    private Bitmap bitmap;

    public Bitmap getOneBitmap() {
        return bitmap;
    }

    public void setOneBitmap(Bitmap frame) {
        bitmap = frame.copy(Bitmap.Config.RGB_565, true);
    }

}
