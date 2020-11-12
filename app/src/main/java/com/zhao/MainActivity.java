package com.zhao;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    // 应用权限请求
    private static final int REQUEST_PERMISSION = 200;

    // 传感器数据收集
    private SensorActivity sensorActivity;

    // 布局文件中的 JavaCameraView
    private MyJavaCameraView mCvCameraView;

    // 菜单栏相关参数
    public static final int MODE_STOP = 0;
    public static final int MODE_START = 1;
    public static int viewMode = MODE_STOP;
    private MenuItem mMenuItem;

    // 帧的计数
    private static int frameCounter = 0;
    // 判断是否是开始处理的第一帧，以用于KeyExtraction
    public static boolean isFirstFrame = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 设置全屏
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        // 设置常亮
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // 运行时检查权限
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            // 对应添加在 AndroidManifest
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS
            }, REQUEST_PERMISSION);
            return;
        }

        // 传感器管理器
        SensorManager sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        sensorActivity = new SensorActivity(sensorManager);
        mCvCameraView = findViewById(R.id.camera_view);
        mCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    protected void onResume() {
        super.onResume();
        // 每次当前Activity被激活时都会调用此方法，在此处检测OpenCV库文件是否加载完毕
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        // 设置分辨率 800*480
        mCvCameraView.setAbsolution(800, 480);
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorActivity.unregisterListener();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        sensorActivity.unregisterListener();
    }

    // 通过OpenCV管理Android服务，异步初始化OpenCV
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                // 只有当mLoaderCallback收到SUCCESS消息时才会预览显示，消息是从onResume中发出的
                case LoaderCallbackInterface.SUCCESS:
                    Log.d(TAG, "OpenCV loader successfully!");
                    mCvCameraView.enableView(); // 预览显示
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    // 处理请求权限的响应
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        if (requestCode == REQUEST_PERMISSION) {
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "You can't use camera without permission",
                        Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    // 设置菜单栏
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.d(TAG, "called onCreateOptionsMenu");
        mMenuItem = menu.add("Capture"); // 菜单栏里添加一个菜单项
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.d(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mMenuItem)
            viewMode = 1- viewMode;
        if (viewMode == 1) {
            sensorActivity.registerListener();
            Toast.makeText(MainActivity.this, "Start Working!", Toast.LENGTH_SHORT).show();
        } else {
            sensorActivity.unregisterListener();
            Toast.makeText(MainActivity.this, "Stop!", Toast.LENGTH_SHORT).show();
        }
        return true;
    }

    // 实现 CvCameraViewListener2 接口的三个方法
    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // 图像处理都写在此处，函数在相机刷新每一帧都会调用一次
        final Mat frame = inputFrame.rgba();
        int frameWidth = frame.width();
        int frameHeight = frame.height();
        int frameChannel = frame.channels();
        Log.d(TAG, "onCameraFrame: CameraFrame = "+ frameWidth + "*" + frameHeight + ", " + frameChannel);

        Bitmap bitmap = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.RGB_565);
        Utils.matToBitmap(frame, bitmap); // OpenCV Mat2Bitmap
        BitmapApplication myApp = (BitmapApplication) getApplicationContext();
        myApp.setOneBitmap(bitmap); // 存入Bitmap格式的帧画面，以备service中取出分析

        // 系统开始处理帧画面
        if (viewMode == MODE_START) {
            // 保存每帧画面在本地
            new Thread(new FrameSaver("frame-"+frameCounter, bitmap)).start();

            Intent keyboardIntent = KeystrokeService.newIntent(MainActivity.this);
            // 帧计数传入Service，帧通过MyApplication传入
            keyboardIntent.putExtra("frameCounter", frameCounter);
            frameCounter++;
            // 进入服务(记得在AndroidManifest.xml中注册服务)
            startService(keyboardIntent);

        }
        return frame;
    }


}