package com.example.mobileai;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Camera;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Main2Activity extends AppCompatActivity {

    //Load the tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/squeezenet.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;
    private static final int requestCode = 1;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224,224,3};
    String mCurrentPhotoPath;
    static final int CAPTURE_IMAGE_REQUEST = 1;
    int REQUEST_PERMISSION = 200;

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;
    File file;
    File photoFile = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
//        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);



        //initialize tensorflow with the AssetManager and the Model
        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);

        imageView = (ImageView) findViewById(R.id.imageview);
        resultView = (TextView) findViewById(R.id.results);
        progressBar = Snackbar.make(imageView,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_PERMISSION);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA},
                    REQUEST_PERMISSION);
        }



        final FloatingActionButton predict = (FloatingActionButton) findViewById(R.id.fab);
        predict.setOnClickListener(new View.OnClickListener() {


            @Override
            public void onClick(View view) {
                captureImage();
               // Intent photoCapIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
               // startActivityForResult(photoCapIntent,requestCode);

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {

                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                try {

                imageView.setImageBitmap(bitmap);

                progressBar.show();

                predict(bitmap);
            }

            catch (Exception e) {
                e.printStackTrace();
            }

        }
    }


            private void captureImage() {

        //int requestCode = 100;

                if (ContextCompat.checkSelfPermission(Main2Activity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(Main2Activity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
                } else {
                    Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                        // Create the File where the photo should go
                        try {

                            photoFile = createImageFile();
                            //displayMessage(getBaseContext(), photoFile.getAbsolutePath());
                            //Log.i("path_check", photoFile.getAbsolutePath());

                            // Continue only if the File was successfully created
                            if (photoFile != null) {
                                Uri photoURI = FileProvider.getUriForFile(Main2Activity.this,
                                        "com.example.mobileai.fileprovider",
                                        photoFile);
                                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                                startActivityForResult(takePictureIntent, requestCode);
                            }
                        } catch (Exception ex) {
                            // Error occurred while creating the File
                            //displayMessage(getBaseContext(), ex.getMessage().toString());
                        }


                    } else {
                       // displayMessage(getBaseContext(), "Null");
                    }
                }


            }

            private File createImageFile() throws IOException {
                // Create an image file name
                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                String imageFileName = "JPEG_" + timeStamp + "_";
                File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                File image = File.createTempFile(
                        imageFileName,  /* prefix */
                        ".jpg",         /* suffix */
                        storageDir      /* directory */
                );

              //   Save a file: path for use with ACTION_VIEW intents
                mCurrentPhotoPath = image.getAbsolutePath();
                return image;
            }

            private void displayMessage(Context context, String message) {
                //Toast.makeText(context, message, Toast.LENGTH_LONG).show();
            }





            //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
            public Object[] argmax(float[] array) {
                int best = -1;
                float best_confidence = 0.0f;

                for (int i = 0; i < array.length; i++) {

                    float value = array[i];

                    if (value > best_confidence) {

                        best_confidence = value;
                        best = i;
                    }
                }
                return new Object[]{best, best_confidence};
            }

            public void predict(final Bitmap bitmap) {


                //Runs inference in background thread
                new AsyncTask<Integer, Integer, Integer>() {

                    @Override

                    protected Integer doInBackground(Integer... params) {

                        //Resize the image into 224 x 224
                        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 224);

                        //Normalize the pixels
                        floatValues = ImageUtils.normalizeBitmap(resized_image, 224, 127.5f, 1.0f);

                        //Pass input into the tensorflow
                        tf.feed(INPUT_NAME, floatValues, 1, 224, 224, 3);

                        //compute predictions
                        tf.run(new String[]{OUTPUT_NAME});

                        //copy the output into the PREDICTIONS array
                        tf.fetch(OUTPUT_NAME, PREDICTIONS);

                        //Obtained highest prediction
                        Object[] results = argmax(PREDICTIONS);


                        int class_index = (Integer) results[0];
                        float confidence = (Float) results[1];


                        try {

                            final String conf = String.valueOf(confidence * 100).substring(0, 5);

                            //Convert predicted class index into actual label name
                            final String label = ImageUtils.getLabel(getAssets().open("labels.json"), class_index);


                            //Display result on UI
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {

                                    progressBar.dismiss();
                                    resultView.setText(label + " : " + conf + "%");

                                }
                            });

                        } catch (Exception e) {


                        }


                        return 0;
                    }


                }.execute(0);

            }
        }
