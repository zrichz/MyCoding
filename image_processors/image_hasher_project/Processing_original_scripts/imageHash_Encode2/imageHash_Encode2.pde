/*
Aug 2023. image encode/decode
 -------------------
 |        |         |
 |  orig  | encoded |
 |        |         |
 -------------------
 takes an image, shuffles pixels (encodes), using a pre-built "list.ser" - a serialised set of 512x512 positions
 A separate Processing sketch is used to decode the images
 */

import java.io.*;   // contains import/export and exceptions
import java.util.*; // contains shuffle
ArrayList<Integer> enc_Pos = null;

PImage img;
int x=0;
int testPixels=512*512; //number of test pixels. will be replaced with actual number of pixels in image in final code

void setup() {
  noLoop();
  size(1024, 512, P2D);

  String filepath = sketchPath("") + "data/list.ser";  // specify the file path where you want to READ IN the .ser (serial) file

  // read in the serialised list of encoded positions
  try {
    FileInputStream fileIn = new FileInputStream(filepath);
    ObjectInputStream in = new ObjectInputStream(fileIn);
    enc_Pos = (ArrayList<Integer>) in.readObject();
    in.close();
    fileIn.close();
  }
  catch (IOException i) {
    i.printStackTrace();
  }
  catch (ClassNotFoundException c) {
    System.out.println("Class not found");
    c.printStackTrace();
  }

  // Now you can use the ArrayList (called 'enc_Pos')

  // Load image to ENCODE...
  img = loadImage("in4.jpg");
}

void draw() {
  image(img, 0, 0);
  println("pic loaded");
  int w=img.width;

  // encode...
  print("\nEncoding...");
  //for (int pixel=0; pixel<img1.width*img1.height; pixel++) { //final version
  for (int px=0; px<testPixels; px++) { // test version
    int ox=px%w;              // original x pos
    int oy=px/w;              // original y pos
    int nx=enc_Pos.get(px)%w; // encoded x pos
    int ny=enc_Pos.get(px)/w; // encoded y pos
    copy(ox, oy, 1, 1, w+nx, ny, 1, 1); // note: offset to RHS of original image using img1.width
    if (px%(testPixels/10)==0) print(int(100*px/testPixels)+"%.."); // show progress every 10% of pixels
  }

  // save RHS, encoded image...
  PImage enc_img = get(512, 0, 512, 512); // Capture a 512x512 pixel area with top left at (512,0) - i.e. the encoded image
  enc_img.save("data/encoded_in4.png"); // Save the captured area as an image
  println("\nsaved encoded image");
}
