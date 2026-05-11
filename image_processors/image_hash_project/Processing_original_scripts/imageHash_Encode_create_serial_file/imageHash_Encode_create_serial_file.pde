/*
Aug 2023. image hash
 -------------------
 |        |         |
 |  orig  | encoded |
 |        |         |
 -------------------
 takes an image, shuffles pixels (encodes)
 */

import java.io.*;   // contains import/export and exceptions
import java.util.*; // contains shuffle
ArrayList<Integer> truePos = new ArrayList<Integer>();
ArrayList<Integer> enc_Pos = new ArrayList<Integer>();
Random myRnd = new Random(10); // Seed for deterministic shuffle

PImage img;
int x=0;
int testPixels=512*512; //number of test pixels. will be replaced with actual number of pixels in image in final code

void setup() {
  noLoop();
  size(1024, 512, P2D);
  //img = loadImage("AI_512x512.png");
  img = loadImage("dali.jpg");

  for (int i = 0; i < (512*512); i++) {      // Fill the "originalNumbers" array list with 0 to 999,999
    truePos.add(i);
  }
  enc_Pos = new ArrayList<Integer>(truePos); // Copy original list to shuffledNumbers
  Collections.shuffle(enc_Pos, myRnd);       // Shuffle the list
}

void draw() {
  image(img, 0, 0);
  println("pic loaded");
  int w=img.width;

  // encode...
  print("\nencoding...");
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
  enc_img.save("encoded.png"); // Save the captured area as an image
  println("\nsaved encoded image");
 
  String filepath = sketchPath("") + "data/list.ser";  // specify the file path where you want to save the serial file 
  try {
    FileOutputStream fileOut = new FileOutputStream(filepath);
    ObjectOutputStream out = new ObjectOutputStream(fileOut);
    out.writeObject(enc_Pos);
    out.close();
    fileOut.close();
  }
  catch (IOException i) {
    i.printStackTrace();
  }
  println("\nsaved file as : "+ filepath);

}
