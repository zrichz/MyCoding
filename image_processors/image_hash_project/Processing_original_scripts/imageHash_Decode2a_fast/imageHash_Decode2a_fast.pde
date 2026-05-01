/*
Aug 2023. image hash
 //  ------------------
 // |        |         |
 // |  enc   |   dec   |
 // |        |         |
 //  ------------------
 Decodes an encoded image using a (previously generated) serialised list of encoded positions
 */

import java.io.*;
import java.util.*;
ArrayList<Integer> list = null;

PImage myEncodedImage;
PImage myDecodedImage;
int x=0;
int numPixels=512*512; //number of pixels for 512x512 images

void setup() {
  noLoop();
  size(1024, 512, P2D);
  String filepath = sketchPath("") + "data/list.ser";  // specify the file path where you want to READ IN the .ser (serial) file

  // read in the serialised list of encoded positions
  try {
    FileInputStream fileIn = new FileInputStream(filepath);
    ObjectInputStream in = new ObjectInputStream(fileIn);
    list = (ArrayList<Integer>) in.readObject();
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

  // Now you can use the ArrayList (called 'list')

  myEncodedImage = loadImage("encoded_AI2.png");
  myDecodedImage = loadImage("encoded_AI2.png");
}

void draw() {
  image(myEncodedImage, 0, 0); // draw encoded image on LHS of screen

  myEncodedImage.loadPixels();
  for (int i=0; i<numPixels; i++) { // test version
    myDecodedImage.pixels[i]=myEncodedImage.pixels[list.get(i)];
  }
  updatePixels();

  image(myDecodedImage, 512, 0); // draw decoded image on RHS of screen

  // save RHS, DECODED image...
  PImage enc_img = get(512, 0, 512, 512); // Capture a 512x512 pixel area with top left at (512,0) - i.e. the encoded image
  enc_img.save("data/DECODED2a.png"); // Save the captured area as an image
  println("\nSaved DECODED image");
}
