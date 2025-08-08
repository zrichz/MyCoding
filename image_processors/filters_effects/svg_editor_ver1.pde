// Import Processing SVG library
import processing.svg.*;

// Global variables
ArrayList<ArrayList<PVector>> outlines;
String svgPath = "gimp_exported_path.svg";
float scale = 1.0;

void setup() {
  size(800, 800);
  background(255);
  stroke(0);
  noFill();
  
  // Load SVG file
  PShape svg = loadShape(svgPath);
  outlines = extractOutlines(svg);
  
  // Print outline info
  println("Extracted Outlines:");
  for (int i = 0; i < outlines.size(); i++) {
    println("\nOutline " + (i+1) + ":");
    ArrayList<PVector> outline = outlines.get(i);
    for (PVector p : outline) {
      println("x: " + nf(p.x, 0, 2) + ", y: " + nf(p.y, 0, 2));
    }
  }
}

void draw() {
  background(255);
  translate(width/2, height/2);
  
  // Draw all outlines
  for (ArrayList<PVector> outline : outlines) {
    beginShape();
    for (PVector p : outline) {
      vertex(p.x * scale, p.y * scale);
    }
    endShape(CLOSE);
  }
}

ArrayList<ArrayList<PVector>> extractOutlines(PShape svg) {
  ArrayList<ArrayList<PVector>> result = new ArrayList<>();
  
  // Get vertices from SVG shape
  for (int i = 0; i < svg.getChildCount(); i++) {
    PShape child = svg.getChild(i);
    ArrayList<PVector> points = new ArrayList<>();
    
    for (int j = 0; j < child.getVertexCount(); j++) {
      PVector v = child.getVertex(j);
      points.add(v);
    }
    
    if (points.size() > 0) {
      result.add(points);
    }
  }
  
  return result;
}

void keyPressed() {
  if (key == '+') scale *= 1.1;
  if (key == '-') scale *= 0.9;
}