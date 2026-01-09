/**
 SDXL_Combined_Prompter.pde
 Processing sketch to autogenerate complete SDXL prompts with Primary and Secondary stages.
 - Combines Primary (9 stages) and Secondary (7 categories)
 - Outputs in format: <primary> | <secondary>
 - Click Cycle to step options. Use Generate N to create multiple randomized prompts.
 - Save writes generated prompts to a text file.
 
 Author: Copilot
 Date: 2026-01-09
*/

import java.util.ArrayList;

// PRIMARY STAGES (9)
final int PRIMARY_COUNT = 9;
ArrayList<String>[] primaryStages = (ArrayList<String>[]) new ArrayList[PRIMARY_COUNT];
String[] primaryNames = {
  "Subject identity",
  "Pose and action",
  "Framing and crop",
  "Clothing and key props",
  "Expression and gaze",
  "Body descriptors",
  "Composition anchors",
  "Context or location",
  "Semantic technical anchors"
};
int[] primaryIndices = new int[PRIMARY_COUNT];

// SECONDARY CATEGORIES (7)
final int SECONDARY_COUNT = 7;
ArrayList<String>[] secondaryCategories = (ArrayList<String>[]) new ArrayList[SECONDARY_COUNT];
String[] secondaryNames = {
  "Lighting",
  "Camera / Perspective",
  "Lens / Focal length",
  "Color grading / Film style",
  "Depth of field / Bokeh",
  "Texture / Finish",
  "Mood / Subtle effects"
};
int[] secondaryIndices = new int[SECONDARY_COUNT];

ArrayList<String> generated = new ArrayList<String>();

// UI layout
int leftMargin = 20;
int topMargin = 20;
int boxHeight = 40;
int boxWidth = 680;
int middleGap = 20;
int secondaryStartY;
int rightPanelX;
int buttonW = 120;
int buttonH = 28;
int generateCount = 5;

void setup() {
  size(1480, 900);
  secondaryStartY = topMargin + 60 + PRIMARY_COUNT * boxHeight + 40;
  rightPanelX = leftMargin + boxWidth + middleGap;
  textFont(createFont("Arial", 13));
  initPrimaryStages();
  initSecondaryCategories();
}

void draw() {
  background(245);
  
  // Title
  fill(30);
  textSize(20);
  text("SDXL Combined Prompt Generator", leftMargin, topMargin - 2);
  textSize(11);
  text("Click Cycle to step options. Use Generate N to create randomized <primary> | <secondary> prompts.", 
       leftMargin, topMargin + 22);
  
  // PRIMARY section header
  textSize(14);
  fill(60, 80, 160);
  text("PRIMARY PROMPT STAGES", leftMargin, topMargin + 50);
  
  // Draw primary stage boxes
  for (int i = 0; i < PRIMARY_COUNT; i++) {
    int y = topMargin + 60 + i * boxHeight;
    drawStageBox(i, leftMargin, y, boxWidth, boxHeight - 4, true);
  }
  
  // SECONDARY section header
  textSize(14);
  fill(160, 80, 60);
  int secHeaderY = secondaryStartY - 20;
  text("SECONDARY STYLE CATEGORIES", leftMargin, secHeaderY);
  
  // Draw secondary category boxes
  for (int i = 0; i < SECONDARY_COUNT; i++) {
    int y = secondaryStartY + i * boxHeight;
    drawStageBox(i, leftMargin, y, boxWidth, boxHeight - 4, false);
  }
  
  // Right panel: controls and generated list
  drawRightPanel();
}

void drawStageBox(int idx, int x, int y, int w, int h, boolean isPrimary) {
  stroke(200);
  fill(255);
  rect(x, y, w, h, 6);
  
  String[] names = isPrimary ? primaryNames : secondaryNames;
  int[] indices = isPrimary ? primaryIndices : secondaryIndices;
  ArrayList<String>[] stages = isPrimary ? primaryStages : secondaryCategories;
  
  fill(0);
  textSize(11);
  text(names[idx], x + 8, y + 14);
  textSize(10);
  String current = stages[idx].get(indices[idx]);
  text(current, x + 8, y + 28, w - 140);
  
  // Cycle button
  int bx = x + w - buttonW - 12;
  int by = y + 6;
  fill(220);
  rect(bx, by, buttonW, buttonH, 6);
  fill(0);
  textAlign(CENTER, CENTER);
  textSize(11);
  text("Cycle", bx + buttonW/2, by + buttonH/2);
  textAlign(LEFT);
}

void drawRightPanel() {
  int x = rightPanelX;
  int y = topMargin + 60;
  int w = width - x - 20;
  int h = height - y - 20;
  stroke(200);
  fill(255);
  rect(x, y, w, h, 6);
  
  fill(0);
  textSize(13);
  text("Controls", x + 12, y + 18);
  
  // Generate count
  textSize(11);
  text("Generate count (N): " + generateCount, x + 12, y + 40);
  
  // Buttons
  int bx = x + 12;
  int by = y + 52;
  drawButton(bx, by, "Generate N", buttonW, buttonH);
  drawButton(bx + buttonW + 12, by, "Save", buttonW, buttonH);
  drawButton(bx + 2*(buttonW + 12), by, "Clear", buttonW, buttonH);
  
  // Preview combined prompt
  textSize(12);
  text("Preview Combined Prompt", x + 12, by + buttonH + 28);
  String preview = assembleCombinedPrompt(false);
  textSize(10);
  textLeading(13);
  text(preview, x + 12, by + buttonH + 44, w - 24, 180);
  
  // Generated prompts list
  textSize(12);
  text("Generated Combined Prompts", x + 12, by + buttonH + 240);
  textSize(9);
  textLeading(12);
  int listY = by + buttonH + 260;
  int displayCount = min(generated.size(), 25);
  for (int i = 0; i < displayCount; i++) {
    String shortened = generated.get(i);
    if (shortened.length() > 120) shortened = shortened.substring(0, 117) + "...";
    text((i+1) + ". " + shortened, x + 12, listY + i * 14, w - 24, 14);
  }
  if (generated.size() > 25) {
    text("... (" + (generated.size() - 25) + " more)", x + 12, listY + 25 * 14);
  }
  
  // Instructions
  textSize(9);
  fill(80);
  text("Keyboard: [+/-] change count | [G] generate | [C] clear", x + 12, height - 28);
}

void drawButton(int x, int y, String label, int w, int h) {
  fill(220);
  rect(x, y, w, h, 6);
  fill(0);
  textAlign(CENTER, CENTER);
  textSize(11);
  text(label, x + w/2, y + h/2);
  textAlign(LEFT);
}

void mousePressed() {
  // Check primary cycle buttons
  for (int i = 0; i < PRIMARY_COUNT; i++) {
    int y = topMargin + 60 + i * boxHeight;
    int bx = leftMargin + boxWidth - buttonW - 12;
    int by = y + 6;
    if (mouseX >= bx && mouseX <= bx + buttonW && mouseY >= by && mouseY <= by + buttonH) {
      primaryIndices[i] = (primaryIndices[i] + 1) % primaryStages[i].size();
      return;
    }
  }
  
  // Check secondary cycle buttons
  for (int i = 0; i < SECONDARY_COUNT; i++) {
    int y = secondaryStartY + i * boxHeight;
    int bx = leftMargin + boxWidth - buttonW - 12;
    int by = y + 6;
    if (mouseX >= bx && mouseX <= bx + buttonW && mouseY >= by && mouseY <= by + buttonH) {
      secondaryIndices[i] = (secondaryIndices[i] + 1) % secondaryCategories[i].size();
      return;
    }
  }
  
  // Right panel buttons
  int x = rightPanelX + 12;
  int by = topMargin + 60 + 52;
  if (mouseX >= x && mouseX <= x + buttonW && mouseY >= by && mouseY <= by + buttonH) {
    generatePrompts(generateCount);
    return;
  }
  if (mouseX >= x + buttonW + 12 && mouseX <= x + 2*buttonW + 12 && mouseY >= by && mouseY <= by + buttonH) {
    saveGeneratedPrompts();
    return;
  }
  if (mouseX >= x + 2*(buttonW + 12) && mouseX <= x + 3*buttonW + 24 && mouseY >= by && mouseY <= by + buttonH) {
    generated.clear();
    return;
  }
}

String assemblePrimaryPrompt() {
  StringBuilder sb = new StringBuilder();
  for (int i = 0; i < PRIMARY_COUNT; i++) {
    sb.append(primaryStages[i].get(primaryIndices[i]));
    if (i < PRIMARY_COUNT - 1) sb.append("; ");
  }
  return sb.toString();
}

String assembleSecondaryPrompt() {
  StringBuilder sb = new StringBuilder();
  for (int i = 0; i < SECONDARY_COUNT; i++) {
    sb.append(secondaryCategories[i].get(secondaryIndices[i]));
    if (i < SECONDARY_COUNT - 1) sb.append(", ");
  }
  return sb.toString();
}

String assembleCombinedPrompt(boolean includeLabels) {
  StringBuilder sb = new StringBuilder();
  if (includeLabels) {
    sb.append("Primary: ");
    sb.append(assemblePrimaryPrompt());
    sb.append("\n\nSecondary: ");
    sb.append(assembleSecondaryPrompt());
  } else {
    sb.append(assemblePrimaryPrompt());
    sb.append(" | ");
    sb.append(assembleSecondaryPrompt());
  }
  return sb.toString();
}

void generatePrompts(int n) {
  for (int k = 0; k < n; k++) {
    // Randomly select primary options
    StringBuilder primary = new StringBuilder();
    for (int i = 0; i < PRIMARY_COUNT; i++) {
      int r = (int) random(primaryStages[i].size());
      primary.append(primaryStages[i].get(r));
      if (i < PRIMARY_COUNT - 1) primary.append("; ");
    }
    
    // Randomly select secondary options
    StringBuilder secondary = new StringBuilder();
    for (int i = 0; i < SECONDARY_COUNT; i++) {
      int r = (int) random(secondaryCategories[i].size());
      secondary.append(secondaryCategories[i].get(r));
      if (i < SECONDARY_COUNT - 1) secondary.append(", ");
    }
    
    // Combine: <primary> | <secondary>
    String combined = primary.toString() + " | " + secondary.toString();
    generated.add(combined);
  }
}

void saveGeneratedPrompts() {
  if (generated.size() == 0) return;
  String[] out = new String[generated.size()];
  for (int i = 0; i < generated.size(); i++) out[i] = generated.get(i);
  saveStrings("generated_combined_prompts.txt", out);
  println("Saved " + generated.size() + " prompts to generated_combined_prompts.txt");
}

void keyPressed() {
  if (key == '+') generateCount++;
  if (key == '-' && generateCount > 1) generateCount--;
  if (key == 'g' || key == 'G') generatePrompts(generateCount);
  if (key == 'c' || key == 'C') generated.clear();
  if (key == 's' || key == 'S') saveGeneratedPrompts();
}

// Initialize all Primary stages (copied from primary prompter)
void initPrimaryStages() {
  // Subject identity (40 options - abbreviated for space)
  primaryStages[0] = new ArrayList<String>();
  primaryStages[0].add("female, early 30s, freckled skin");
  primaryStages[0].add("male, late 20s, light stubble");
  primaryStages[0].add("nonbinary, mid 20s, short hair");
  primaryStages[0].add("female, mid 40s, glasses");
  primaryStages[0].add("male, early 40s, salt-and-pepper beard");
  primaryStages[0].add("female, late 20s, long curly hair");
  primaryStages[0].add("male, mid 30s, shaved head");
  primaryStages[0].add("female, early 20s, natural makeup");
  primaryStages[0].add("male, late 30s, athletic build");
  primaryStages[0].add("female, mid 50s, silver hair");
  primaryStages[0].add("male, early 20s, gap-toothed smile");
  primaryStages[0].add("female, late 30s, freckles and glasses");
  primaryStages[0].add("male, mid 50s, lined face");
  primaryStages[0].add("female, early 40s, short bob");
  primaryStages[0].add("male, late 40s, receding hairline");
  primaryStages[0].add("female, mid 20s, pierced ear");
  primaryStages[0].add("male, early 30s, tattooed forearm");
  primaryStages[0].add("female, late 40s, warm complexion");
  primaryStages[0].add("male, mid 20s, tousled hair");
  primaryStages[0].add("female, early 50s, soft features");
  primaryStages[0].add("male, late 20s, glasses and beard");
  primaryStages[0].add("female, mid 30s, athletic build");
  primaryStages[0].add("male, early 60s, weathered hands");
  primaryStages[0].add("female, late 20s, short cropped hair");
  primaryStages[0].add("male, mid 40s, wearing a cap");
  primaryStages[0].add("female, early 30s, light tan");
  primaryStages[0].add("male, late 30s, warm smile");
  primaryStages[0].add("female, mid 20s, natural freckles");
  primaryStages[0].add("male, early 50s, salt-and-pepper stubble");
  primaryStages[0].add("female, late 30s, long straight hair");
  primaryStages[0].add("male, mid 20s, slim build");
  primaryStages[0].add("female, early 40s, wearing glasses");
  primaryStages[0].add("male, late 50s, lined smile");
  primaryStages[0].add("female, mid 30s, visible birthmark");
  primaryStages[0].add("male, early 40s, casual beard");
  primaryStages[0].add("female, late 20s, relaxed posture");
  primaryStages[0].add("male, mid 30s, cropped hair and stubble");

  // Pose and action (39 options)
  primaryStages[1] = new ArrayList<String>();
  primaryStages[1].add("three-quarter turn, left arm raised");
  primaryStages[1].add("standing, hands in pockets");
  primaryStages[1].add("seated, one knee up");
  primaryStages[1].add("walking toward camera");
  primaryStages[1].add("sitting on a low wall, legs crossed");
  primaryStages[1].add("leaning on a railing, looking down");
  primaryStages[1].add("holding a mug with both hands");
  primaryStages[1].add("adjusting jacket collar casually");
  primaryStages[1].add("reading a paperback, head tilted");
  primaryStages[1].add("tying shoelace, looking away");
  primaryStages[1].add("brushing hair back with one hand");
  primaryStages[1].add("standing with arms folded loosely");
  primaryStages[1].add("sitting on stairs, elbows on knees");
  primaryStages[1].add("walking dog on a short lead");
  primaryStages[1].add("holding a takeaway coffee, mid-sip");
  primaryStages[1].add("checking phone while standing");
  primaryStages[1].add("leaning against doorframe, relaxed");
  primaryStages[1].add("sitting at table, hands clasped");
  primaryStages[1].add("looking out of a window, thoughtful");
  primaryStages[1].add("standing under an umbrella, slight smile");
  primaryStages[1].add("carrying a backpack, mid-step");
  primaryStages[1].add("sitting on a bench, one arm draped");
  primaryStages[1].add("holding bicycle by the handlebars");
  primaryStages[1].add("walking up a short flight of steps");
  primaryStages[1].add("adjusting glasses, slight tilt of head");
  primaryStages[1].add("reaching for a shelf, casual stance");
  primaryStages[1].add("sitting cross-legged on the floor");
  primaryStages[1].add("leaning over a kitchen counter");
  primaryStages[1].add("standing with one foot on a low step");
  primaryStages[1].add("holding a camera at chest height");
  primaryStages[1].add("sitting in a cafe booth, relaxed");
  primaryStages[1].add("standing by a window, hands in pockets");
  primaryStages[1].add("walking past a shopfront, glancing");
  primaryStages[1].add("sitting on a windowsill, knees drawn up");
  primaryStages[1].add("holding a newspaper, reading");
  primaryStages[1].add("standing with coat draped over shoulder");
  primaryStages[1].add("leaning on a bicycle, casual");
  primaryStages[1].add("sitting on a low wall, feet dangling");
  primaryStages[1].add("mid-laugh, head thrown back slightly");

  // Framing and crop (39 options)
  primaryStages[2] = new ArrayList<String>();
  primaryStages[2].add("waist-up");
  primaryStages[2].add("head and shoulders");
  primaryStages[2].add("full body");
  primaryStages[2].add("three-quarter body");
  primaryStages[2].add("close-up face");
  primaryStages[2].add("knee-up");
  primaryStages[2].add("hip-up");
  primaryStages[2].add("environmental portrait, subject small in frame");
  primaryStages[2].add("tight portrait, eyes centered");
  primaryStages[2].add("half-body, slight tilt");
  primaryStages[2].add("over-the-shoulder crop");
  primaryStages[2].add("waist-up, slight left offset");
  primaryStages[2].add("head-to-toe, centered");
  primaryStages[2].add("upper torso, three-quarter turn");
  primaryStages[2].add("close crop on hands and face");
  primaryStages[2].add("mid-shot with foreground blur");
  primaryStages[2].add("portrait orientation, headroom");
  primaryStages[2].add("landscape orientation, subject left");
  primaryStages[2].add("tight headshot with soft bokeh");
  primaryStages[2].add("full body with negative space above");
  primaryStages[2].add("waist-up, slight downward angle");
  primaryStages[2].add("head and shoulders, eye-level");
  primaryStages[2].add("three-quarter body, slight wide angle");
  primaryStages[2].add("close-up of profile");
  primaryStages[2].add("mid-shot with environmental context");
  primaryStages[2].add("candid half-body crop");
  primaryStages[2].add("full body, slight motion blur");
  primaryStages[2].add("tight portrait, off-center composition");
  primaryStages[2].add("head and shoulders, soft framing");
  primaryStages[2].add("waist-up, natural posture");
  primaryStages[2].add("full body, slight low angle");
  primaryStages[2].add("close-up with hands visible");
  primaryStages[2].add("three-quarter body, centered");
  primaryStages[2].add("mid-shot with foreground element");
  primaryStages[2].add("tight crop on face and shoulders");
  primaryStages[2].add("full body, environmental detail visible");
  primaryStages[2].add("head and shoulders, slight side lighting");
  primaryStages[2].add("waist-up, casual stance");
  primaryStages[2].add("close-up with soft edge vignette");

  // Clothing and key props (39 options)
  primaryStages[3] = new ArrayList<String>();
  primaryStages[3].add("cream linen shirt");
  primaryStages[3].add("black leather jacket");
  primaryStages[3].add("striped sweater");
  primaryStages[3].add("tailored coat and scarf");
  primaryStages[3].add("denim jacket and white tee");
  primaryStages[3].add("wool jumper and jeans");
  primaryStages[3].add("floral dress, simple cut");
  primaryStages[3].add("casual hoodie and trainers");
  primaryStages[3].add("button-up shirt and chinos");
  primaryStages[3].add("raincoat and wellies");
  primaryStages[3].add("t-shirt and cardigan");
  primaryStages[3].add("checked shirt and denim");
  primaryStages[3].add("knit sweater and skirt");
  primaryStages[3].add("polo shirt and jeans");
  primaryStages[3].add("oversized knit and leggings");
  primaryStages[3].add("work jacket and boots");
  primaryStages[3].add("simple blouse and trousers");
  primaryStages[3].add("sweatshirt and joggers");
  primaryStages[3].add("casual blazer and tee");
  primaryStages[3].add("striped tee and denim shorts");
  primaryStages[3].add("light rain jacket and umbrella");
  primaryStages[3].add("wool coat and scarf");
  primaryStages[3].add("checked scarf and beanie");
  primaryStages[3].add("summer dress and sandals");
  primaryStages[3].add("puffer jacket and jeans");
  primaryStages[3].add("work shirt and apron");
  primaryStages[3].add("school uniform style blazer");
  primaryStages[3].add("cycling jacket and helmet");
  primaryStages[3].add("sweater vest and shirt");
  primaryStages[3].add("linen trousers and loafers");
  primaryStages[3].add("casual shirt and backpack");
  primaryStages[3].add("striped jumper and coat");
  primaryStages[3].add("simple tee and denim jacket");
  primaryStages[3].add("fitted coat and scarf");
  primaryStages[3].add("overshirt and chinos");
  primaryStages[3].add("wool hat and gloves");
  primaryStages[3].add("checked shirt and boots");
  primaryStages[3].add("light cardigan and jeans");
  primaryStages[3].add("work boots and utility jacket");

  // Expression and gaze (38 options)
  primaryStages[4] = new ArrayList<String>();
  primaryStages[4].add("soft smile, looking slightly off-camera");
  primaryStages[4].add("neutral expression, direct gaze");
  primaryStages[4].add("candid smile, eyes down");
  primaryStages[4].add("contemplative, looking to the left");
  primaryStages[4].add("gentle laugh, head tilted");
  primaryStages[4].add("focused, looking at hands");
  primaryStages[4].add("relaxed, eyes half-closed");
  primaryStages[4].add("slight grin, looking away");
  primaryStages[4].add("thoughtful, distant gaze");
  primaryStages[4].add("subtle smile, direct eye contact");
  primaryStages[4].add("soft expression, looking up");
  primaryStages[4].add("calm, eyes on horizon");
  primaryStages[4].add("mild amusement, glancing aside");
  primaryStages[4].add("serene, slight smile");
  primaryStages[4].add("pensive, looking downwards");
  primaryStages[4].add("warm smile, eyes crinkled");
  primaryStages[4].add("reserved smile, head slightly bowed");
  primaryStages[4].add("gentle smirk, looking off-frame");
  primaryStages[4].add("open smile, candid expression");
  primaryStages[4].add("softly serious, direct gaze");
  primaryStages[4].add("subdued smile, eyes to camera");
  primaryStages[4].add("mild surprise, eyebrows raised");
  primaryStages[4].add("relaxed grin, looking to side");
  primaryStages[4].add("quiet contentment, eyes closed briefly");
  primaryStages[4].add("slight frown, thoughtful");
  primaryStages[4].add("gentle curiosity, head tilt");
  primaryStages[4].add("soft laugh, looking down");
  primaryStages[4].add("calm, steady gaze");
  primaryStages[4].add("subtle amusement, eyes to left");
  primaryStages[4].add("warm, approachable smile");
  primaryStages[4].add("reflective, distant look");
  primaryStages[4].add("mildly inquisitive, direct gaze");
  primaryStages[4].add("soft grin, slight squint");
  primaryStages[4].add("content expression, relaxed eyes");
  primaryStages[4].add("quiet smile, head turned slightly");
  primaryStages[4].add("gentle amusement, eyes lowered");
  primaryStages[4].add("softly bemused, looking aside");
  primaryStages[4].add("calm, neutral expression");

  // Body descriptors (38 options)
  primaryStages[5] = new ArrayList<String>();
  primaryStages[5].add("athletic build");
  primaryStages[5].add("slim build");
  primaryStages[5].add("curvy body type");
  primaryStages[5].add("visible tattoos on forearm");
  primaryStages[5].add("broad shoulders");
  primaryStages[5].add("petite frame");
  primaryStages[5].add("average build");
  primaryStages[5].add("long limbs");
  primaryStages[5].add("stocky build");
  primaryStages[5].add("lean silhouette");
  primaryStages[5].add("soft body shape");
  primaryStages[5].add("muscular forearms visible");
  primaryStages[5].add("slender waist");
  primaryStages[5].add("rounded shoulders");
  primaryStages[5].add("tall and slim");
  primaryStages[5].add("shorter stature");
  primaryStages[5].add("broad hips");
  primaryStages[5].add("narrow shoulders");
  primaryStages[5].add("visible freckles on arms");
  primaryStages[5].add("light sun tan");
  primaryStages[5].add("pale complexion");
  primaryStages[5].add("strong posture");
  primaryStages[5].add("relaxed shoulders");
  primaryStages[5].add("slight stoop");
  primaryStages[5].add("long neck");
  primaryStages[5].add("compact frame");
  primaryStages[5].add("soft midsection");
  primaryStages[5].add("defined jawline");
  primaryStages[5].add("rounded face");
  primaryStages[5].add("visible collarbones");
  primaryStages[5].add("broad chest");
  primaryStages[5].add("lean legs");
  primaryStages[5].add("callused hands");
  primaryStages[5].add("slight baby weight");
  primaryStages[5].add("toned calves");
  primaryStages[5].add("visible veins on hands");
  primaryStages[5].add("natural posture, relaxed");
  primaryStages[5].add("slight asymmetry in stance");

  // Composition anchors (38 options)
  primaryStages[6] = new ArrayList<String>();
  primaryStages[6].add("centered, negative space to the right");
  primaryStages[6].add("leaning against wall, left side of frame");
  primaryStages[6].add("slight head tilt, off-center composition");
  primaryStages[6].add("foreground subject, blurred background");
  primaryStages[6].add("subject slightly left, leading lines to right");
  primaryStages[6].add("tight framing with window light behind");
  primaryStages[6].add("subject near bottom third, sky visible");
  primaryStages[6].add("balanced with props on either side");
  primaryStages[6].add("subject framed by doorway");
  primaryStages[6].add("subject in lower-left, negative space above");
  primaryStages[6].add("diagonal composition, subject moving right");
  primaryStages[6].add("symmetrical composition, centered subject");
  primaryStages[6].add("subject against textured wall");
  primaryStages[6].add("soft foreground element partially obscuring");
  primaryStages[6].add("subject offset to create breathing room");
  primaryStages[6].add("tight crop with hands visible");
  primaryStages[6].add("subject framed by bookshelf");
  primaryStages[6].add("leading lines from foreground to subject");
  primaryStages[6].add("subject leaning into frame from right");
  primaryStages[6].add("low-angle composition, subject dominant");
  primaryStages[6].add("high-angle, subject small in frame");
  primaryStages[6].add("subject centered with shallow depth");
  primaryStages[6].add("subject placed on left third, open space right");
  primaryStages[6].add("subject framed by archway");
  primaryStages[6].add("soft vignette, subject centered");
  primaryStages[6].add("subject partially behind foreground object");
  primaryStages[6].add("balanced negative space above head");
  primaryStages[6].add("subject aligned with vertical lines");
  primaryStages[6].add("subject in foreground, street in background");
  primaryStages[6].add("subject leaning into negative space");
  primaryStages[6].add("tight portrait with environmental hint");
  primaryStages[6].add("subject slightly off-center, natural pose");
  primaryStages[6].add("subject framed by window light");
  primaryStages[6].add("subject in lower third, sky and buildings above");
  primaryStages[6].add("subject against plain backdrop, natural pose");
  primaryStages[6].add("subject interacting with prop in frame");
  primaryStages[6].add("subject centered with subtle motion blur");
  primaryStages[6].add("subject placed near leading architectural lines");

  // Context or location (39 options)
  primaryStages[7] = new ArrayList<String>();
  primaryStages[7].add("studio portrait");
  primaryStages[7].add("outdoor urban alley");
  primaryStages[7].add("window-lit interior");
  primaryStages[7].add("bathroom mirror selfie");
  primaryStages[7].add("kitchen counter in morning light");
  primaryStages[7].add("local high street, daytime");
  primaryStages[7].add("park bench near river");
  primaryStages[7].add("train station platform");
  primaryStages[7].add("coastal promenade, overcast");
  primaryStages[7].add("cafe table by the window");
  primaryStages[7].add("living room with bookshelf");
  primaryStages[7].add("garden patio with potted plants");
  primaryStages[7].add("commuter street, early morning");
  primaryStages[7].add("market stall area, casual crowd");
  primaryStages[7].add("bookshop aisle, warm light");
  primaryStages[7].add("bus stop on a rainy day");
  primaryStages[7].add("suburban front garden");
  primaryStages[7].add("country lane with hedgerows");
  primaryStages[7].add("ferry terminal, coastal travel");
  primaryStages[7].add("small-town high street, late afternoon");
  primaryStages[7].add("university quad, autumn leaves");
  primaryStages[7].add("local pub beer garden");
  primaryStages[7].add("train carriage window seat");
  primaryStages[7].add("city square with pigeons");
  primaryStages[7].add("farmers market on a Saturday");
  primaryStages[7].add("canal towpath, morning mist");
  primaryStages[7].add("railway bridge, industrial backdrop");
  primaryStages[7].add("seaside pier, muted light");
  primaryStages[7].add("village green with benches");
  primaryStages[7].add("cozy bookshop corner");
  primaryStages[7].add("weekday office kitchen");
  primaryStages[7].add("local bakery storefront");
  primaryStages[7].add("bus interior, natural light");
  primaryStages[7].add("small coastal town street");
  primaryStages[7].add("city rooftop with distant skyline");
  primaryStages[7].add("suburban high street cafe");
  primaryStages[7].add("country pub interior");
  primaryStages[7].add("holiday cottage kitchen");
  primaryStages[7].add("train station concourse, travel vibe");

  // Semantic technical anchors (38 options)
  primaryStages[8] = new ArrayList<String>();
  primaryStages[8].add("mirror selfie");
  primaryStages[8].add("tripod shot");
  primaryStages[8].add("studio lighting setup");
  primaryStages[8].add("candid handheld shot");
  primaryStages[8].add("phone camera at chest height");
  primaryStages[8].add("window-lit natural portrait");
  primaryStages[8].add("overhead kitchen light");
  primaryStages[8].add("golden hour outdoor shot");
  primaryStages[8].add("soft overcast daylight");
  primaryStages[8].add("indoor tungsten lamp");
  primaryStages[8].add("phone selfie with arm extended");
  primaryStages[8].add("camera on table, timer shot");
  primaryStages[8].add("handheld at waist level");
  primaryStages[8].add("shot from slightly above eye level");
  primaryStages[8].add("shot from slightly below eye level");
  primaryStages[8].add("window backlight with reflector");
  primaryStages[8].add("natural window light from left");
  primaryStages[8].add("natural window light from right");
  primaryStages[8].add("softbox-like diffused light");
  primaryStages[8].add("ambient cafe lighting");
  primaryStages[8].add("streetlight evening shot");
  primaryStages[8].add("shopfront window reflection");
  primaryStages[8].add("car interior shot, passenger seat");
  primaryStages[8].add("train window light, motion hint");
  primaryStages[8].add("umbrella overhead, rainy day");
  primaryStages[8].add("doorway light, subject half-lit");
  primaryStages[8].add("soft fill from reflector");
  primaryStages[8].add("natural shade under tree");
  primaryStages[8].add("soft sidelighting from lamp");
  primaryStages[8].add("ambient market stall lighting");
  primaryStages[8].add("overcast diffuse sky, even light");
  primaryStages[8].add("warm kitchen morning light");
  primaryStages[8].add("cool evening window light");
  primaryStages[8].add("handheld phone with slight motion");
  primaryStages[8].add("camera on tripod, slight depth");
  primaryStages[8].add("shot through glass, subtle reflection");
  primaryStages[8].add("soft backlight with rim highlight");
  primaryStages[8].add("natural light with subtle shadow");

  // Initialize indices
  for (int i = 0; i < PRIMARY_COUNT; i++) primaryIndices[i] = 0;
}

// Initialize all Secondary categories (copied from secondary prompter)
void initSecondaryCategories() {
  // Lighting (39 options)
  secondaryCategories[0] = new ArrayList<String>();
  secondaryCategories[0].add("soft natural window light from left");
  secondaryCategories[0].add("soft natural window light from right");
  secondaryCategories[0].add("diffused overcast daylight");
  secondaryCategories[0].add("warm golden hour side light");
  secondaryCategories[0].add("cool early morning light");
  secondaryCategories[0].add("soft backlight with subtle rim");
  secondaryCategories[0].add("ambient indoor daylight, even");
  secondaryCategories[0].add("soft kitchen morning light");
  secondaryCategories[0].add("muted late afternoon light");
  secondaryCategories[0].add("soft shade under tree");
  secondaryCategories[0].add("softbox-like diffused lamp");
  secondaryCategories[0].add("practical lamp, warm tone");
  secondaryCategories[0].add("window light with gentle reflector fill");
  secondaryCategories[0].add("soft sidelighting from lamp");
  secondaryCategories[0].add("overhead soft ambient light");
  secondaryCategories[0].add("soft window backlight with fill");
  secondaryCategories[0].add("streetlight evening, subtle warmth");
  secondaryCategories[0].add("shopfront window light, muted");
  secondaryCategories[0].add("soft cloudy coastal light");
  secondaryCategories[0].add("indoor tungsten with neutral balance");
  secondaryCategories[0].add("soft directional light through blinds");
  secondaryCategories[0].add("soft diffuse light from north-facing window");
  secondaryCategories[0].add("gentle reflector fill, natural look");
  secondaryCategories[0].add("soft golden rim light");
  secondaryCategories[0].add("even daylight with slight shadow");
  secondaryCategories[0].add("soft window light, left side, low contrast");
  secondaryCategories[0].add("muted overcast backlight");
  secondaryCategories[0].add("soft cafe ambient light");
  secondaryCategories[0].add("soft porch light at dusk");
  secondaryCategories[0].add("soft daylight through thin curtain");
  secondaryCategories[0].add("soft natural light, slight warmth");
  secondaryCategories[0].add("soft directional light, low contrast");
  secondaryCategories[0].add("soft evening window light, cool tone");
  secondaryCategories[0].add("soft lamp light with subtle shadow");
  secondaryCategories[0].add("soft daylight with gentle highlights");
  secondaryCategories[0].add("soft natural light, even skin tones");
  secondaryCategories[0].add("soft backlight with subtle lens flare");
  secondaryCategories[0].add("soft ambient market stall light");
  secondaryCategories[0].add("soft train window light, muted");

  // Camera / Perspective (39 options)
  secondaryCategories[1] = new ArrayList<String>();
  secondaryCategories[1].add("eye-level perspective");
  secondaryCategories[1].add("slightly above eye-level");
  secondaryCategories[1].add("slightly below eye-level");
  secondaryCategories[1].add("three-quarter angle, natural");
  secondaryCategories[1].add("straight-on, relaxed");
  secondaryCategories[1].add("slight downward tilt");
  secondaryCategories[1].add("slight upward tilt");
  secondaryCategories[1].add("environmental portrait perspective");
  secondaryCategories[1].add("intimate close perspective");
  secondaryCategories[1].add("medium distance, natural");
  secondaryCategories[1].add("wide environmental perspective");
  secondaryCategories[1].add("tight headshot perspective");
  secondaryCategories[1].add("over-the-shoulder viewpoint");
  secondaryCategories[1].add("candid handheld viewpoint");
  secondaryCategories[1].add("phone-chest-height viewpoint");
  secondaryCategories[1].add("mirror selfie perspective");
  secondaryCategories[1].add("tripod-stable eye-level");
  secondaryCategories[1].add("slight motion perspective, natural");
  secondaryCategories[1].add("table-top timer perspective");
  secondaryCategories[1].add("window-seat perspective");
  secondaryCategories[1].add("bench-side perspective");
  secondaryCategories[1].add("doorway-framed perspective");
  secondaryCategories[1].add("street-level perspective");
  secondaryCategories[1].add("car-interior passenger perspective");
  secondaryCategories[1].add("train-window perspective");
  secondaryCategories[1].add("low-angle, modest dominance");
  secondaryCategories[1].add("high-angle, modest vulnerability");
  secondaryCategories[1].add("three-quarter environmental view");
  secondaryCategories[1].add("tight profile perspective");
  secondaryCategories[1].add("softly off-center perspective");
  secondaryCategories[1].add("balanced centered perspective");
  secondaryCategories[1].add("slight wide-angle environmental");
  secondaryCategories[1].add("natural handheld framing");
  secondaryCategories[1].add("softly cropped portrait perspective");
  secondaryCategories[1].add("mid-distance candid framing");
  secondaryCategories[1].add("soft foreground framing viewpoint");
  secondaryCategories[1].add("slight tilt for casual feel");
  secondaryCategories[1].add("eye-level with slight headroom");
  secondaryCategories[1].add("three-quarter with negative space");

  // Lens / Focal length (20 options)
  secondaryCategories[2] = new ArrayList<String>();
  secondaryCategories[2].add("50mm standard lens look");
  secondaryCategories[2].add("35mm environmental portrait");
  secondaryCategories[2].add("85mm short telephoto portrait");
  secondaryCategories[2].add("24mm slight wide environmental");
  secondaryCategories[2].add("70mm short telephoto feel");
  secondaryCategories[2].add("28mm modest wide angle");
  secondaryCategories[2].add("100mm short telephoto tight portrait");
  secondaryCategories[2].add("40mm natural field of view");
  secondaryCategories[2].add("60mm gentle compression");
  secondaryCategories[2].add("35mm with natural context");
  secondaryCategories[2].add("85mm with soft compression");
  secondaryCategories[2].add("50mm with slight bokeh");
  secondaryCategories[2].add("24-70mm versatile zoom feel");
  secondaryCategories[2].add("35mm slightly intimate");
  secondaryCategories[2].add("50mm close portrait");
  secondaryCategories[2].add("85mm head-and-shoulders");
  secondaryCategories[2].add("28mm for modest environmental hint");
  secondaryCategories[2].add("35mm for casual street feel");
  secondaryCategories[2].add("50mm for natural skin rendering");
  secondaryCategories[2].add("85mm for flattering compression");

  // Color grading / Film style (38 options)
  secondaryCategories[3] = new ArrayList<String>();
  secondaryCategories[3].add("neutral color balance, natural skin tones");
  secondaryCategories[3].add("slightly warm, low saturation");
  secondaryCategories[3].add("muted tones, low contrast");
  secondaryCategories[3].add("soft film-like color, subtle grain");
  secondaryCategories[3].add("cool tones, natural look");
  secondaryCategories[3].add("soft teal and warm highlights, restrained");
  secondaryCategories[3].add("gentle Kodak-like warmth, subtle");
  secondaryCategories[3].add("subtle Portra-inspired warmth");
  secondaryCategories[3].add("faded film look, low contrast");
  secondaryCategories[3].add("clean digital look, minimal processing");
  secondaryCategories[3].add("slightly desaturated, natural");
  secondaryCategories[3].add("soft pastel highlights, restrained");
  secondaryCategories[3].add("warm indoor tungsten balance");
  secondaryCategories[3].add("cool overcast grading, neutral skin");
  secondaryCategories[3].add("soft contrast, natural shadows");
  secondaryCategories[3].add("gentle contrast boost, realistic");
  secondaryCategories[3].add("slight vintage fade, subtle");
  secondaryCategories[3].add("natural color with slight warmth");
  secondaryCategories[3].add("soft film grain and neutral color");
  secondaryCategories[3].add("low-key natural color, realistic");
  secondaryCategories[3].add("muted autumnal palette");
  secondaryCategories[3].add("soft morning warmth, low saturation");
  secondaryCategories[3].add("neutral with slight highlight roll-off");
  secondaryCategories[3].add("clean daylight balance, realistic");
  secondaryCategories[3].add("soft contrast, warm midtones");
  secondaryCategories[3].add("slight cross-processed feel, subtle");
  secondaryCategories[3].add("gentle matte finish, natural");
  secondaryCategories[3].add("soft warm highlights, neutral shadows");
  secondaryCategories[3].add("cool evening tones, restrained");
  secondaryCategories[3].add("soft filmic warmth, low vibrance");
  secondaryCategories[3].add("natural color, slight clarity");
  secondaryCategories[3].add("soft pastel wash, subtle");
  secondaryCategories[3].add("neutral with slight vignette");
  secondaryCategories[3].add("soft cinematic teal-orange, very subtle");
  secondaryCategories[3].add("muted color with natural skin");
  secondaryCategories[3].add("soft warm kitchen tones");
  secondaryCategories[3].add("clean neutral with slight warmth");
  secondaryCategories[3].add("soft low-contrast film look");

  // Depth of field / Bokeh (30 options)
  secondaryCategories[4] = new ArrayList<String>();
  secondaryCategories[4].add("shallow depth of field, soft bokeh");
  secondaryCategories[4].add("moderate depth, background readable");
  secondaryCategories[4].add("deep focus, environmental detail");
  secondaryCategories[4].add("tight bokeh, smooth highlights");
  secondaryCategories[4].add("soft background blur, natural");
  secondaryCategories[4].add("slight background separation");
  secondaryCategories[4].add("soft foreground blur, subject sharp");
  secondaryCategories[4].add("gentle bokeh with circular highlights");
  secondaryCategories[4].add("soft bokeh, low contrast background");
  secondaryCategories[4].add("moderate DOF, subject isolated");
  secondaryCategories[4].add("shallow DOF, eyes sharply focused");
  secondaryCategories[4].add("soft background with texture hint");
  secondaryCategories[4].add("slight bokeh, natural falloff");
  secondaryCategories[4].add("soft edge blur, subject crisp");
  secondaryCategories[4].add("moderate blur, context preserved");
  secondaryCategories[4].add("tight focus on face, soft shoulders");
  secondaryCategories[4].add("soft bokeh with subtle chromatic fringing");
  secondaryCategories[4].add("gentle background blur, natural");
  secondaryCategories[4].add("shallow DOF, subtle rim separation");
  secondaryCategories[4].add("soft bokeh, low-key highlights");
  secondaryCategories[4].add("moderate DOF, hands visible");
  secondaryCategories[4].add("shallow DOF, slight motion blur in background");
  secondaryCategories[4].add("soft background blur, natural depth");
  secondaryCategories[4].add("tight focus on eyes, soft surroundings");
  secondaryCategories[4].add("gentle bokeh, natural falloff");
  secondaryCategories[4].add("moderate DOF for environmental hint");
  secondaryCategories[4].add("soft bokeh, subtle texture in background");
  secondaryCategories[4].add("shallow DOF, natural separation");
  secondaryCategories[4].add("slight background blur, readable context");
  secondaryCategories[4].add("soft bokeh, minimal artifacts");

  // Texture / Finish (20 options)
  secondaryCategories[5] = new ArrayList<String>();
  secondaryCategories[5].add("subtle film grain");
  secondaryCategories[5].add("clean digital finish");
  secondaryCategories[5].add("very light film grain, natural");
  secondaryCategories[5].add("soft clarity, minimal sharpening");
  secondaryCategories[5].add("gentle texture, realistic skin");
  secondaryCategories[5].add("matte finish, low contrast");
  secondaryCategories[5].add("slight clarity boost, natural");
  secondaryCategories[5].add("soft micro-contrast, realistic");
  secondaryCategories[5].add("minimal noise reduction, natural");
  secondaryCategories[5].add("light film grain and subtle texture");
  secondaryCategories[5].add("clean skin rendering, low retouch");
  secondaryCategories[5].add("natural skin texture preserved");
  secondaryCategories[5].add("softened highlights, natural detail");
  secondaryCategories[5].add("subtle sharpening on eyes");
  secondaryCategories[5].add("gentle clarity on facial features");
  secondaryCategories[5].add("minimal post-processing look");
  secondaryCategories[5].add("soft matte skin finish");
  secondaryCategories[5].add("natural pores visible, realistic");
  secondaryCategories[5].add("slight vignette, natural");
  secondaryCategories[5].add("low-key finish, realistic texture");

  // Mood / Subtle effects (20 options)
  secondaryCategories[6] = new ArrayList<String>();
  secondaryCategories[6].add("quiet, candid mood");
  secondaryCategories[6].add("everyday, unposed feel");
  secondaryCategories[6].add("natural, documentary tone");
  secondaryCategories[6].add("gentle nostalgia, restrained");
  secondaryCategories[6].add("calm, approachable atmosphere");
  secondaryCategories[6].add("subtle warmth, homely");
  secondaryCategories[6].add("slight melancholy, natural");
  secondaryCategories[6].add("softly cheerful, candid");
  secondaryCategories[6].add("understated, authentic");
  secondaryCategories[6].add("modest travel vibe, realistic");
  secondaryCategories[6].add("weekday morning routine feel");
  secondaryCategories[6].add("casual weekend mood");
  secondaryCategories[6].add("quiet domestic scene");
  secondaryCategories[6].add("subtle motion hint, natural");
  secondaryCategories[6].add("softly reflective mood");
  secondaryCategories[6].add("low-key documentary feel");
  secondaryCategories[6].add("gentle intimacy, not posed");
  secondaryCategories[6].add("everyday errand, candid");
  secondaryCategories[6].add("subtle story-telling, natural");
  secondaryCategories[6].add("restrained, competent amateur look");

  // Initialize indices
  for (int i = 0; i < SECONDARY_COUNT; i++) secondaryIndices[i] = 0;
}
