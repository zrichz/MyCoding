/**
 PrimaryPromptSchemaGenerator_Expanded.pde
 Processing sketch to autogenerate Primary prompts for SDXL self-portraits/body shots.
 - Each of the 9 stages contains 40 realistic, everyday options focused on UK settings.
 - Use "Generate N" to create multiple randomized prompts and "Save" to export.
*/

import java.util.ArrayList;

final int STAGE_COUNT = 9;
ArrayList<String>[] stages = (ArrayList<String>[]) new ArrayList[STAGE_COUNT];
String[] stageNames = {
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

int[] indices = new int[STAGE_COUNT];
ArrayList<String> generated = new ArrayList<String>();

// UI layout
int leftMargin = 20;
int topMargin = 20;
int stageHeight = 48;
int stageWidth = 540;
int rightPanelX;
int buttonW = 120;
int buttonH = 28;
int generateCount = 5;

void setup() {
  size(1200, 720);
  rightPanelX = leftMargin + stageWidth + 20;
  textFont(createFont("Arial", 14));
  initStageOptions();
}

void initStageOptions() {
  // Populate 40 realistic options for each stage. Edit these lists to suit your needs.
  stages[0] = new ArrayList<String>(); // Subject identity (40)
  stages[0].add("female, early 30s, freckled skin");
  stages[0].add("male, late 20s, light stubble");
  stages[0].add("nonbinary, mid 20s, short hair");
  stages[0].add("female, mid 40s, glasses");
  stages[0].add("male, early 40s, salt-and-pepper beard");
  stages[0].add("female, late 20s, long curly hair");
  stages[0].add("male, mid 30s, shaved head");
  stages[0].add("female, early 20s, natural makeup");
  stages[0].add("male, late 30s, athletic build");
  stages[0].add("female, mid 50s, silver hair");
  stages[0].add("male, early 20s, gap-toothed smile");
  stages[0].add("female, late 30s, freckles and glasses");
  stages[0].add("male, mid 50s, lined face");
  stages[0].add("female, early 40s, short bob");
  stages[0].add("male, late 40s, receding hairline");
  stages[0].add("female, mid 20s, pierced ear");
  stages[0].add("male, early 30s, tattooed forearm");
  stages[0].add("female, late 40s, warm complexion");
  stages[0].add("male, mid 20s, tousled hair");
  stages[0].add("female, early 50s, soft features");
  stages[0].add("male, late 20s, glasses and beard");
  stages[0].add("female, mid 30s, athletic build");
  stages[0].add("male, early 60s, weathered hands");
  stages[0].add("female, late 20s, short cropped hair");
  stages[0].add("male, mid 40s, wearing a cap");
  stages[0].add("female, early 30s, light tan");
  stages[0].add("male, late 30s, warm smile");
  stages[0].add("female, mid 20s, natural freckles");
  stages[0].add("male, early 50s, salt-and-pepper stubble");
  stages[0].add("female, late 30s, long straight hair");
  stages[0].add("male, mid 20s, slim build");
  stages[0].add("female, early 40s, wearing glasses");
  stages[0].add("male, late 50s, lined smile");
  stages[0].add("female, mid 30s, visible birthmark");
  stages[0].add("male, early 40s, casual beard");
  stages[0].add("female, late 20s, relaxed posture");
  stages[0].add("male, mid 30s, cropped hair and stubble");

  stages[1] = new ArrayList<String>(); // Pose and action (40)
  stages[1].add("three-quarter turn, left arm raised");
  stages[1].add("standing, hands in pockets");
  stages[1].add("seated, one knee up");
  stages[1].add("walking toward camera");
  stages[1].add("sitting on a low wall, legs crossed");
  stages[1].add("leaning on a railing, looking down");
  stages[1].add("holding a mug with both hands");
  stages[1].add("adjusting jacket collar casually");
  stages[1].add("reading a paperback, head tilted");
  stages[1].add("tying shoelace, looking away");
  stages[1].add("brushing hair back with one hand");
  stages[1].add("standing with arms folded loosely");
  stages[1].add("sitting on stairs, elbows on knees");
  stages[1].add("walking dog on a short lead");
  stages[1].add("holding a takeaway coffee, mid-sip");
  stages[1].add("checking phone while standing");
  stages[1].add("leaning against doorframe, relaxed");
  stages[1].add("sitting at table, hands clasped");
  stages[1].add("looking out of a window, thoughtful");
  stages[1].add("standing under an umbrella, slight smile");
  stages[1].add("carrying a backpack, mid-step");
  stages[1].add("sitting on a bench, one arm draped");
  stages[1].add("holding bicycle by the handlebars");
  stages[1].add("walking up a short flight of steps");
  stages[1].add("adjusting glasses, slight tilt of head");
  stages[1].add("reaching for a shelf, casual stance");
  stages[1].add("sitting cross-legged on the floor");
  stages[1].add("leaning over a kitchen counter");
  stages[1].add("standing with one foot on a low step");
  stages[1].add("holding a camera at chest height");
  stages[1].add("sitting in a cafe booth, relaxed");
  stages[1].add("standing by a window, hands in pockets");
  stages[1].add("walking past a shopfront, glancing");
  stages[1].add("sitting on a windowsill, knees drawn up");
  stages[1].add("holding a newspaper, reading");
  stages[1].add("standing with coat draped over shoulder");
  stages[1].add("leaning on a bicycle, casual");
  stages[1].add("sitting on a low wall, feet dangling");
  stages[1].add("mid-laugh, head thrown back slightly");

  stages[2] = new ArrayList<String>(); // Framing and crop (40)
  stages[2].add("waist-up");
  stages[2].add("head and shoulders");
  stages[2].add("full body");
  stages[2].add("three-quarter body");
  stages[2].add("close-up face");
  stages[2].add("knee-up");
  stages[2].add("hip-up");
  stages[2].add("environmental portrait, subject small in frame");
  stages[2].add("tight portrait, eyes centered");
  stages[2].add("half-body, slight tilt");
  stages[2].add("over-the-shoulder crop");
  stages[2].add("waist-up, slight left offset");
  stages[2].add("head-to-toe, centered");
  stages[2].add("upper torso, three-quarter turn");
  stages[2].add("close crop on hands and face");
  stages[2].add("mid-shot with foreground blur");
  stages[2].add("portrait orientation, headroom");
  stages[2].add("landscape orientation, subject left");
  stages[2].add("tight headshot with soft bokeh");
  stages[2].add("full body with negative space above");
  stages[2].add("waist-up, slight downward angle");
  stages[2].add("head and shoulders, eye-level");
  stages[2].add("three-quarter body, slight wide angle");
  stages[2].add("close-up of profile");
  stages[2].add("mid-shot with environmental context");
  stages[2].add("candid half-body crop");
  stages[2].add("full body, slight motion blur");
  stages[2].add("tight portrait, off-center composition");
  stages[2].add("head and shoulders, soft framing");
  stages[2].add("waist-up, natural posture");
  stages[2].add("full body, slight low angle");
  stages[2].add("close-up with hands visible");
  stages[2].add("three-quarter body, centered");
  stages[2].add("mid-shot with foreground element");
  stages[2].add("tight crop on face and shoulders");
  stages[2].add("full body, environmental detail visible");
  stages[2].add("head and shoulders, slight side lighting");
  stages[2].add("waist-up, casual stance");
  stages[2].add("close-up with soft edge vignette");

  stages[3] = new ArrayList<String>(); // Clothing and key props (40)
  stages[3].add("cream linen shirt");
  stages[3].add("black leather jacket");
  stages[3].add("striped sweater");
  stages[3].add("tailored coat and scarf");
  stages[3].add("denim jacket and white tee");
  stages[3].add("wool jumper and jeans");
  stages[3].add("floral dress, simple cut");
  stages[3].add("casual hoodie and trainers");
  stages[3].add("button-up shirt and chinos");
  stages[3].add("raincoat and wellies");
  stages[3].add("t-shirt and cardigan");
  stages[3].add("checked shirt and denim");
  stages[3].add("knit sweater and skirt");
  stages[3].add("polo shirt and jeans");
  stages[3].add("oversized knit and leggings");
  stages[3].add("work jacket and boots");
  stages[3].add("simple blouse and trousers");
  stages[3].add("sweatshirt and joggers");
  stages[3].add("casual blazer and tee");
  stages[3].add("striped tee and denim shorts");
  stages[3].add("light rain jacket and umbrella");
  stages[3].add("wool coat and scarf");
  stages[3].add("checked scarf and beanie");
  stages[3].add("summer dress and sandals");
  stages[3].add("puffer jacket and jeans");
  stages[3].add("work shirt and apron");
  stages[3].add("school uniform style blazer");
  stages[3].add("cycling jacket and helmet");
  stages[3].add("sweater vest and shirt");
  stages[3].add("linen trousers and loafers");
  stages[3].add("casual shirt and backpack");
  stages[3].add("striped jumper and coat");
  stages[3].add("simple tee and denim jacket");
  stages[3].add("fitted coat and scarf");
  stages[3].add("overshirt and chinos");
  stages[3].add("wool hat and gloves");
  stages[3].add("checked shirt and boots");
  stages[3].add("light cardigan and jeans");
  stages[3].add("work boots and utility jacket");

  stages[4] = new ArrayList<String>(); // Expression and gaze (40)
  stages[4].add("soft smile, looking slightly off-camera");
  stages[4].add("neutral expression, direct gaze");
  stages[4].add("candid smile, eyes down");
  stages[4].add("contemplative, looking to the left");
  stages[4].add("gentle laugh, head tilted");
  stages[4].add("focused, looking at hands");
  stages[4].add("relaxed, eyes half-closed");
  stages[4].add("slight grin, looking away");
  stages[4].add("thoughtful, distant gaze");
  stages[4].add("subtle smile, direct eye contact");
  stages[4].add("soft expression, looking up");
  stages[4].add("calm, eyes on horizon");
  stages[4].add("mild amusement, glancing aside");
  stages[4].add("serene, slight smile");
  stages[4].add("pensive, looking downwards");
  stages[4].add("warm smile, eyes crinkled");
  stages[4].add("reserved smile, head slightly bowed");
  stages[4].add("gentle smirk, looking off-frame");
  stages[4].add("open smile, candid expression");
  stages[4].add("softly serious, direct gaze");
  stages[4].add("subdued smile, eyes to camera");
  stages[4].add("mild surprise, eyebrows raised");
  stages[4].add("relaxed grin, looking to side");
  stages[4].add("quiet contentment, eyes closed briefly");
  stages[4].add("slight frown, thoughtful");
  stages[4].add("gentle curiosity, head tilt");
  stages[4].add("soft laugh, looking down");
  stages[4].add("calm, steady gaze");
  stages[4].add("subtle amusement, eyes to left");
  stages[4].add("warm, approachable smile");
  stages[4].add("reflective, distant look");
  stages[4].add("mildly inquisitive, direct gaze");
  stages[4].add("soft grin, slight squint");
  stages[4].add("content expression, relaxed eyes");
  stages[4].add("quiet smile, head turned slightly");
  stages[4].add("gentle amusement, eyes lowered");
  stages[4].add("softly bemused, looking aside");
  stages[4].add("calm, neutral expression");

  stages[5] = new ArrayList<String>(); // Body descriptors (40)
  stages[5].add("athletic build");
  stages[5].add("slim build");
  stages[5].add("curvy body type");
  stages[5].add("visible tattoos on forearm");
  stages[5].add("broad shoulders");
  stages[5].add("petite frame");
  stages[5].add("average build");
  stages[5].add("long limbs");
  stages[5].add("stocky build");
  stages[5].add("lean silhouette");
  stages[5].add("soft body shape");
  stages[5].add("muscular forearms visible");
  stages[5].add("slender waist");
  stages[5].add("rounded shoulders");
  stages[5].add("tall and slim");
  stages[5].add("shorter stature");
  stages[5].add("broad hips");
  stages[5].add("narrow shoulders");
  stages[5].add("visible freckles on arms");
  stages[5].add("light sun tan");
  stages[5].add("pale complexion");
  stages[5].add("strong posture");
  stages[5].add("relaxed shoulders");
  stages[5].add("slight stoop");
  stages[5].add("long neck");
  stages[5].add("compact frame");
  stages[5].add("soft midsection");
  stages[5].add("defined jawline");
  stages[5].add("rounded face");
  stages[5].add("visible collarbones");
  stages[5].add("broad chest");
  stages[5].add("lean legs");
  stages[5].add("callused hands");
  stages[5].add("slight baby weight");
  stages[5].add("toned calves");
  stages[5].add("visible veins on hands");
  stages[5].add("natural posture, relaxed");
  stages[5].add("slight asymmetry in stance");

  stages[6] = new ArrayList<String>(); // Composition anchors (40)
  stages[6].add("centered, negative space to the right");
  stages[6].add("leaning against wall, left side of frame");
  stages[6].add("slight head tilt, off-center composition");
  stages[6].add("foreground subject, blurred background");
  stages[6].add("subject slightly left, leading lines to right");
  stages[6].add("tight framing with window light behind");
  stages[6].add("subject near bottom third, sky visible");
  stages[6].add("balanced with props on either side");
  stages[6].add("subject framed by doorway");
  stages[6].add("subject in lower-left, negative space above");
  stages[6].add("diagonal composition, subject moving right");
  stages[6].add("symmetrical composition, centered subject");
  stages[6].add("subject against textured wall");
  stages[6].add("soft foreground element partially obscuring");
  stages[6].add("subject offset to create breathing room");
  stages[6].add("tight crop with hands visible");
  stages[6].add("subject framed by bookshelf");
  stages[6].add("leading lines from foreground to subject");
  stages[6].add("subject leaning into frame from right");
  stages[6].add("low-angle composition, subject dominant");
  stages[6].add("high-angle, subject small in frame");
  stages[6].add("subject centered with shallow depth");
  stages[6].add("subject placed on left third, open space right");
  stages[6].add("subject framed by archway");
  stages[6].add("soft vignette, subject centered");
  stages[6].add("subject partially behind foreground object");
  stages[6].add("balanced negative space above head");
  stages[6].add("subject aligned with vertical lines");
  stages[6].add("subject in foreground, street in background");
  stages[6].add("subject leaning into negative space");
  stages[6].add("tight portrait with environmental hint");
  stages[6].add("subject slightly off-center, natural pose");
  stages[6].add("subject framed by window light");
  stages[6].add("subject in lower third, sky and buildings above");
  stages[6].add("subject against plain backdrop, natural pose");
  stages[6].add("subject interacting with prop in frame");
  stages[6].add("subject centered with subtle motion blur");
  stages[6].add("subject placed near leading architectural lines");

  stages[7] = new ArrayList<String>(); // Context or location (40)
  stages[7].add("studio portrait");
  stages[7].add("outdoor urban alley");
  stages[7].add("window-lit interior");
  stages[7].add("bathroom mirror selfie");
  stages[7].add("kitchen counter in morning light");
  stages[7].add("local high street, daytime");
  stages[7].add("park bench near river");
  stages[7].add("train station platform");
  stages[7].add("coastal promenade, overcast");
  stages[7].add("cafe table by the window");
  stages[7].add("living room with bookshelf");
  stages[7].add("garden patio with potted plants");
  stages[7].add("commuter street, early morning");
  stages[7].add("market stall area, casual crowd");
  stages[7].add("bookshop aisle, warm light");
  stages[7].add("bus stop on a rainy day");
  stages[7].add("suburban front garden");
  stages[7].add("country lane with hedgerows");
  stages[7].add("ferry terminal, coastal travel");
  stages[7].add("small-town high street, late afternoon");
  stages[7].add("university quad, autumn leaves");
  stages[7].add("local pub beer garden");
  stages[7].add("train carriage window seat");
  stages[7].add("city square with pigeons");
  stages[7].add("farmers market on a Saturday");
  stages[7].add("canal towpath, morning mist");
  stages[7].add("railway bridge, industrial backdrop");
  stages[7].add("seaside pier, muted light");
  stages[7].add("village green with benches");
  stages[7].add("cozy bookshop corner");
  stages[7].add("weekday office kitchen");
  stages[7].add("local bakery storefront");
  stages[7].add("bus interior, natural light");
  stages[7].add("small coastal town street");
  stages[7].add("city rooftop with distant skyline");
  stages[7].add("suburban high street cafe");
  stages[7].add("country pub interior");
  stages[7].add("holiday cottage kitchen");
  stages[7].add("train station concourse, travel vibe");

  stages[8] = new ArrayList<String>(); // Semantic technical anchors (40)
  stages[8].add("mirror selfie");
  stages[8].add("tripod shot");
  stages[8].add("studio lighting setup");
  stages[8].add("candid handheld shot");
  stages[8].add("phone camera at chest height");
  stages[8].add("window-lit natural portrait");
  stages[8].add("overhead kitchen light");
  stages[8].add("golden hour outdoor shot");
  stages[8].add("soft overcast daylight");
  stages[8].add("indoor tungsten lamp");
  stages[8].add("phone selfie with arm extended");
  stages[8].add("camera on table, timer shot");
  stages[8].add("handheld at waist level");
  stages[8].add("shot from slightly above eye level");
  stages[8].add("shot from slightly below eye level");
  stages[8].add("window backlight with reflector");
  stages[8].add("natural window light from left");
  stages[8].add("natural window light from right");
  stages[8].add("softbox-like diffused light");
  stages[8].add("ambient cafe lighting");
  stages[8].add("streetlight evening shot");
  stages[8].add("shopfront window reflection");
  stages[8].add("car interior shot, passenger seat");
  stages[8].add("train window light, motion hint");
  stages[8].add("umbrella overhead, rainy day");
  stages[8].add("doorway light, subject half-lit");
  stages[8].add("soft fill from reflector");
  stages[8].add("natural shade under tree");
  stages[8].add("soft sidelighting from lamp");
  stages[8].add("ambient market stall lighting");
  stages[8].add("overcast diffuse sky, even light");
  stages[8].add("warm kitchen morning light");
  stages[8].add("cool evening window light");
  stages[8].add("handheld phone with slight motion");
  stages[8].add("camera on tripod, slight depth");
  stages[8].add("shot through glass, subtle reflection");
  stages[8].add("soft backlight with rim highlight");
  stages[8].add("natural light with subtle shadow")

  ; // end of stages[8] additions

  // initialize indices
  for (int i = 0; i < STAGE_COUNT; i++) indices[i] = 0;
}

void draw() {
  background(245);
  fill(30);
  textSize(18);
  text("Primary Prompt Schema Generator (Expanded)", leftMargin, topMargin - 2);
  textSize(12);
  text("Click Cycle to step options. Use Generate to create randomized prompts.", leftMargin, topMargin + 18);

  // Draw stage boxes
  for (int i = 0; i < STAGE_COUNT; i++) {
    int y = topMargin + 40 + i * stageHeight;
    drawStageBox(i, leftMargin, y, stageWidth, stageHeight - 6);
  }

  // Right panel: controls and generated list
  drawRightPanel();
}

void drawStageBox(int idx, int x, int y, int w, int h) {
  stroke(200);
  fill(255);
  rect(x, y, w, h, 6);
  fill(0);
  textSize(12);
  text(stageNames[idx], x + 8, y + 16);
  textSize(11);
  String current = stages[idx].get(indices[idx]);
  text(current, x + 8, y + 34);

  // Cycle button
  int bx = x + w - buttonW - 12;
  int by = y + 8;
  fill(220);
  rect(bx, by, buttonW, buttonH, 6);
  fill(0);
  textAlign(CENTER, CENTER);
  text("Cycle", bx + buttonW/2, by + buttonH/2);
  textAlign(LEFT);
}

void drawRightPanel() {
  int x = rightPanelX;
  int y = topMargin + 40;
  int w = width - x - 20;
  int h = height - y - 20;
  stroke(200);
  fill(255);
  rect(x, y, w, h, 6);

  fill(0);
  textSize(12);
  text("Controls", x + 12, y + 18);

  // Generate count input label
  textSize(11);
  text("Generate count (N): " + generateCount, x + 12, y + 40);

  // Buttons
  int bx = x + 12;
  int by = y + 52;
  drawButton(bx, by, "Generate N", buttonW, buttonH);
  drawButton(bx + buttonW + 12, by, "Save", buttonW, buttonH);
  drawButton(bx + 2*(buttonW + 12), by, "Clear", buttonW, buttonH);

  // Show current assembled primary prompt preview
  textSize(12);
  text("Preview Primary Prompt", x + 12, by + buttonH + 28);
  String preview = assemblePrimaryPrompt(false);
  textSize(11);
  textLeading(14);
  text(preview, x + 12, by + buttonH + 44, w - 24, 120);

  // Generated prompts list
  textSize(12);
  text("Generated Prompts", x + 12, by + buttonH + 180);
  textSize(11);
  int listY = by + buttonH + 200;
  for (int i = 0; i < generated.size(); i++) {
    text((i+1) + ". " + generated.get(i), x + 12, listY + i * 18);
  }

  // Instructions
  textSize(10);
  fill(80);
  text("Tip: Edit the option arrays in initStageOptions() to customize vocabulary.", x + 12, height - 28);
}

void drawButton(int x, int y, String label, int w, int h) {
  fill(220);
  rect(x, y, w, h, 6);
  fill(0);
  textAlign(CENTER, CENTER);
  text(label, x + w/2, y + h/2);
  textAlign(LEFT);
}

void mousePressed() {
  // Check stage cycle buttons
  for (int i = 0; i < STAGE_COUNT; i++) {
    int y = topMargin + 40 + i * stageHeight;
    int bx = leftMargin + stageWidth - buttonW - 12;
    int by = y + 8;
    if (mouseX >= bx && mouseX <= bx + buttonW && mouseY >= by && mouseY <= by + buttonH) {
      indices[i] = (indices[i] + 1) % stages[i].size();
      return;
    }
  }

  // Right panel buttons
  int x = rightPanelX + 12;
  int by = topMargin + 40 + 52;
  if (mouseX >= x && mouseX <= x + buttonW && mouseY >= by && mouseY <= by + buttonH) {
    // Generate N
    generatePrompts(generateCount);
    return;
  }
  if (mouseX >= x + buttonW + 12 && mouseX <= x + 2*buttonW + 12 && mouseY >= by && mouseY <= by + buttonH) {
    // Save
    saveGeneratedPrompts();
    return;
  }
  if (mouseX >= x + 2*(buttonW + 12) && mouseX <= x + 3*buttonW + 24 && mouseY >= by && mouseY <= by + buttonH) {
    // Clear
    generated.clear();
    return;
  }
}

String assemblePrimaryPrompt(boolean includeLabel) {
  // Build the primary prompt in the specified order, using semicolons between clauses
  StringBuilder sb = new StringBuilder();
  if (includeLabel) sb.append("Primary: ");
  for (int i = 0; i < STAGE_COUNT; i++) {
    String clause = stages[i].get(indices[i]);
    sb.append(clause);
    if (i < STAGE_COUNT - 1) sb.append("; ");
  }
  return sb.toString();
}

void generatePrompts(int n) {
  for (int k = 0; k < n; k++) {
    // Randomly pick one option from each stage
    StringBuilder sb = new StringBuilder();
    sb.append("Primary: ");
    for (int i = 0; i < STAGE_COUNT; i++) {
      int r = (int) random(stages[i].size());
      String clause = stages[i].get(r);
      sb.append(clause);
      if (i < STAGE_COUNT - 1) sb.append("; ");
    }
    generated.add(sb.toString());
  }
}

void saveGeneratedPrompts() {
  if (generated.size() == 0) return;
  String[] out = new String[generated.size()];
  for (int i = 0; i < generated.size(); i++) out[i] = generated.get(i);
  saveStrings("generated_primary_prompts.txt", out);
}

// Optional keyboard controls
void keyPressed() {
  if (key == '+') generateCount++;
  if (key == '-' && generateCount > 1) generateCount--;
  if (key == 'g' || key == 'G') generatePrompts(generateCount);
  if (key == 'c' || key == 'C') generated.clear();
}
