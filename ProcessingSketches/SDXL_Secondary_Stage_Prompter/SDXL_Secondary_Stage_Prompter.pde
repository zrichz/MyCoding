/**
 SecondaryStyleGenerator.pde
Processing sketch to autogenerate Secondary style prompts for SDXL.
- Multiple style categories (lighting, camera, lens, color, DOF, texture, mood).
- Each category contains realistic, restrained options aimed at amateur, natural photography.
- Click Cycle to step options. Use Generate N to create multiple randomized secondary prompts.
- Save writes generated prompts to a text file.
Author: Copilot
Date: 2026-01-09
*/

import java.util.ArrayList;

final int CAT_COUNT = 7;
ArrayList<String>[] cats = (ArrayList<String>[]) new ArrayList[CAT_COUNT];
String[] catNames = {
  "Lighting",
  "Camera / Perspective",
  "Lens / Focal length",
  "Color grading / Film style",
  "Depth of field / Bokeh",
  "Texture / Finish",
  "Mood / Subtle effects"
};

int[] indices = new int[CAT_COUNT];
ArrayList<String> generated = new ArrayList<String>();

// UI layout
int leftMargin = 20;
int topMargin = 20;
int catHeight = 56;
int catWidth = 560;
int rightPanelX;
int buttonW = 120;
int buttonH = 28;
int generateCount = 6;

void setup() {
  size(1280, 720);
  rightPanelX = leftMargin + catWidth + 20;
  textFont(createFont("Arial", 14));
  initCategories();
}

void initCategories() {
  // Lighting (40)
  cats[0] = new ArrayList<String>();
  cats[0].add("soft natural window light from left");
  cats[0].add("soft natural window light from right");
  cats[0].add("diffused overcast daylight");
  cats[0].add("warm golden hour side light");
  cats[0].add("cool early morning light");
  cats[0].add("soft backlight with subtle rim");
  cats[0].add("ambient indoor daylight, even");
  cats[0].add("soft kitchen morning light");
  cats[0].add("muted late afternoon light");
  cats[0].add("soft shade under tree");
  cats[0].add("softbox-like diffused lamp");
  cats[0].add("practical lamp, warm tone");
  cats[0].add("window light with gentle reflector fill");
  cats[0].add("soft sidelighting from lamp");
  cats[0].add("overhead soft ambient light");
  cats[0].add("soft window backlight with fill");
  cats[0].add("streetlight evening, subtle warmth");
  cats[0].add("shopfront window light, muted");
  cats[0].add("soft cloudy coastal light");
  cats[0].add("indoor tungsten with neutral balance");
  cats[0].add("soft directional light through blinds");
  cats[0].add("soft diffuse light from north-facing window");
  cats[0].add("gentle reflector fill, natural look");
  cats[0].add("soft golden rim light");
  cats[0].add("even daylight with slight shadow");
  cats[0].add("soft window light, left side, low contrast");
  cats[0].add("muted overcast backlight");
  cats[0].add("soft cafe ambient light");
  cats[0].add("soft porch light at dusk");
  cats[0].add("soft daylight through thin curtain");
  cats[0].add("soft natural light, slight warmth");
  cats[0].add("soft directional light, low contrast");
  cats[0].add("soft evening window light, cool tone");
  cats[0].add("soft lamp light with subtle shadow");
  cats[0].add("soft daylight with gentle highlights");
  cats[0].add("soft natural light, even skin tones");
  cats[0].add("soft backlight with subtle lens flare");
  cats[0].add("soft ambient market stall light");
  cats[0].add("soft train window light, muted");

  // Camera / Perspective (40)
  cats[1] = new ArrayList<String>();
  cats[1].add("eye-level perspective");
  cats[1].add("slightly above eye-level");
  cats[1].add("slightly below eye-level");
  cats[1].add("three-quarter angle, natural");
  cats[1].add("straight-on, relaxed");
  cats[1].add("slight downward tilt");
  cats[1].add("slight upward tilt");
  cats[1].add("environmental portrait perspective");
  cats[1].add("intimate close perspective");
  cats[1].add("medium distance, natural");
  cats[1].add("wide environmental perspective");
  cats[1].add("tight headshot perspective");
  cats[1].add("over-the-shoulder viewpoint");
  cats[1].add("candid handheld viewpoint");
  cats[1].add("phone-chest-height viewpoint");
  cats[1].add("mirror selfie perspective");
  cats[1].add("tripod-stable eye-level");
  cats[1].add("slight motion perspective, natural");
  cats[1].add("table-top timer perspective");
  cats[1].add("window-seat perspective");
  cats[1].add("bench-side perspective");
  cats[1].add("doorway-framed perspective");
  cats[1].add("street-level perspective");
  cats[1].add("car-interior passenger perspective");
  cats[1].add("train-window perspective");
  cats[1].add("low-angle, modest dominance");
  cats[1].add("high-angle, modest vulnerability");
  cats[1].add("three-quarter environmental view");
  cats[1].add("tight profile perspective");
  cats[1].add("softly off-center perspective");
  cats[1].add("balanced centered perspective");
  cats[1].add("slight wide-angle environmental");
  cats[1].add("natural handheld framing");
  cats[1].add("softly cropped portrait perspective");
  cats[1].add("mid-distance candid framing");
  cats[1].add("soft foreground framing viewpoint");
  cats[1].add("slight tilt for casual feel");
  cats[1].add("eye-level with slight headroom");
  cats[1].add("three-quarter with negative space");

  // Lens / Focal length (20)
  cats[2] = new ArrayList<String>();
  cats[2].add("50mm standard lens look");
  cats[2].add("35mm environmental portrait");
  cats[2].add("85mm short telephoto portrait");
  cats[2].add("24mm slight wide environmental");
  cats[2].add("70mm short telephoto feel");
  cats[2].add("28mm modest wide angle");
  cats[2].add("100mm short telephoto tight portrait");
  cats[2].add("40mm natural field of view");
  cats[2].add("60mm gentle compression");
  cats[2].add("35mm with natural context");
  cats[2].add("85mm with soft compression");
  cats[2].add("50mm with slight bokeh");
  cats[2].add("24-70mm versatile zoom feel");
  cats[2].add("35mm slightly intimate");
  cats[2].add("50mm close portrait");
  cats[2].add("85mm head-and-shoulders");
  cats[2].add("28mm for modest environmental hint");
  cats[2].add("35mm for casual street feel");
  cats[2].add("50mm for natural skin rendering");
  cats[2].add("85mm for flattering compression");

  // Color grading / Film style (40)
  cats[3] = new ArrayList<String>();
  cats[3].add("neutral color balance, natural skin tones");
  cats[3].add("slightly warm, low saturation");
  cats[3].add("muted tones, low contrast");
  cats[3].add("soft film-like color, subtle grain");
  cats[3].add("cool tones, natural look");
  cats[3].add("soft teal and warm highlights, restrained");
  cats[3].add("gentle Kodak-like warmth, subtle");
  cats[3].add("subtle Portra-inspired warmth");
  cats[3].add("faded film look, low contrast");
  cats[3].add("clean digital look, minimal processing");
  cats[3].add("slightly desaturated, natural");
  cats[3].add("soft pastel highlights, restrained");
  cats[3].add("warm indoor tungsten balance");
  cats[3].add("cool overcast grading, neutral skin");
  cats[3].add("soft contrast, natural shadows");
  cats[3].add("gentle contrast boost, realistic");
  cats[3].add("slight vintage fade, subtle");
  cats[3].add("natural color with slight warmth");
  cats[3].add("soft film grain and neutral color");
  cats[3].add("low-key natural color, realistic");
  cats[3].add("muted autumnal palette");
  cats[3].add("soft morning warmth, low saturation");
  cats[3].add("neutral with slight highlight roll-off");
  cats[3].add("clean daylight balance, realistic");
  cats[3].add("soft contrast, warm midtones");
  cats[3].add("slight cross-processed feel, subtle");
  cats[3].add("gentle matte finish, natural");
  cats[3].add("soft warm highlights, neutral shadows");
  cats[3].add("cool evening tones, restrained");
  cats[3].add("soft filmic warmth, low vibrance");
  cats[3].add("natural color, slight clarity");
  cats[3].add("soft pastel wash, subtle");
  cats[3].add("neutral with slight vignette");
  cats[3].add("soft cinematic teal-orange, very subtle");
  cats[3].add("muted color with natural skin");
  cats[3].add("soft warm kitchen tones");
  cats[3].add("clean neutral with slight warmth");
  cats[3].add("soft low-contrast film look");

  // Depth of field / Bokeh (30)
  cats[4] = new ArrayList<String>();
  cats[4].add("shallow depth of field, soft bokeh");
  cats[4].add("moderate depth, background readable");
  cats[4].add("deep focus, environmental detail");
  cats[4].add("tight bokeh, smooth highlights");
  cats[4].add("soft background blur, natural");
  cats[4].add("slight background separation");
  cats[4].add("soft foreground blur, subject sharp");
  cats[4].add("gentle bokeh with circular highlights");
  cats[4].add("soft bokeh, low contrast background");
  cats[4].add("moderate DOF, subject isolated");
  cats[4].add("shallow DOF, eyes sharply focused");
  cats[4].add("soft background with texture hint");
  cats[4].add("slight bokeh, natural falloff");
  cats[4].add("soft edge blur, subject crisp");
  cats[4].add("moderate blur, context preserved");
  cats[4].add("tight focus on face, soft shoulders");
  cats[4].add("soft bokeh with subtle chromatic fringing");
  cats[4].add("gentle background blur, natural");
  cats[4].add("shallow DOF, subtle rim separation");
  cats[4].add("soft bokeh, low-key highlights");
  cats[4].add("moderate DOF, hands visible");
  cats[4].add("shallow DOF, slight motion blur in background");
  cats[4].add("soft background blur, natural depth");
  cats[4].add("tight focus on eyes, soft surroundings");
  cats[4].add("gentle bokeh, natural falloff");
  cats[4].add("moderate DOF for environmental hint");
  cats[4].add("soft bokeh, subtle texture in background");
  cats[4].add("shallow DOF, natural separation");
  cats[4].add("slight background blur, readable context");
  cats[4].add("soft bokeh, minimal artifacts");

  // Texture / Finish (20)
  cats[5] = new ArrayList<String>();
  cats[5].add("subtle film grain");
  cats[5].add("clean digital finish");
  cats[5].add("very light film grain, natural");
  cats[5].add("soft clarity, minimal sharpening");
  cats[5].add("gentle texture, realistic skin");
  cats[5].add("matte finish, low contrast");
  cats[5].add("slight clarity boost, natural");
  cats[5].add("soft micro-contrast, realistic");
  cats[5].add("minimal noise reduction, natural");
  cats[5].add("light film grain and subtle texture");
  cats[5].add("clean skin rendering, low retouch");
  cats[5].add("natural skin texture preserved");
  cats[5].add("softened highlights, natural detail");
  cats[5].add("subtle sharpening on eyes");
  cats[5].add("gentle clarity on facial features");
  cats[5].add("minimal post-processing look");
  cats[5].add("soft matte skin finish");
  cats[5].add("natural pores visible, realistic");
  cats[5].add("slight vignette, natural");
  cats[5].add("low-key finish, realistic texture");

  // Mood / Subtle effects (20)
  cats[6] = new ArrayList<String>();
  cats[6].add("quiet, candid mood");
  cats[6].add("everyday, unposed feel");
  cats[6].add("natural, documentary tone");
  cats[6].add("gentle nostalgia, restrained");
  cats[6].add("calm, approachable atmosphere");
  cats[6].add("subtle warmth, homely");
  cats[6].add("slight melancholy, natural");
  cats[6].add("softly cheerful, candid");
  cats[6].add("understated, authentic");
  cats[6].add("modest travel vibe, realistic");
  cats[6].add("weekday morning routine feel");
  cats[6].add("casual weekend mood");
  cats[6].add("quiet domestic scene");
  cats[6].add("subtle motion hint, natural");
  cats[6].add("softly reflective mood");
  cats[6].add("low-key documentary feel");
  cats[6].add("gentle intimacy, not posed");
  cats[6].add("everyday errand, candid");
  cats[6].add("subtle story-telling, natural");
  cats[6].add("restrained, competent amateur look");

  // initialize indices
  for (int i = 0; i < CAT_COUNT; i++) indices[i] = 0;
}

void draw() {
  background(250);
  fill(30);
  textSize(18);
  text("Secondary Style Generator", leftMargin, topMargin - 2);
  textSize(12);
  text("Generate restrained, realistic style strings for SDXL (amateur, natural look).", leftMargin, topMargin + 18);

  // Draw category boxes
  for (int i = 0; i < CAT_COUNT; i++) {
    int y = topMargin + 40 + i * catHeight;
    drawCatBox(i, leftMargin, y, catWidth, catHeight - 8);
  }

  // Right panel: controls and generated list
  drawRightPanel();
}

void drawCatBox(int idx, int x, int y, int w, int h) {
  stroke(200);
  fill(255);
  rect(x, y, w, h, 6);
  fill(0);
  textSize(12);
  text(catNames[idx], x + 8, y + 16);
  textSize(11);
  String current = cats[idx].get(indices[idx]);
  text(current, x + 8, y + 34, w - 140);
  // Cycle button
  int bx = x + w - buttonW - 12;
  int by = y + 12;
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

  // Show current assembled secondary prompt preview
  textSize(12);
  text("Preview Secondary Prompt", x + 12, by + buttonH + 28);
  String preview = assembleSecondaryPrompt(false);
  textSize(11);
  textLeading(14);
  text(preview, x + 12, by + buttonH + 44, w - 24, 140);

  // Generated prompts list
  textSize(12);
  text("Generated Secondary Prompts", x + 12, by + buttonH + 200);
  textSize(11);
  int listY = by + buttonH + 220;
  for (int i = 0; i < generated.size(); i++) {
    text((i+1) + ". " + generated.get(i), x + 12, listY + i * 18);
  }

  // Tips
  textSize(10);
  fill(80);
  text("Tip: Pair one generated Secondary string with a Primary prompt. Keep negatives concise.", x + 12, height - 28);
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
  // Check category cycle buttons
  for (int i = 0; i < CAT_COUNT; i++) {
    int y = topMargin + 40 + i * catHeight;
    int bx = leftMargin + catWidth - buttonW - 12;
    int by = y + 12;
    if (mouseX >= bx && mouseX <= bx + buttonW && mouseY >= by && mouseY <= by + buttonH) {
      indices[i] = (indices[i] + 1) % cats[i].size();
      return;
    }
  }

  // Right panel buttons
  int x = rightPanelX + 12;
  int by = topMargin + 40 + 52;
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

String assembleSecondaryPrompt(boolean includeLabel) {
  StringBuilder sb = new StringBuilder();
  if (includeLabel) sb.append("Secondary: ");
  // Order: Lighting | Camera | Lens | Color | DOF | Texture | Mood
  sb.append(cats[0].get(indices[0]));
  sb.append(", ");
  sb.append(cats[1].get(indices[1]));
  sb.append(", ");
  sb.append(cats[2].get(indices[2]));
  sb.append(", ");
  sb.append(cats[3].get(indices[3]));
  sb.append(", ");
  sb.append(cats[4].get(indices[4]));
  sb.append(", ");
  sb.append(cats[5].get(indices[5]));
  sb.append(", ");
  sb.append(cats[6].get(indices[6]));
  return sb.toString();
}

void generatePrompts(int n) {
  for (int k = 0; k < n; k++) {
    StringBuilder sb = new StringBuilder();
    sb.append("Secondary: ");
    for (int i = 0; i < CAT_COUNT; i++) {
      int r = (int) random(cats[i].size());
      sb.append(cats[i].get(r));
      if (i < CAT_COUNT - 1) sb.append(", ");
    }
    generated.add(sb.toString());
  }
}

void saveGeneratedPrompts() {
  if (generated.size() == 0) return;
  String[] out = new String[generated.size()];
  for (int i = 0; i < generated.size(); i++) out[i] = generated.get(i);
  saveStrings("generated_secondary_prompts.txt", out);
}

// Keyboard shortcuts
void keyPressed() {
  if (key == '+') generateCount++;
  if (key == '-' && generateCount > 1) generateCount--;
  if (key == 'g' || key == 'G') generatePrompts(generateCount);
  if (key == 'c' || key == 'C') generated.clear();
}
