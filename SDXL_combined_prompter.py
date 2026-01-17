"""
SDXL Combined Prompt Generator
Python/Gradio version - generates 200 complete SDXL prompts with Primary and Secondary stages.
Format: <primary> | <secondary>

Author: Copilot
Date: 2026-01-09
"""

import gradio as gr
import random
from datetime import datetime

# PRIMARY STAGES (9)
PRIMARY_STAGES = {
    "Subject identity": [
        "personxyz",
        
    ],
    "Pose and action": [
        "three-quarter turn, left arm raised",
        "standing, hands in pockets",
        "seated, one knee up",
        "walking toward camera",
        "sitting on a low wall, legs crossed",
        "leaning on a railing, looking down",
        "holding a mug with both hands",
        "adjusting jacket collar casually",
        "reading a paperback, head tilted",
        "tying shoelace, looking away",
        "brushing hair back with one hand",
        "standing with arms folded loosely",
        "sitting on stairs, elbows on knees",
        "walking dog on a short lead",
        "holding a takeaway coffee, mid-sip",
        "checking phone while standing",
        "leaning against doorframe, relaxed",
        "sitting at table, hands clasped",
        "looking out of a window, thoughtful",
        "standing, slight smile",
        "carrying a backpack, mid-step",
        "sitting on a bench, one arm draped",
        "holding bicycle by the handlebars",
        "walking up a short flight of steps",
        "adjusting glasses, slight tilt of head",
        "reaching for a shelf, casual stance",
        "sitting cross-legged on the floor",
        "leaning over a kitchen counter",
        "standing with one foot on a low step",
        "holding a camera at chest height",
        "sitting in a cafe booth, relaxed",
        "standing by a window, hands in pockets",
        "walking past a shopfront, glancing",
        "sitting on a windowsill, knees drawn up",
        "holding a newspaper, reading",
        "standing with coat draped over shoulder",
        "leaning on a bicycle, casual",
        "sitting on a low wall, feet dangling",
        "mid-laugh, head thrown back slightly",
    ],
    "Framing and crop": [
        "waist-up",
        "head and shoulders",
        "full body",
        "three-quarter body",
        "close-up face",
        "knee-up",
        "hip-up",
        "environmental portrait, subject small in frame",
        "tight portrait, eyes centered",
        "half-body, slight tilt",
        "over-the-shoulder crop",
        "waist-up, slight left offset",
        "head-to-toe, centered",
        "upper torso, three-quarter turn",
        "close crop on hands and face",
        "mid-shot with foreground blur",
        "portrait orientation, headroom",
        "landscape orientation, subject left",
        "tight headshot with soft bokeh",
        "full body with negative space above",
        "waist-up, slight downward angle",
        "head and shoulders, eye-level",
        "three-quarter body, slight wide angle",
        "close-up of profile",
        "mid-shot with environmental context",
        "candid half-body crop",
        "full body, slight motion blur",
        "tight portrait, off-center composition",
        "head and shoulders, soft framing",
        "waist-up, natural posture",
        "full body, slight low angle",
        "close-up with hands visible",
        "three-quarter body, centered",
        "mid-shot with foreground element",
        "tight crop on face and shoulders",
        "full body, environmental detail visible",
        "head and shoulders, slight side lighting",
        "waist-up, casual stance",
        "close-up with soft edge vignette",
    ],
    "Clothing and key props": [
        "cream linen shirt",
        "blue leather jacket",
        "striped sweater",
        "tailored coat and scarf",
        "denim jacket and white tee",
        "wool jumper and jeans",
        "floral minidress, simple cut",
        "casual hoodie and trainers",
        "raincoat and wellies",
        "t-shirt and shorts",
        "checked shirt and denim",
        "knit sweater and skirt",
        "polo shirt and jeans",
        "yoga top and leggings",
        "bikini and sarong",
        "rave outfit with glowsticks",
        "oversized knit and leggings",
        "simple blouse and mid-length skirt",
        "sweatshirt and joggers",
        "cosplay Soul Calibur outfit",
        "casual blazer and tee",
        "striped tee and denim shorts",
        "light rain jacket and umbrella",
        "wool coat and scarf",
        "evening dress and shawl",
        "checked scarf and beanie",
        "summer dress and sandals",
        "puffer jacket and jeans",
        "uniform style blazer",
        "cycling jacket and helmet",
        "sweater vest and shirt",
        "casual shirt and backpack",
        "striped jumper and coat",
        "athleisure wear",
        "simple tee and denim jacket",
        "fitted coat and scarf",
        "clubbing outfit",
        "wool hat and gloves",
        "silk shirt and trousers",
        "light gilet and jeans",
        "casual shirt and watch",
        "pyjama top and slippers",
    ],
    "Expression and gaze": [
        "soft smile, looking at camera",
        "neutral expression, direct gaze to viewer",
        "candid smile, eyes to camera",
        "contemplative, looking at viewer",
        "gentle laugh, eyes to camera",
        "focused, looking at viewer",
        "relaxed, eyes to camera",
        "slight grin, looking at camera",
        "thoughtful, direct gaze",
        "subtle smile, direct eye contact",
        "soft expression, looking at viewer",
        "calm, eyes to camera",
        "mild amusement, looking at camera",
        "serene, slight smile to viewer",
        "warm smile, eyes crinkled at camera",
        "reserved smile, eyes to viewer",
        "gentle smirk, looking at camera",
        "open smile, eyes to camera",
        "softly serious, direct gaze at viewer",
        "subdued smile, eyes to camera",
        "mild surprise, looking at viewer",
        "relaxed grin, eyes to camera",
        "quiet contentment, looking at viewer",
        "slight frown, eyes to camera",
        "gentle curiosity, looking at viewer",
        "soft laugh, eyes to camera",
        "calm, steady gaze at viewer",
        "subtle amusement, looking at camera",
        "warm, approachable smile to viewer",
        "reflective, eyes to camera",
        "mildly inquisitive, direct gaze at viewer",
        "soft grin, eyes to camera",
        "content expression, looking at viewer",
        "quiet smile, eyes to camera",
        "gentle amusement, looking at viewer",
        "softly bemused, eyes to camera",
        "calm, neutral expression looking at viewer",
    ],
    "Body descriptors": [
        "visible freckles on arms",
        "light sun tan",
        "pale complexion",
        "strong posture",
        "relaxed shoulders",
        "toned calves",
        "natural posture, relaxed",
        "slight asymmetry in stance",
    ],
    "Composition anchors": [
        "centered, negative space to the right",
        "leaning against wall, left side of frame",
        "slight head tilt, off-center composition",
        "foreground subject, blurred background",
        "subject slightly left, leading lines to right",
        "tight framing with window light behind",
        "subject near bottom third, sky visible",
        "balanced with props on either side",
        "subject framed by doorway",
        "subject in lower-left, negative space above",
        "diagonal composition, subject moving right",
        "symmetrical composition, centered subject",
        "subject against textured wall",
        "soft foreground element partially obscuring",
        "subject offset to create breathing room",
        "tight crop with hands visible",
        "subject framed by bookshelf",
        "leading lines from foreground to subject",
        "subject leaning into frame from right",
        "low-angle composition, subject dominant",
        "high-angle, subject small in frame",
        "subject centered with shallow depth",
        "subject placed on left third, open space right",
        "subject framed by archway",
        "soft vignette, subject centered",
        "subject partially behind foreground object",
        "balanced negative space above head",
        "subject aligned with vertical lines",
        "subject in foreground, street in background",
        "subject leaning into negative space",
        "tight portrait with environmental hint",
        "subject slightly off-center, natural pose",
        "subject framed by window light",
        "subject in lower third, sky and buildings above",
        "subject against plain backdrop, natural pose",
        "subject interacting with prop in frame",
        "subject centered with subtle motion blur",
        "subject placed near leading architectural lines",
    ],
    "Context or location": [
        "studio portrait",
        "outdoor urban alley",
        "window-lit interior",
        "bathroom mirror selfie",
        "kitchen counter in morning light",
        "local high street, daytime",
        "park bench near river",
        "train station platform",
        "coastal promenade, sunny",
        "cafe table by the window",
        "living room with bookshelf",
        "garden patio with potted plants",
        "commuter street, early morning",
        "market stall area, casual crowd",
        "bookshop aisle, warm light",
        "bus stop on a rainy day",
        "suburban front garden",
        "country lane with hedgerows",
        "ferry terminal, coastal travel",
        "small-town high street, late afternoon",
        "university quad, autumn leaves",
        "local pub beer garden",
        "train carriage window seat",
        "city square with pigeons",
        "farmers market on a Saturday",
        "canal towpath, morning mist",
        "railway bridge, industrial backdrop",
        "seaside pier, muted light",
        "village green with benches",
        "cozy bookshop corner",
        "weekday office kitchen",
        "local bakery storefront",
        "bus interior, natural light",
        "small coastal town street",
        "city rooftop with distant skyline",
        "suburban high street cafe",
        "country pub interior",
        "holiday cottage kitchen",
        "train station concourse, travel vibe",
        "home office with desk lamp",
        "laundromat on a quiet evening",
        "hotel lobby with soft lighting",
        "gym reception area",
        "dental office waiting room",
        "supermarket checkout lane",
        "library reading room, afternoon",
        "hair salon styling station",
        "conference room with glass walls",
        "warehouse break room",
    ],
    "Semantic technical anchors": [
        "mirror selfie",
        "tripod shot",
        "studio lighting setup",
        "candid handheld shot",
        "phone camera at chest height",
        "window-lit natural portrait",
        "overhead kitchen light",
        "golden hour outdoor shot",
        "soft overcast daylight",
        "indoor tungsten lamp",
        "phone selfie with arm extended",
        "camera on table, timer shot",
        "handheld at waist level",
        "shot from slightly above eye level",
        "shot from slightly below eye level",
        "window backlight with reflector",
        "natural window light from left",
        "natural window light from right",
        "softbox-like diffused light",
        "ambient cafe lighting",
        "streetlight evening shot",
        "shopfront window reflection",
        "car interior shot, passenger seat",
        "train window light, motion hint",
        "umbrella overhead, rainy day",
        "doorway light, subject half-lit",
        "soft fill from reflector",
        "natural shade under tree",
        "soft sidelighting from lamp",
        "ambient market stall lighting",
        "overcast diffuse sky, even light",
        "warm kitchen morning light",
        "cool evening window light",
        "handheld phone with slight motion",
        "camera on tripod, slight depth",
        "shot through glass, subtle reflection",
        "soft backlight with rim highlight",
        "natural light with subtle shadow",
    ],
}

# SECONDARY CATEGORIES (7)
SECONDARY_CATEGORIES = {
    "Lighting": [
        "soft natural window light from left",
        "soft natural window light from right",
        "diffused overcast daylight",
        "warm golden hour side light",
        "cool early morning light",
        "soft backlight with subtle rim",
        "ambient indoor daylight, even",
        "soft kitchen morning light",
        "muted late afternoon light",
        "soft shade under tree",
        "softbox-like diffused lamp",
        "practical lamp, warm tone",
        "window light with gentle reflector fill",
        "soft sidelighting from lamp",
        "overhead soft ambient light",
        "soft window backlight with fill",
        "streetlight evening, subtle warmth",
        "shopfront window light, muted",
        "soft cloudy coastal light",
        "indoor tungsten with neutral balance",
        "soft directional light through blinds",
        "soft diffuse light from north-facing window",
        "gentle reflector fill, natural look",
        "soft golden rim light",
        "even daylight with slight shadow",
        "soft window light, left side, low contrast",
        "muted overcast backlight",
        "soft cafe ambient light",
        "soft porch light at dusk",
        "harsh daylight through thin curtain",
        "soft natural light, slight warmth",
        "harsh directional light, high contrast",
        "soft evening window light, cool tone",
        "harsh lamp light with shadows",
        "soft daylight with gentle highlights",
        "soft natural light, even skin tones",
        "soft backlight with subtle lens flare",
        "soft ambient market stall light",
        "train window light",
    ],
    "Camera / Perspective": [
        "eye-level perspective",
        "slightly above eye-level",
        "slightly below eye-level",
        "three-quarter angle, natural",
        "straight-on, relaxed",
        "slight downward tilt",
        "slight upward tilt",
        "environmental portrait perspective",
        "intimate close perspective",
        "medium distance, natural",
        "wide environmental perspective",
        "tight headshot perspective",
        "over-the-shoulder viewpoint",
        "candid handheld viewpoint",
        "phone-chest-height viewpoint",
        "mirror selfie perspective",
        "tripod-stable eye-level",
        "slight motion perspective, natural",
        "table-top timer perspective",
        "window-seat perspective",
        "bench-side perspective",
        "doorway-framed perspective",
        "street-level perspective",
        "car-interior passenger perspective",
        "train-window perspective",
        "low-angle, modest dominance",
        "high-angle, modest vulnerability",
        "three-quarter environmental view",
        "tight profile perspective",
        "softly off-center perspective",
        "balanced centered perspective",
        "slight wide-angle environmental",
        "natural handheld framing",
        "softly cropped portrait perspective",
        "mid-distance candid framing",
        "soft foreground framing viewpoint",
        "slight tilt for casual feel",
        "eye-level with slight headroom",
        "three-quarter with negative space",
    ],
    "Lens / Focal length": [
        "50mm standard lens look",
        "35mm environmental portrait",
        "85mm short telephoto portrait",
        "24mm slight wide environmental",
        "70mm short telephoto feel",
        "28mm modest wide angle",
        "100mm short telephoto tight portrait",
        "40mm natural field of view",
        "60mm gentle compression",
        "35mm with natural context",
        "85mm with soft compression",
        "50mm with slight bokeh",
        "24-70mm versatile zoom feel",
        "35mm slightly intimate",
        "50mm close portrait",
        "85mm head-and-shoulders",
        "28mm for modest environmental hint",
        "35mm for casual street feel",
        "50mm for natural skin rendering",
        "85mm for flattering compression",
    ],
    "Color grading / Film style": [
        "neutral color balance, natural skin tones",
        "slightly warm, low saturation",
        "muted tones, low contrast",
        "soft film-like color, subtle grain",
        "cool tones, natural look",
        "soft teal and warm highlights, restrained",
        "gentle Kodak-like warmth, subtle",
        "subtle Portra-inspired warmth",
        "faded film look, low contrast",
        "clean digital look, minimal processing",
        "slightly desaturated, natural",
        "soft pastel highlights, restrained",
        "warm indoor tungsten balance",
        "cool overcast grading, neutral skin",
        "soft contrast, natural shadows",
        "gentle contrast boost, realistic",
        "slight vintage fade, subtle",
        "natural color with slight warmth",
        "soft film grain and neutral color",
        "low-key natural color, realistic",
        "muted autumnal palette",
        "soft morning warmth, low saturation",
        "neutral with slight highlight roll-off",
        "clean daylight balance, realistic",
        "soft contrast, warm midtones",
        "slight cross-processed feel, subtle",
        "gentle matte finish, natural",
        "soft warm highlights, neutral shadows",
        "cool evening tones, restrained",
        "soft filmic warmth, low vibrance",
        "natural color, slight clarity",
        "soft pastel wash, subtle",
        "neutral with slight vignette",
        "soft cinematic teal-orange, very subtle",
        "muted color with natural skin",
        "soft warm kitchen tones",
        "clean neutral with slight warmth",
        "soft low-contrast film look",
    ],
    "Depth of field / Bokeh": [
        "shallow depth of field, soft bokeh",
        "moderate depth, background readable",
        "deep focus, environmental detail",
        "soft background blur, natural",
        "slight background separation",
        "soft foreground blur, subject sharp",
        "gentle bokeh with circular highlights",
        "soft bokeh, low contrast background",
        "moderate DOF, subject isolated",
        "shallow DOF, eyes sharply focused",
        "soft bokeh with subtle chromatic fringing",
        "shallow DOF, subtle rim separation",
        "shallow DOF, slight motion blur in background",
        
    ],
    "Texture / Finish": [
        "subtle film grain",
        "clean digital finish",
        "very light film grain, natural",
        "soft clarity, minimal sharpening",
        "gentle texture, realistic skin",
        "matte finish, low contrast",
        "slight clarity boost, natural",
        "soft micro-contrast, realistic",
        "minimal noise reduction, natural",
        "light film grain and subtle texture",
        "clean skin rendering, low retouch",
        "natural skin texture preserved",
        "softened highlights, natural detail",
        "subtle sharpening on eyes",
        "gentle clarity on facial features",
        "minimal post-processing look",
        "soft matte skin finish",
        "natural pores visible, realistic",
        "slight vignette, natural",
        "low-key finish, realistic texture",
    ],
    "Mood / Subtle effects": [
        "quiet, candid mood",
        "everyday, unposed feel",
        "natural, documentary tone",
        "calm, approachable atmosphere",
        "subtle warmth, homely",
        "cheerful, candid",
        "understated, authentic",
        "modest travel vibe, realistic",
        "weekday morning routine feel",
        "casual weekend mood",
        "quiet domestic scene",
        "subtle motion hint, natural",
        "low-key documentary feel",
        "gentle intimacy, not posed",
        "everyday errand, candid",
        "subtle story-telling, natural",
        "restrained, competent amateur look",
    ],
}


def generate_prompts(primary_enabled, secondary_enabled):
    """Generate 200 combined prompts based on enabled stages."""
    prompts = []
    
    for _ in range(200):
        # Generate primary prompt
        primary_parts = []
        for stage_name, options in PRIMARY_STAGES.items():
            if primary_enabled.get(stage_name, True):
                primary_parts.append(random.choice(options))
        primary = "; ".join(primary_parts)
        
        # Generate secondary prompt
        secondary_parts = []
        for cat_name, options in SECONDARY_CATEGORIES.items():
            if secondary_enabled.get(cat_name, True):
                secondary_parts.append(random.choice(options))
        secondary = ", ".join(secondary_parts)
        
        # Combine in format: <primary> | <secondary>
        combined = f"{primary} | {secondary}"
        prompts.append(combined)
    
    return prompts


def generate_and_display(*checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (9 primary + 7 secondary = 16 total)
    primary_enabled = {}
    secondary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    secondary_names = list(SECONDARY_CATEGORIES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    for i, name in enumerate(secondary_names):
        secondary_enabled[name] = checkboxes[len(primary_names) + i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, secondary_enabled)
    
    # Format output - show only last 8 prompts with prefix and suffix
    last_8 = prompts[-8:]
    output_lines = [f'--prompt "{prompt}" --negative_prompt "asian, poor quality"' for prompt in last_8]
    output = "\n\n".join(output_lines)
    
    return output, prompts


def save_prompts(prompts_data):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sdxl_combined_prompts_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "asian, poor quality"\n')
    
    return f"✓ Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SDXL Combined Prompt Generator")
    gr.Markdown("Generate 200 randomized prompts in format: `<primary> | <secondary>`")
    gr.Markdown("Enable/disable stages to customize output. All stages enabled by default.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            primary_checks = []
            for stage_name in PRIMARY_STAGES.keys():
                primary_checks.append(gr.Checkbox(label=stage_name, value=True))
        
        with gr.Column(scale=1):
            gr.Markdown("### SECONDARY CATEGORIES")
            secondary_checks = []
            for cat_name in SECONDARY_CATEGORIES.keys():
                secondary_checks.append(gr.Checkbox(label=cat_name, value=True))
    
    generate_btn = gr.Button("Generate 200 Prompts", variant="primary", size="lg")
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    output_text = gr.Textbox(
        label="Generated Prompts (200 total)",
        lines=20,
        max_lines=30,
        interactive=False
    )
    
    # Hidden state to store prompts for saving
    prompts_state = gr.State([])
    
    # Wire up interactions
    all_checkboxes = primary_checks + secondary_checks
    generate_btn.click(
        fn=generate_and_display,
        inputs=all_checkboxes,
        outputs=[output_text, prompts_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
