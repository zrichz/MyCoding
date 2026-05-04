# MyCoding Workspace Instructions

## Python Environment

**Always use the venvmycoding313 virtual environment for this workspace.**

- Virtual environment path: `/home/rich/MyCoding/venvmycoding313`
- Python executable: `/home/rich/MyCoding/venvmycoding313/bin/python`
- Activate command: `source /home/rich/MyCoding/activate_venv.sh` or `source /home/rich/MyCoding/venvmycoding313/bin/activate`
- If `which python` shows wrong version after activation, run: `hash -r`

When running Python scripts or installing packages in this workspace, always use this virtual environment.

**Note:** The venv uses Python 3.13. Installed packages: numpy, scipy, gradio.

## Gradio Applications

When creating interactive applications or GUIs for this workspace:

- **Prefer Gradio** for building interfaces where possible
- **Always set `inbrowser=True`** in `demo.launch()` to automatically open the browser
- Example: `demo.launch(inbrowser=True)`

do NOT use icons, such as in the following example: 
          'message = (f"✅ Successfully embedded {width_a}×{height_a} image A into image B\n"
              f"📍 Used {len(block_positions)} randomly distributed 3×3 blocks\n"
              f"🔒 Seed: '{seed}'\n"
              f"📊 Max pixel change: ±{max_change} (LSB encoding)\n"
              f"💾 Image B size preserved: {width_b}×{height_b}")'

instead, use plain text for messages in Gradio interfaces to ensure compatibility and avoid rendering issues. Do NOT use Exclamation marks in any text messages in code or gradio interface.

## Workspace Preferences

This workspace contains various Python projects for image processing, machine learning, generative art, and AI tools. Refer to any additional preferences in `PROJECT_PREFERENCES.md` at the workspace root.
