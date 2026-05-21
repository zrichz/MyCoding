# Project Preferences & Configuration

## Python Environment
- **Virtual Environment**: `/home/rich/MyCoding/venvmycoding313` (Python 3.13.12)
- **Activation**: `source /home/rich/MyCoding/venvmycoding313/bin/activate`
- **Python Executable**: `/home/rich/MyCoding/venvmycoding313/bin/python`
- **Installed Packages**: gradio, numpy, plotly, opencv-python, pillow, etc.

## Default Settings for Gradio Apps
- Always set `inbrowser=True` in `demo.launch()` to auto-open browser
- Use `theme=gr.themes.Soft()` for consistent UI
- if you decide to test a gradio app by running it, use a timeout of 10seconds, not 5. Only test a gradio app after significant changes. Small changes do not require a full test - the user will do that himself.

## Shell Script Template
```bash
#!/bin/bash
echo "Starting [App Name]..."
source /home/rich/MyCoding/venvmycoding313/bin/activate
python [script_name].py
```

## Notes
- This workspace contains various creative coding projects
- Focus on image processing, machine learning, and visualization
- Gradio is preferred for UI interfaces
- do not use icons within text fields in GUI, do not use exclamation marks for text, eg: "Finished!" - use "Finished." instead
