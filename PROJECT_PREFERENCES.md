# Project Preferences & Configuration

## Python Environment
- **Virtual Environment**: `/home/rich/MyVenvs/myVenv312` (Python 3.12.3)
- **Activation**: `source /home/rich/MyVenvs/myVenv312/bin/activate`
- **Python Executable**: `/home/rich/MyVenvs/myVenv312/bin/python`
- **Installed Packages**: gradio, numpy, plotly, opencv-python, pillow, etc.

## Default Settings for Gradio Apps
- Always set `inbrowser=True` in `demo.launch()` to auto-open browser
- Use `theme=gr.themes.Soft()` for consistent UI

## Shell Script Template
```bash
#!/bin/bash
echo "Starting [App Name]..."
source /home/rich/MyVenvs/myVenv312/bin/activate
python [script_name].py
```

## Notes
- This workspace contains various creative coding projects
- Focus on image processing, machine learning, and visualization
- Gradio is preferred for UI interfaces
- do not use icons within text fields in GUI, do not use exclamation marks for text, eg: "Finished!" - use "Finished." instead
