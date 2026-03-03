# Project Preferences & Configuration

## Python Environment
- **Virtual Environment**: `newvenv2026`
- **Activation**: Use `call newvenv2026\Scripts\activate` in batch files
- **Installed Packages**: gradio, numpy, plotly, opencv-python, pillow, etc.

## Default Settings for Gradio Apps
- Always set `inbrowser=True` in `demo.launch()` to auto-open browser
- Use `theme=gr.themes.Soft()` for consistent UI

## Batch File Template
```batch
@echo off
echo Starting [App Name]...
call newvenv2026\Scripts\activate
python [script_name].py
pause
```

## Notes
- This workspace contains various creative coding projects
- Focus on image processing, machine learning, and visualization
- Gradio is preferred for UI interfaces
