@echo off
echo Starting Video Optical Flow Visualizer...
cd /d "C:\MyPythonCoding\MyCoding\image_processors\video_optical_flow_openCV"
call "..\..\.venv\Scripts\activate.bat"
python optical_flow_visualizer.py
pause
