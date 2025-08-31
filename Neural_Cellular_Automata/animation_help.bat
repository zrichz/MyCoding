@echo off
echo NCA Animation Generator - Help
echo =============================
echo.
echo This tool creates animated GIFs from trained NCA models showing
echo the step-by-step evolution process. Perfect for social media!
echo.
echo GUI VERSION (Recommended):
echo   launch_animator.bat
echo.
echo COMMAND LINE VERSION:
echo   python nca_quick_animator.py model.pth output.gif [options]
echo.
echo EXAMPLES:
echo   python nca_quick_animator.py my_model.pth evolution.gif
echo   python nca_quick_animator.py my_model.pth spiral.gif --init circle --steps 200
echo.
echo INITIALIZATION TYPES:
echo   center       - Single pixel at center (classic)
echo   random_single - One random point
echo   random_multi - Multiple random points  
echo   sparse       - Scattered pixels
echo   edge         - Starting from edge
echo   circle       - Small circle seed
echo.
echo TIPS FOR SOCIAL MEDIA:
echo   - Use 256px size for smaller files
echo   - 120-150 steps for good balance
echo   - Frame interval 2-3 for smooth motion
echo   - Enable step labels to show progression
echo.
echo First, train a model with NCA_baseline.py, then use these tools!
echo.
pause
