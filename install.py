import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirements for flowvid extension")
