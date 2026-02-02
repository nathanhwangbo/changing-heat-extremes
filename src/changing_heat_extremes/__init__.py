import os
import sys
from pathlib import Path

#######################################################
# If PROJ_DATA is missing from the environment, add it.
#     this is needed for cartopy/regionmask/ect.
########################################################
if "PROJ_DATA" not in os.environ:
    # env_root is .pixi/envs/default
    # proj_path is .pixi/envs/default/share/proj

    # env_root is two levels from the python executable (bin -> env root)
    env_root = Path(sys.executable).parent.parent
    proj_path = env_root / "share" / "proj"

    if proj_path.exists():
        os.environ["PROJ_DATA"] = str(proj_path)
        os.environ["PROJ_LIB"] = str(proj_path)
        print(f"Manually added environment variable PROJ_DATA: {proj_path}")

###################################################################
# if firefox/geckodriver are missing from the path, update the path
#      this is needed to export hvplot figures in png format
###################################################################
current_env_bin = Path(sys.executable).parent  # where firefox/geckodriver live.
if str(current_env_bin) not in os.environ["PATH"]:
    os.environ["PATH"] = str(current_env_bin) + os.pathsep + os.environ["PATH"]
    print(f"Added {current_env_bin} to PATH for Bokeh export.")

    # also make sure selenium manager path can be detected
    os.environ["SE_MANAGER_PATH"] = str(current_env_bin) + os.sep + "selenium-manager"
    print("Manualy added env variable SE_MANAGER_PATH")
