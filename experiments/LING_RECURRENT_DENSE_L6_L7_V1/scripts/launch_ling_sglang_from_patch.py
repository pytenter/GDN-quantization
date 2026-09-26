#!/usr/bin/env python3
"""Install the explicitly selected KDA patch, then start pinned SGLang."""

import importlib.util
import os
import runpy


path = os.environ["LING_KDA_PATCH_PATH"]
spec = importlib.util.spec_from_file_location("ling_selected_kda_patch", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.install_patch()

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
