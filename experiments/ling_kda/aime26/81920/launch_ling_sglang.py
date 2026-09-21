#!/usr/bin/env python3
"""Install AIME26 KDA hooks, then enter the pinned SGLang launch module."""

import runpy

from sglang_kda_runtime_patch import install_patch


install_patch()


if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
