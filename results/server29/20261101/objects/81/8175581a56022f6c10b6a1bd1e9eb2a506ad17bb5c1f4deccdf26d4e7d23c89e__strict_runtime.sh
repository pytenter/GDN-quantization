#!/bin/sh
set -eu
exec /bin/sh "$(CDPATH= cd -- "$(dirname -- "$0")/../parent_cpu/runtime_presets" && pwd)/strict_runtime.sh" "$@"
