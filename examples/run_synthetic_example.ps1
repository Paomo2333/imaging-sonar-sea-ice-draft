$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$generated = Join-Path $PSScriptRoot "generated"

python (Join-Path $PSScriptRoot "generate_synthetic_inputs.py") --output-dir $generated
python (Join-Path $repo "run_from_config.py") --config (Join-Path $repo "config\example_run.json")
