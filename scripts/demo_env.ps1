# Dot-source this file to point the current PowerShell session at the
# demonstration knowledge base:
#
#     . .\scripts\demo_env.ps1
#
# It changes nothing on disk. It only sets environment variables for this
# terminal, so your normal knowledge base (corpus\ and data\corpus\) is
# left alone. Close the terminal, or run  . .\scripts\demo_env.ps1 -Off ,
# to go back.
param([switch]$Off)

$vars = @{
    PYTHONPATH                 = "src"
    PYTHONIOENCODING           = "utf-8"
    RASVCX_SEED_DIR            = "corpus_demo"                 # demo seed index
    RASVCX_CORPUS_DIR          = "data/demo_kb"                # versions published during the demo
    RASVCX_ACTIVE_VERSION_PATH = "data/demo_kb/active_version.json"
    RASVCX_INGEST_DB_PATH      = "data/demo_kb/ingest_jobs.sqlite"
    RASVCX_QDRANT_MODE         = "embedded"                    # dense index persisted on disk:
    RASVCX_QDRANT_PATH         = "data/demo_qdrant"            # embedded once, reused on restart
    RASVCX_EVAL_OUTPUT_DIR     = "eval_output"
    HF_HUB_OFFLINE             = "1"                           # models are cached: no Hub calls at startup
}

foreach ($name in $vars.Keys) {
    if ($Off) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
    else { Set-Item "Env:$name" $vars[$name] }
}
if ($Off) { "RASVC-X demo environment cleared." }
else { "RASVC-X demo environment set (seed=corpus_demo, kb=data/demo_kb, qdrant=embedded)." }
