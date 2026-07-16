#!/usr/bin/env python3
from pathlib import Path
import json
import trackio

ROOT = Path(__file__).resolve().parents[1]
trackio.init(project="adam-degenerate-polynomials-repro", name="cpu-two-claim-reproduction",
             config={"openreview_id": "uYWVGk1Qt0", "claims": 2, "device": "cpu", "gpu_used": False},
             embed=False, auto_log_gpu=False, auto_log_cpu=False)
artifact = trackio.Artifact("adam-degenerate-polynomials-cpu-reproduction", type="dataset",
                            description="Arbitrary-precision Adam, million-step GD/momentum, RMSProp ablation, raw outputs, tests, and provenance.")
artifact.add_dir(ROOT / "reproduction", name="reproduction")
artifact.add_dir(ROOT / "outputs", name="outputs")
for name in ("paper.pdf", "claims.md", "SOURCE_AUDIT.md", "ENVIRONMENT.md", "README.md"):
    artifact.add_file(ROOT / name, name=name)
logged = trackio.log_artifact(artifact, aliases=["challenge", "cpu", "complete"])
trackio.finish()
print(json.dumps({"artifact": logged.qualified_name, "files": len(logged.manifest or []), "size": logged.size}, sort_keys=True))

