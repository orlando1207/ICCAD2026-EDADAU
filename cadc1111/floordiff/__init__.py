"""FloorDiff: conditional diffusion model for FloorSet-Lite floorplan prediction.

Stage-1 (prediction-only) implementation of
docs/superpowers/specs/2026-07-14-diffusion-repo-anatomy-and-contest-model-design.md:
imitation-first — the model predicts (cx, cy, s=log-aspect) per block, conditioned on
constraint-type features; no constraint enforcement (deferred to legalization).
"""
