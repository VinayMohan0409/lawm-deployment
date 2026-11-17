# Model files

Obtain the matching checkpoint bundle from the model project and place these
files here, or set their locations in your local YAML configuration:

- `best.pt`: model checkpoint.
- `run_manifest.json`: architecture/training configuration.
- `prior.npz`: latent-action prior matched to the checkpoint hash.

These files are local inputs, excluded from Git. They are not needed for
`--capture-goal` or `--standin` simulation. The reference checkpoint uses five
steps, four latent queries per step and eight dimensions per query. The real
input contains 1024 hand points and 5120 scene points.
