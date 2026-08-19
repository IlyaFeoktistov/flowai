# hello-world — example flowAI plugin

Demonstrates all three things a flowAI plugin can provide, in the
smallest form each takes:

- **`/hello [name]`** — a slash command (`hello.py`)
- **`hello_world_echo`** — an MCP tool the model can call (`server.py`)
- **`post_file_edit`/`pre_commit`** — hooks (`hooks.py`)

## Install

Plugins live in `plugins/` at the root of the flowAI checkout (tracked in
git as an empty, always-present directory — see its `.gitkeep` — but
anything you put inside is git-ignored). Copy this whole folder there:

```bash
cp -r examples/plugins/hello-world plugins/hello-world
```

Restart flowai — plugins load once at startup. Run `/plugin` to confirm
it was picked up.

## Uninstall / disable

Delete the folder to remove it, or drop an empty `.disabled` file inside
it to turn it off without deleting anything:

```bash
touch plugins/hello-world/.disabled
```

## Manifest format

See `src/mcp_agent/plugins.py`'s module docstring for the full
`plugin.json` schema and exactly what each hook receives — and for the
simpler, manifest-free `.flowai/skills/`/`.flowai/hooks/` convention if
what you want is a one-off extension for a single project rather than a
shareable plugin.
