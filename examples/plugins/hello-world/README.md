# hello-world — example flowAI plugin

Demonstrates all three things a flowAI plugin can provide, in the
smallest form each takes:

- **`/hello [name]`** — a slash command (`hello.py`)
- **`hello_world_echo`** — an MCP tool the model can call (`server.py`)
- **`post_file_edit`/`pre_commit`** — hooks (`hooks.py`)

## Install

Plugins live under flowAI's data directory, not inside the flowAI repo
itself. Copy this whole folder there:

```bash
cp -r examples/plugins/hello-world ~/.local/share/flowai/plugins/hello-world
```

(or `$XDG_DATA_HOME/flowai/plugins/` / `$FLOWAI_DATA_DIR/plugins/` if
either is set — see `storage.py`). Restart flowai — plugins load once at
startup. Run `/plugin` to confirm it was picked up.

## Uninstall / disable

Delete the folder to remove it, or drop an empty `.disabled` file inside
it to turn it off without deleting anything:

```bash
touch ~/.local/share/flowai/plugins/hello-world/.disabled
```

## Manifest format

See `mcp_agent/plugins.py`'s module docstring in the flowAI source for
the full `plugin.json` schema and exactly what each hook receives.
