# project-skills-hooks — example per-project skills & hooks

Demonstrates the manifest-free `.flowai/skills`/`.flowai/hooks`
mechanism — distinct from `examples/plugins/hello-world`, which
demonstrates the global, manifest-based **plugin** format instead. Use
this shape when the extension is only useful for ONE specific project
you're working on, not something you'd share as a standalone plugin.

- **`.flowai/skills/todo.py`** — a `/todo <text>` command, scoped to
  whichever project has this `.flowai/` folder. The filename (minus
  `.py`) IS the command name — no manifest entry needed.
- **`.flowai/hooks/no_secrets.py`** — `post_file_edit` (warns) and
  `pre_commit` (blocks) hooks, a crude grep for accidentally-committed
  secrets. Any `.py` file in `.flowai/hooks/` contributes whichever of
  `post_file_edit`/`pre_commit` it defines.

## Try it

Copy the `.flowai` folder into the root of the project you have open in
flowai (not into flowAI's own checkout):

```bash
cp -r examples/project-skills-hooks/.flowai /path/to/your/project/.flowai
```

Restart flowai in that project. `/todo write more tests` appends a line
to `TODO.local.md`; editing a file that contains something matching
`api_key = "..."` prints a warning; committing one is blocked.

## Uninstall

Delete `.flowai/skills/todo.py` / `.flowai/hooks/no_secrets.py` (or the
whole `.flowai/` folder) — no registry, nothing else to clean up.

See [docs/plugins.md](../../docs/plugins.md) for the full mechanism,
including how a project skill relates to a same-named global plugin
command.
