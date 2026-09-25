---
name: release-notes
description: Draft release notes for the changes since the last git tag. Use when the user asks for release notes, a changelog entry or "what changed since the last release".
allowed-tools: bash, read_file, grep_search, glob_search, git_log, git_diff
---
Draft release notes for this repository.

1. Find the latest tag: `git describe --tags --abbrev=0` (if there is no tag, use the whole history).
2. List commits since it: `git log <tag>..HEAD --oneline --no-merges`.
3. Group them into **Features**, **Fixes** and **Other**, one line per user-visible change —
   skip pure refactors, formatting and test-only commits.
4. Use the format from `template.md` next to this file.

Focus (optional, from the user): $ARGUMENTS
