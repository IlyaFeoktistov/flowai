"""Plan mode's bash check — a DENYLIST of commands that change something.

An allowlist of read-only commands can never be complete (ps, lsof,
nvidia-smi, docker ps, ... — every missing one made the model give up the
whole task in plan mode), and a false "denied" costs more here than a
missed write: with ask_permissions on, every bash call still goes through
the regular permission dialog, so the user sees the exact command first. So in plan mode
anything runs unless it's recognizably mutating.

Analyzer/Planner in the pipeline keep the stricter allowlist
(agent_builder._is_read_only_bash_command) — they run without the user
watching each step.

The command is split into simple commands on | || && ; & and ( ) (which
also opens up $(...) and subshells), and each one is judged by the program
in command position — `grep rm file` is fine, `xargs rm`, `sudo rm` and
`bash -c 'rm ...'` are not.
"""
import re
import shlex

_SEPARATORS = {"|", "||", "&&", ";", "&", "|&", "(", ")", "$(", ";;"}

# Always mutating, whatever the arguments.
_MUTATING_PROGRAMS = {
    "rm", "rmdir", "mv", "cp", "mkdir", "touch", "ln", "chmod", "chown", "chgrp",
    "truncate", "dd", "tee", "patch", "install", "shred", "unlink", "rsync", "scp",
    "kill", "pkill", "killall", "shutdown", "reboot", "poweroff", "halt",
    "mkfs", "mount", "umount", "crontab", "useradd", "userdel", "passwd",
    "make", "cmake", "ninja", "npx",
}

# Program -> subcommands that mutate (anything else of that program is fine,
# e.g. `npm ls`, `pip show`, `docker ps`, `systemctl status`).
_MUTATING_SUBCOMMANDS: dict[str, set[str]] = {
    "git": {
        "add", "commit", "push", "pull", "fetch", "merge", "rebase", "reset", "checkout",
        "switch", "restore", "stash", "cherry-pick", "revert", "rm", "mv", "clean", "apply",
        "am", "init", "clone", "gc", "prune", "update-ref", "update-index", "notes",
        "worktree", "submodule", "filter-branch", "replace", "bisect",
    },
    **{pm: {"install", "uninstall", "remove", "rm", "add", "update", "upgrade", "ci", "get",
            "publish", "link", "unlink", "init", "sync", "lock", "purge", "autoremove",
            "dist-upgrade", "mod", "build", "run", "exec", "tidy", "generate", "download",
            "reinstall", "dedupe", "prune", "audit"}
       for pm in ("pip", "pip3", "uv", "poetry", "npm", "yarn", "pnpm", "bun", "cargo", "go",
                  "brew", "apt", "apt-get", "dpkg", "snap", "gem", "composer", "conda")},
    **{d: {"rm", "rmi", "run", "stop", "kill", "start", "restart", "exec", "build", "push",
           "pull", "create", "prune", "tag", "cp", "commit", "load", "import", "up", "down",
           "pause", "unpause", "rename", "update"}
       for d in ("docker", "podman")},
    "systemctl": {"start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask",
                  "kill", "daemon-reload", "edit", "set-property"},
    "service": {"start", "stop", "restart", "reload"},
    "ollama": {"run", "pull", "push", "rm", "cp", "create", "stop"},
}
# `git branch -d x`, `git tag -d x`, `git config k v` — mutate only with these.
_GIT_FLAGGED = {"branch": {"-d", "-D", "-m", "-M", "-c", "-C", "-f", "--delete", "--move", "--force"},
                "tag": {"-d", "--delete", "-f", "--force", "-a", "-s"}}

# Prefixes that run the NEXT word as the real command.
_WRAPPERS = {"sudo", "doas", "env", "nohup", "time", "nice", "ionice", "timeout", "xargs",
             "command", "exec", "stdbuf", "watch", "strace", "chroot", "flock"}
_SHELLS = {"sh", "bash", "zsh", "dash", "fish"}

_IN_PLACE = re.compile(r"^(-i|--in-place)")


def _tokens(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes — bash will refuse it anyway
        return None


def _segments(tokens: list[str]) -> list[list[str]]:
    segments, current = [], []
    for tok in tokens:
        if tok in _SEPARATORS:
            if current:
                segments.append(current)
            current = []
        elif tok.startswith("$(") and len(tok) > 2:
            if current:
                segments.append(current)
            current = [tok[2:]]
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _writes_via_redirect(tokens: list[str]) -> bool:
    for i, tok in enumerate(tokens):
        if tok in (">", ">>", "&>", "&>>", ">|"):
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            if target != "/dev/null":
                return True
    return False


def _strip_wrappers(seg: list[str]) -> list[str]:
    while seg:
        prog = seg[0].rsplit("/", 1)[-1]
        if "=" in seg[0] and not seg[0].startswith("="):  # VAR=value cmd
            seg = seg[1:]
        elif prog in _WRAPPERS:
            seg = seg[1:]
            # the wrapper's own flags/arguments (sudo -u x, timeout 10, nice -n 5)
            while seg and (seg[0].startswith("-") or seg[0].replace(".", "").isdigit()):
                takes_value = prog in ("sudo", "doas") and seg[0] in ("-u", "-g", "-C", "-h", "-p")
                seg = seg[2:] if takes_value else seg[1:]
        else:
            return seg
    return seg


def _segment_mutates(seg: list[str]) -> bool:
    seg = _strip_wrappers(seg)
    if not seg:
        return False
    prog = seg[0].rsplit("/", 1)[-1]
    args = seg[1:]

    if prog in _SHELLS or prog == "eval":
        # bash -c '<script>' / eval '<script>' — judge the script itself.
        if prog == "eval":
            return is_mutating_bash(" ".join(args))
        if "-c" in args:
            i = args.index("-c")
            return i + 1 < len(args) and is_mutating_bash(args[i + 1])
        return any(not a.startswith("-") for a in args)  # bash script.sh — unknown script
    if prog in _MUTATING_PROGRAMS:
        return True
    if prog in ("sed", "perl", "ruby") and any(_IN_PLACE.match(a) for a in args):
        return True
    if prog in ("awk", "gawk") and "inplace" in args:
        return True
    if prog == "find":
        if any(a in ("-delete", "-fprint", "-fprintf", "-fls", "-fprint0") for a in args):
            return True
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            if flag in args:
                i = args.index(flag)
                if _segment_mutates(args[i + 1:]):
                    return True
        return False
    if prog == "curl":
        return any(a in ("-o", "-O", "--output", "--remote-name", "-T", "--upload-file") or
                   a.startswith("--output=") for a in args)
    if prog == "wget":
        to_stdout = any(a in ("-O-", "-qO-", "--spider") for a in args) or (
            "-O" in args and args.index("-O") + 1 < len(args) and args[args.index("-O") + 1] == "-")
        return not to_stdout
    if prog == "git":
        # skip global options (-C dir, -c k=v, --no-pager)
        rest = list(args)
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
        if not rest:
            return False
        sub, sub_args = rest[0], rest[1:]
        if sub in _GIT_FLAGGED:
            return any(a in _GIT_FLAGGED[sub] for a in sub_args)
        if sub == "config":
            positional = [a for a in sub_args if not a.startswith("-")]
            return len(positional) >= 2 or any(a in ("--unset", "--unset-all", "--add", "--replace-all",
                                                      "--remove-section", "--rename-section") for a in sub_args)
        if sub == "remote":
            return bool(sub_args) and sub_args[0] in ("add", "remove", "rm", "rename", "set-url", "prune")
        return sub in _MUTATING_SUBCOMMANDS["git"]
    if prog in _MUTATING_SUBCOMMANDS:
        subs = [a for a in args if not a.startswith("-")]
        return any(s in _MUTATING_SUBCOMMANDS[prog] for s in subs[:2])
    return False


def is_mutating_bash(command: str) -> bool:
    """True if the command recognizably writes/deletes/installs/kills
    something. Unknown programs count as NOT mutating (see module docstring)."""
    # A heredoc body is data, not commands — `cat <<EOF` followed by text
    # mentioning rm must not be judged by that text.
    # The rest of the heredoc's first line (`cat <<EOF > file`) still counts.
    head = command
    body = re.search(r"<<-?\s*['\"]?\w", command)
    if body:
        line_end = command.find("\n", body.start())
        head = command if line_end == -1 else command[:line_end]
    tokens = _tokens(head)
    if tokens is None:
        return False
    if _writes_via_redirect(tokens):
        return True
    return any(_segment_mutates(seg) for seg in _segments(tokens))
