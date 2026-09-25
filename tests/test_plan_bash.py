"""mcp_agent/plan_bash.py — plan mode's bash denylist: inspecting anything
passes, recognizably mutating commands (also behind sudo/xargs/bash -c/
$(...)/find -exec) are denied."""
import pytest

from mcp_agent.plan_bash import is_mutating_bash


@pytest.mark.parametrize("command", [
    "ps -e -o pid,user,%mem,vsz,cmd --sort=-%mem | head -20",
    "lsof -i :8000", "nvidia-smi", "docker ps -a", "systemctl status ollama",
    "git log --oneline 2>/dev/null | head", "grep -rn 'rm -rf' src", "free -h && df -h",
    "git branch -a", "git config --get user.name", "npm ls", "pip show fastapi",
    "cat <<EOF\nrm -rf /\nEOF", "curl -s http://x", "wget -qO- http://x",
    "find . -name '*.py' -exec grep -l foo {} +", "ollama ps", "echo $(date)",
    "sudo ss -ltnp", "top -b -n1 | head -30", "make_report_viewer --help", "cmd 2>&1 | tail",
])
def test_inspecting_commands_pass(command):
    assert is_mutating_bash(command) is False


@pytest.mark.parametrize("command", [
    "rm -rf build", "echo x > f.txt", "echo a >> log", "cat <<EOF > f.txt\nhi\nEOF",
    "sudo rm x", "sudo -u root rm x", "find . | xargs rm", "bash -c 'rm -rf x'", "echo $(rm x)",
    "git commit -m x", "git -C repo push", "git branch -D x", "git config user.name x",
    "npm install", "pip install x", "go mod tidy", "sed -i s/a/b/ f", "find . -delete",
    "find . -exec rm {} ;", "docker rm x", "kill 123", "ls; mv a b", "tee out",
    "curl -o f http://x", "wget http://x", "make", "systemctl restart ollama",
    "ollama rm m", "timeout 5 rm x", "env A=1 cp a b",
])
def test_mutating_commands_denied(command):
    assert is_mutating_bash(command) is True


@pytest.mark.parametrize("command", [
    'D="/tmp/review-staged-x-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$D" && echo "$D"',
    "python3 /home/u/.flowai/skills/scripts/match_rules.py . <(git diff --name-only) > /tmp/review-rules.txt",
    "echo x > /tmp/a.txt", "cp src/app.py /tmp/app.py.bak", "rm -rf /tmp/scratch", "mkdir -p /tmp/a /tmp/b",
    "git diff | tee /tmp/diff.txt", 'echo x >> "$TMPDIR/log"', "touch ${TMPDIR}/f",
])
def test_scratch_writes_to_temp_dir_pass(command):
    assert is_mutating_bash(command) is False


@pytest.mark.parametrize("command", [
    "rm -rf /tmp/../home/u/project", "cp /tmp/a src/app.py", "mv /tmp/a src/", "mkdir -p /tmp/a build",
    'D=build; mkdir -p "$D"', "echo x > /tmpfoo", "cp -t src /tmp/a", "chmod 777 src/x",
])
def test_writes_outside_temp_dir_still_denied(command):
    assert is_mutating_bash(command) is True
