"""
Голосовой ввод/вывод для WSL2: своих ALSA/PulseAudio CLI-тулов на этой машине
нет и нет passwordless sudo, чтобы их поставить (см. живую проверку в этой
сессии) — зато есть powershell.exe, доступный из WSL по абсолютному пути, и
он говорит напрямую с реальным аудио Windows-хоста. Мост в обе стороны уже
проверен вживую в этой сессии: воспроизведение — через
System.Media.SoundPlayer, запись — через MCI (mciSendString), оба —
подтверждены на слух самим пользователем.

Живой баг при отладке моста: SoundPlayer, которому дали путь через UNC
\\wsl.localhost\\... (так же как WSL монтирует Windows-диски), либо не играл
вообще, либо завершался за миллисекунды вместо реальной длительности файла —
без единой ошибки. Тот же файл, скопированный в C:\\Windows\\Temp (обычный
локальный Windows-путь), проигрывался корректно. Поэтому здесь ВСЕГДА
работаем через файлы под _WIN_TMP (который у WSL примонтирован как
/mnt/c/Windows/Temp) — никогда напрямую по WSL-путям.
"""
import os
import queue
import subprocess
import tempfile
import threading

_WIN_TMP = "/mnt/c/Windows/Temp/flowai-audio"
_REC_PATH = os.path.join(_WIN_TMP, "rec.wav")
_PLAY_PATH = os.path.join(_WIN_TMP, "play.wav")

_PS = "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"

# Chatterbox (+ russian-text-stresser) нужен ОТДЕЛЬНЫЙ интерпретатор
# (venv-tts, Python 3.11) — конфликтует по torch/transformers с основным
# .venv, где живёт diffusers/SDXL (см. tts_worker.py docstring про живую
# диагностику этого конфликта). Поэтому синтез — всегда subprocess, никогда
# прямой import.
# Two different "roots" since this file moved one level deeper under src/
# (src/ui/audio.py): _SRC_ROOT (unchanged dirname count) now correctly
# lands on src/ itself, which is where ui/ (and tts_worker.py inside it)
# actually lives — but venv-tts/ is a real repo-root directory that never
# moved there, so it needs the true root, one dirname further up.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)
_VENV_TTS_PY = os.path.join(_PROJECT_ROOT, "venv-tts", "bin", "python")
_TTS_WORKER = os.path.join(_SRC_ROOT, "ui", "tts_worker.py")

_whisper_model = None  # ленивая синглтон-загрузка, см. transcribe()


def _win_path(local_path: str) -> str:
    """/mnt/c/Windows/Temp/flowai-audio/x.wav -> C:\\Windows\\Temp\\flowai-audio\\x.wav"""
    rel = local_path[len("/mnt/c/"):]
    return "C:\\" + rel.replace("/", "\\")


def record_from_mic(seconds: int) -> str | None:
    """Записывает `seconds` секунд с дефолтного микрофона Windows через MCI,
    сразу в формате 16-бит/16kHz/моно — ровно то, что ждёт Whisper внутри
    (он ресемплит на 16kHz сам, отдавая его сразу этим избегаем лишнего
    ресемплинга и артефактов 8-битной записи, с которой был первый живой
    тест). Возвращает WSL-путь к WAV или None при сбое PowerShell-моста."""
    os.makedirs(_WIN_TMP, exist_ok=True)
    win_out = _win_path(_REC_PATH)
    script = f"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class FlowaiRec {{
    [DllImport("winmm.dll", EntryPoint="mciSendStringA", CharSet=CharSet.Ansi)]
    public static extern int mciSendString(string cmd, string ret, int retLen, IntPtr cb);
}}
'@
[FlowaiRec]::mciSendString('open new type waveaudio alias flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('set flowaimic time format ms bitspersample 16 channels 1 samplespersec 16000 alignment 2 bytespersec 32000 format tag pcm', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('record flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
Start-Sleep -Seconds {seconds}
[FlowaiRec]::mciSendString('stop flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('save flowaimic "{win_out}"', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('close flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
"""
    try:
        subprocess.run([_PS, "-NonInteractive", "-Command", script], capture_output=True, timeout=seconds + 15)
    except Exception:
        return None
    return _REC_PATH if os.path.isfile(_REC_PATH) else None


def start_recording() -> "subprocess.Popen | None":
    """Начинает запись с микрофона без фиксированной длительности — в
    отличие от record_from_mic (Start-Sleep на N секунд), здесь PowerShell
    запускает 'record' и сразу блокируется на чтении строки из своего
    stdin: длину записи задаёт не settings.stt_record_seconds, а сам
    пользователь через stop_recording() (см. Alt+R/Ctrl+C в ui/app.py).
    Возвращает Popen с открытым stdin или None при сбое запуска моста —
    процесс держит устройство 'flowaimic' открытым, пока stop_recording не
    пришлёт сигнал."""
    os.makedirs(_WIN_TMP, exist_ok=True)
    win_out = _win_path(_REC_PATH)
    script = f"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class FlowaiRec {{
    [DllImport("winmm.dll", EntryPoint="mciSendStringA", CharSet=CharSet.Ansi)]
    public static extern int mciSendString(string cmd, string ret, int retLen, IntPtr cb);
}}
'@
[FlowaiRec]::mciSendString('open new type waveaudio alias flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('set flowaimic time format ms bitspersample 16 channels 1 samplespersec 16000 alignment 2 bytespersec 32000 format tag pcm', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('record flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
[Console]::In.ReadLine() | Out-Null
[FlowaiRec]::mciSendString('stop flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('save flowaimic "{win_out}"', $null, 0, [IntPtr]::Zero) | Out-Null
[FlowaiRec]::mciSendString('close flowaimic', $null, 0, [IntPtr]::Zero) | Out-Null
"""
    try:
        # Без -NonInteractive: этот процесс явно должен дочитать строку из
        # СВОЕГО stdin (см. stop_recording) — тот флаг про интерактивные
        # ПРОМПТЫ хоста, но рисковать не стоит, раз именно stdin-чтение тут
        # критично для работы.
        return subprocess.Popen(
            [_PS, "-Command", script],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def stop_recording(proc: "subprocess.Popen", timeout: float = 15.0) -> str | None:
    """Останавливает запись, начатую start_recording(), и возвращает WSL-путь
    к получившемуся WAV (или None при сбое). Блокирующая — звать через
    run_in_executor, как и record_from_mic."""
    try:
        proc.stdin.write(b"\n")
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None
    return _REC_PATH if os.path.isfile(_REC_PATH) else None


def play_wav(path: str) -> bool:
    """Проигрывает WAV через Windows — path может быть ЛЮБЫМ путём, видимым
    из WSL (копируем в _WIN_TMP перед проигрыванием, см. модульный docstring
    про то, почему UNC-путь напрямую не работает)."""
    os.makedirs(_WIN_TMP, exist_ok=True)
    try:
        with open(path, "rb") as src, open(_PLAY_PATH, "wb") as dst:
            dst.write(src.read())
    except OSError:
        return False
    win_path = _win_path(_PLAY_PATH)
    script = f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"
    try:
        subprocess.run([_PS, "-NonInteractive", "-Command", script], capture_output=True, timeout=120)
    except Exception:
        return False
    return True


def transcribe(wav_path: str) -> str:
    """Распознаёт речь из WAV локально через faster-whisper (CPU — GPU занят
    Ollama/SDXL, см. project notes про VRAM). Модель грузится один раз на
    процесс при первом вызове (settings.stt_model, дефолт 'medium') и держится
    резидентной в RAM — повторная загрузка на каждый вызов стоила бы
    секунд впустую на каждый голосовой ход."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        import settings
        _whisper_model = WhisperModel(settings.get("stt_model"), device="cpu", compute_type="int8")

    segments, _info = _whisper_model.transcribe(wav_path, language=None)
    return "".join(seg.text for seg in segments).strip()


def unload_whisper() -> bool:
    """Frees the resident faster-whisper model held by transcribe() above —
    it reloads lazily on the next transcribe() call, same cost as the very
    first STT call in a session. Returns True if something was actually
    unloaded (False if it was never loaded, e.g. Alt+R was never used this
    session) — see model_lifecycle.unload_idle_models, the caller for both
    the manual /settings button and the automatic voice_mode-off sweep."""
    global _whisper_model
    if _whisper_model is None:
        return False
    _whisper_model = None
    return True


def record_voice_sample(seconds: int) -> str | None:
    """Записывает референс-клип для voice cloning (Chatterbox
    audio_prompt_path) и сохраняет его ПОСТОЯННО в storage.data_dir() —
    общее место вне проекта (~/.local/share/flowai/), тот же принцип, что
    уже применён к settings/memory/usage (см. storage.py) — не temp-файл,
    как у record_from_mic, а результат, который переживает сессии. Возвращает
    итоговый путь или None при сбое записи."""
    import shutil
    import storage

    wav_path = record_from_mic(seconds)
    if wav_path is None:
        return None
    dest = str(storage.data_dir() / "voice_clone.wav")
    shutil.copyfile(wav_path, dest)
    return dest


def synthesize_speech(text: str) -> str | None:
    """Синтезирует `text` через Chatterbox (venv-tts subprocess, см.
    tts_worker.py) и возвращает путь к WAV, или None при сбое. Блокирующая —
    звать через run_in_executor из async-кода (RTF ~4-5 на этом CPU, см.
    живой замер в этой сессии — это не быстрая операция). Клонирует голос
    (settings.tts_voice_clone_path), если он задан и файл реально существует
    — молча откатывается на стандартный голос Chatterbox, если файл потерян
    (не должно валить синтез целиком из-за пропавшего referен-клипа)."""
    import settings

    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="flowai-tts-")
    os.close(fd)
    cmd = [_VENV_TTS_PY, _TTS_WORKER, text, out_path]
    clone_path = settings.get("tts_voice_clone_path")
    if clone_path and os.path.isfile(clone_path):
        cmd.append(clone_path)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
    except Exception:
        return None
    if result.returncode != 0 or not os.path.isfile(out_path):
        return None
    return out_path


def speak(text: str) -> bool:
    """Синтезирует и сразу проигрывает `text`. Блокирующая (см.
    synthesize_speech) — звать через run_in_executor."""
    wav_path = synthesize_speech(text)
    if wav_path is None:
        return False
    played = play_wav(wav_path)
    try:
        os.remove(wav_path)
    except OSError:
        pass
    return played


class SpeechStreamer:
    """Озвучивает ответ по предложениям вместо одного блокирующего вызова
    на весь текст. Живой замер (см. synthesize_speech): RTF ~4-5 на этом CPU
    — ждать, пока модель допишет ВЕСЬ ответ, и только потом синтезировать
    его целиком означало полную сумму (время генерации + время синтеза)
    молчания. Здесь — два независимых потока-воркера на две раздельные
    очереди: синтез предложения N+1 идёт, пока играет N, и оба идут, пока LLM
    ещё дописывает предложение N+2 (feed() вызывается по ходу стрима, см.
    ui/stream.py). Живёт один экземпляр на процесс (см. StreamDisplay) —
    потоки-демоны не создаются заново на каждый ход, только очереди
    получают новые sentinel'ы между ходами."""

    def __init__(self) -> None:
        self._text_q: "queue.Queue[str | None]" = queue.Queue()
        self._play_q: "queue.Queue[str | None]" = queue.Queue()
        self._stop_flag = threading.Event()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def feed(self, text: str) -> None:
        """Добавляет готовый кусок текста (обычно — одно предложение) в
        очередь синтеза. Неблокирующая, можно звать прямо из event loop."""
        text = text.strip()
        if text:
            self._stop_flag.clear()
            self._text_q.put(text)

    def finish(self) -> None:
        """Сигнал конца текущего хода — не убивает воркеры, просто даёт
        _play_loop знать, что после текущей очереди больше ничего не придёт
        (используется только для симметрии с feed, воркеры и так молча
        ждут следующего элемента)."""
        self._text_q.put(None)

    def stop(self) -> None:
        """Отмена (Ctrl+C) — выбрасывает всё, что ещё не начало играть.
        Уже звучащий кусок не прерывается: доиграть WAV, который Windows
        уже начал через PlaySync, отсюда нечем — тот же компромис, что и
        раньше был в /music (см. stream_music.stop)."""
        self._stop_flag.set()
        for q in (self._text_q, self._play_q):
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item and os.path.isfile(item):
                    try:
                        os.remove(item)
                    except OSError:
                        pass

    def _synth_loop(self) -> None:
        while True:
            text = self._text_q.get()
            if text is None or self._stop_flag.is_set():
                continue
            wav_path = synthesize_speech(text)
            if wav_path and not self._stop_flag.is_set():
                self._play_q.put(wav_path)

    def _play_loop(self) -> None:
        while True:
            path = self._play_q.get()
            if path is None:
                continue
            if not self._stop_flag.is_set():
                play_wav(path)
            try:
                os.remove(path)
            except OSError:
                pass
