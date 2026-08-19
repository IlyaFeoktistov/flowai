"""
Потоковая генерация музыки (команда /music) — генерирует куски по очереди с
continuation (хвост ПРЕДЫДУЩЕГО куска подаётся как conditioning-seed для
следующего через generate_clip_sync, см. mcp_agent/servers/music_server.py),
чтобы стиль/тональность не начинались с нуля на каждом куске, как было бы с
независимыми вызовами generate_music.

Producer/consumer через asyncio.Queue(maxsize=2): генератор готовит куски
наперёд, пока играет текущий. На этом CPU генерация куска обычно МЕДЛЕННЕЕ
его собственной длительности, так что реально догнать playback и не
отставать редко получается — очередь просто естественно пустеет, и плеер
ждёт на queue.get() между кусками. Это не гарантия gapless-потока, а честная
попытка спрятать ожидание там, где для этого хватает запаса по CPU.

Кускам-файлам НЕ место в generated/ — там лежит то, что пользователь явно
попросил как готовый трек (`/gen`, generate_music-тул агента). Это служебные
временные файлы одного потокового сеанса — system tempfile, удаляются сразу
после проигрывания.
"""
import asyncio
import os
import tempfile

_CHUNK_SECONDS = 10.0
_TAIL_SECONDS = 3.0  # хвост-подсказка для следующего куска

_stop_flag = False


def stop() -> None:
    global _stop_flag
    _stop_flag = True


async def stream_music(prompt: str, on_status=None) -> None:
    """Крутится, пока не позвали stop() (или до конца текущего цикла
    генерации/проигрывания — досрочно прервать уже идущий синхронный вызов
    generate_clip_sync в executor-потоке нельзя, он просто доигрывается)."""
    global _stop_flag
    _stop_flag = False

    from mcp_agent.servers.music_server import generate_clip_sync
    from scipy.io import wavfile
    from ui.audio import play_wav

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    async def _generator() -> None:
        tail_audio = None
        tail_sr = None
        first = True
        while not _stop_flag:
            if on_status:
                on_status("генерирую..." if first else "готовлю продолжение...")
            audio_np, sr = await loop.run_in_executor(
                None, generate_clip_sync, prompt, _CHUNK_SECONDS, tail_audio, tail_sr
            )
            if _stop_flag:
                break
            # continuation-выход начинается с ПОВТОРА поданного хвоста (он —
            # начало последовательности, которую модель продолжает) — срезаем
            # его, иначе один и тот же кусок звука проигрывался бы дважды.
            if tail_audio is not None:
                skip = int(_TAIL_SECONDS * sr)
                new_part = audio_np[skip:]
            else:
                new_part = audio_np

            fd, path = tempfile.mkstemp(suffix=".wav", prefix="flowai-music-stream-")
            os.close(fd)
            wavfile.write(path, sr, new_part)

            tail_len = int(_TAIL_SECONDS * sr)
            tail_audio = audio_np[-tail_len:]
            tail_sr = sr
            first = False
            await queue.put(path)
        await queue.put(None)  # сентинел: генератор закончил

    gen_task = asyncio.create_task(_generator())
    try:
        while True:
            path = await queue.get()
            if path is None:
                break
            if on_status:
                on_status("играю...")
            await loop.run_in_executor(None, play_wav, path)
            try:
                os.remove(path)
            except OSError:
                pass
            if _stop_flag:
                break
    finally:
        _stop_flag = True
        gen_task.cancel()
        # cancel() без await не значит "поток встал" — если генератор в этот
        # момент сидел внутри run_in_executor (generate_clip_sync крутится
        # на CPU/GPU в отдельном треде), отмена asyncio-таска НЕ прерывает
        # сам тред: тяжёлое вычисление доигрывает до конца само, просто его
        # результат потом отбрасывается. Без этого await stream_music() (и
        # созданный от него _music_task в cli.py) считался бы "завершённым"
        # мгновенно — /music, вызванный заново сразу после стопа, запустил
        # бы ВТОРУЮ генерацию, пока первая ещё молотит в фоне тем же
        # CPU/GPU. await здесь заставляет реально дождаться, пока тред
        # отработает и CancelledError долетит.
        try:
            await gen_task
        except asyncio.CancelledError:
            pass
        # Если генератор успел положить кусок в очередь ПОСЛЕ того, как мы
        # вышли из цикла (stop во время проигрывания последнего куска) — он
        # там и останется никем не прочитанным и не удалённым. Подчищаем.
        while not queue.empty():
            leftover = queue.get_nowait()
            if leftover and os.path.isfile(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
