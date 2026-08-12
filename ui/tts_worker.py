"""
Воркер синтеза речи — запускается ЧЕРЕЗ venv-tts/bin/python, НЕ основной
.venv: Chatterbox (+ russian-text-stresser для расстановки ударений) требует
Python 3.11 и свои версии torch/transformers/diffusers, несовместимые с тем,
на чём держится SDXL/img2img в основном окружении (живая диагностика в этой
сессии: transformers 5.x у нас в .venv, а Chatterbox эпохи с этим стрессером
тянет за собой конфликтующий стек через spacy==3.6.*/pymorphy2). Общаться с
основным процессом можно только через subprocess, не через прямой import.

argv: [text, out_wav_path, audio_prompt_path?]. Третий аргумент — опциональный
референс-клип для voice cloning (Chatterbox сам умеет zero-shot клонирование
голоса по короткому сэмплу, см. её generate(audio_prompt_path=...)). Печатает
out_wav_path в stdout при успехе.
"""
import sys


def main() -> None:
    text = sys.argv[1]
    out_path = sys.argv[2]
    audio_prompt_path = sys.argv[3] if len(sys.argv) > 3 else None

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import torchaudio

    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    wav = model.generate(text, language_id="ru", audio_prompt_path=audio_prompt_path)
    torchaudio.save(out_path, wav, model.sr)
    print(out_path)


if __name__ == "__main__":
    main()
