"""
Единая точка входа для LoRA-дообучения flowAI-модели под её же тулы.

Подкоманды (можно по одной, можно все разом через `all`):
  extract  — датасет из реальной истории flowai.db (см. extract_dataset.py)
  train    — QLoRA-дообучение базовой HF-модели на датасете
  merge    — влить LoRA-адаптер в веса базовой модели
  gguf     — сконвертировать смёрдженные веса в GGUF через llama.cpp + квантовать
  install  — положить готовый GGUF в Ollama как новый тег (Modelfile + ollama create)
  all      — прогнать все шаги подряд с одними и теми же путями

ВАЖНО: qwen3-coder:30b в Ollama — это GGUF, НЕ то же самое, что HF-веса,
нужные для train/merge. Для --base-model нужен HF-репозиторий той же (или
меньшей — для быстрой итерации) модели с Hugging Face, скачивается отдельно
при первом запуске train.

Прогресс так же выводится в терминал, как у наших gpt_cli.py экспериментов.
"""

import argparse
import json
import subprocess
import sys

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
# HF_TOKEN и т.п. — тот же .env, из которого сам flowAI (main.py) грузит
# секреты, лежит в корне flowAI на уровень выше finetune/. Снимает rate
# limit на скачивание с Hugging Face (иначе "unauthenticated requests"
# предупреждение при каждом обращении к hub).
load_dotenv(HERE.parent / ".env")


def cmd_extract(args):
    subprocess.run(
        [sys.executable, str(HERE / "extract_dataset.py"), "--out", args.dataset]
        + (["--db", args.db] if args.db else [])
        + (["--include-rejected"] if args.include_rejected else []),
        check=True,
    )


def _load_dataset(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _build_tools_list(examples):
    """Минимальная схема тулов для apply_chat_template — по одному имени на
    уникальный tool name из датасета. Реальные JSON-схемы параметров живут в
    tool_wrappers.py и меняются со временем; здесь используем облегчённую
    версию (имя + произвольные kwargs) — этого достаточно, чтобы шаблон
    корректно разметил секцию доступных функций, а конкретные ключи args
    модель видит напрямую в целевых примерах."""
    names = sorted({ex["target_tool_call"]["name"] for ex in examples})
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"flowAI tool: {name}",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }
        for name in names
    ]


def _tools_subset(all_tools, target_name, max_tools, rng):
    """
    Полный список тулов (37 штук в реальных данных) рендерится в системный
    промпт при КАЖДОМ примере и стоит ~1876 токенов сам по себе — на карте
    6 ГБ это оставляет слишком мало места на саму историю диалога. Берём
    вместо этого маленькое подмножество: целевой тул (обязательно, иначе
    модель не увидит его схему) + случайные "отвлекающие" — модель всё ещё
    учится ВЫБИРАТЬ, просто не из полного реестра каждый раз, а стоит это
    в разы меньше по токенам.
    """
    target = next((t for t in all_tools if t["function"]["name"] == target_name), None)
    others = [t for t in all_tools if t["function"]["name"] != target_name]
    rng.shuffle(others)
    subset = ([target] if target else []) + others[:max(0, max_tools - 1)]
    rng.shuffle(subset)
    return subset


def _langchain_to_qwen_message(message):
    """{"role": "assistant", "tool_calls": [{"name", "args"}]} (LangChain,
    родной формат dataset.jsonl) -> {"name", "arguments"} (что ждёт jinja-
    шаблон Qwen). Остальные роли (user/tool/assistant без tool_calls) не трогаем."""
    if message.get("tool_calls"):
        return {
            **message,
            "tool_calls": [
                {"name": tc["name"], "arguments": tc["args"]} for tc in message["tool_calls"]
            ],
        }
    return message


def _cap_tool_content(messages, per_tool_cap):
    """Обрезает содержимое отдельных tool-сообщений (результаты read_file,
    git_diff и т.п. могут доходить до 20000 симв. — см. TOOL_OUTPUT_CHAR_CAP
    в mcp_agent/model_config.py) — без этого почти все длинные сессии
    вылетали бы по лимиту токенов целиком, хотя обрезать нужно было только
    раздутые куски, а не весь пример."""
    result = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "tool" and isinstance(content, str) and len(content) > per_tool_cap:
            m = {**m, "content": content[:per_tool_cap] + f"...[обрезано, было {len(content)} симв.]"}
        result.append(m)
    return result


def _render_raw(tokenizer, tools, messages, target_tool_call):
    """
    Возвращает (input_ids, loss_mask) — тот же принцип offset-based
    маскирования, что в gpt_cli.py:load_qa_examples: рендерим текст ДО
    целевого вызова (с add_generation_prompt=True) и текст ПОСЛЕ, разница
    в длине даёт границу, где начинается то, на чём считаем loss.
    """
    # Датасет хранит tool_calls в стиле LangChain ({"name", "args"}) — родном
    # для flowai.db. Шаблон чата Qwen ждёт стандартный {"name", "arguments"}
    # (см. chat_template: "tool_call.arguments | tojson"). Перекладываем
    # только на этапе рендеринга, сам dataset.jsonl остаётся верным
    # первоисточнику.
    messages = [_langchain_to_qwen_message(m) for m in messages]
    target_message = {"role": "assistant", "tool_calls": [
        {"name": target_tool_call["name"], "arguments": target_tool_call["args"]}
    ]}
    full_messages = messages + [target_message]

    prefix_text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prefix_text):
        return None  # шаблон непредсказуемо изменил префикс — пропускаем пример, не гадаем

    enc = tokenizer(full_text, return_offsets_mapping=True)
    boundary = len(prefix_text)
    loss_mask = [start >= boundary for start, _ in enc["offset_mapping"]]
    return enc["input_ids"], loss_mask


def _render_and_mask(tokenizer, tools, messages, target_tool_call, max_length,
                      per_tool_cap=800, max_drop_ratio=0.5, max_fit_iterations=12):
    """
    Готовит пример под бюджет токенов:
    1. Обрезает раздутые tool-результаты (см. _cap_tool_content) — дёшево,
       почти всегда достаточно.
    2. Если пришлось бы выкинуть больше max_drop_ratio ходов, чтобы влезть —
       ПРОПУСКАЕМ пример целиком, а не тащим его изувеченным. Без этого
       длинное расследование (у нас были сессии по 250+ ходов) превращается
       в "какие-то обрывки контекста -> вызов тула" без внятной причинно-
       следственной связи — плохой обучающий сигнал, хуже, чем вообще
       никакого. У самого flowAI для этого есть LLM-суммаризация истории
       (mcp_agent/compaction.py), но она не сохраняется в flowai.db, так что
       переиспользовать её тут нечем — предпочитаем честно пропустить,
       а не грубо подменить своим неполноценным суррогатом.
    3. Если резать всё же пришлось (в пределах допустимого) — оставляем
       явную пометку об этом в истории, чтобы модель не путалась на
       необъяснённом разрыве контекста.
    """
    if not messages:
        return None  # пустая история — нечего подавать модели на вход, пропускаем

    messages = _cap_tool_content(messages, per_tool_cap)

    def _approx_tokens(msgs):
        # ~3 символа/токен — грубо, но достаточно, чтобы понять МАСШТАБ
        # лишнего; точность не нужна, точную проверку всё равно делает фаза 2
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs) // 3

    original_pairs = (len(messages) - 1) // 2
    approx_initial = _approx_tokens(messages)
    if approx_initial > max_length and original_pairs > 0:
        keep_ratio = max_length / approx_initial
        if keep_ratio < (1 - max_drop_ratio):
            return None  # пришлось бы выбросить больше половины ходов — не насилуем пример

    # Фаза 1 — грубая прикидка по длине ТЕКСТА (без токенизатора, почти
    # бесплатно): режем сразу крупным куском, а не по 2 сообщения за раз.
    while len(messages) > 3 and _approx_tokens(messages) > max_length * 1.3:
        excess = _approx_tokens(messages) / max_length
        n_pairs_available = (len(messages) - 1) // 2
        drop_pairs = max(1, min(n_pairs_available - 1, int(n_pairs_available * (1 - 1 / excess) * 0.5)))
        del messages[1:1 + drop_pairs * 2]

    # Фаза 2 — точная подгонка настоящим токенизатором. После фазы 1 мы уже
    # близко к бюджету, так что обычно хватает пары итераций. Жёсткий предел
    # на случай, если оценка фазы 1 сильно промахнулась — единичный тяжёлый
    # пример не должен стопорить весь прогон на неопределённое время.
    fitted = None
    for _ in range(max_fit_iterations):
        result = _render_raw(tokenizer, tools, messages, target_tool_call)
        if result is None:
            return None
        input_ids, loss_mask = result
        if len(input_ids) <= max_length:
            fitted = (input_ids, loss_mask)
            break
        if len(messages) <= 3:
            return None  # даже "первый user + минимум хвоста" не влезает — сдаёмся
        del messages[1:3]  # обычно одна пара (assistant с tool_calls, tool-результат)
    else:
        return None  # не сошлось за max_fit_iterations — пропускаем, не зависаем

    final_pairs = (len(messages) - 1) // 2
    if final_pairs >= original_pairs:
        return fitted  # ничего не резали — пометка не нужна

    # Явная пометка о пропуске — чтобы модель не воспринимала внезапный
    # разрыв контекста как естественное начало расследования "с нуля".
    note = {
        "role": "tool", "name": "_context_note",
        "content": f"[пропущено {original_pairs - final_pairs} более ранних шагов "
                    f"расследования — не влезли в контекст]",
    }
    messages_with_note = [messages[0], note] + messages[1:]
    noted_result = _render_raw(tokenizer, tools, messages_with_note, target_tool_call)
    if noted_result is None or len(noted_result[0]) > max_length:
        return fitted  # вставка пометки внезапно не влезла — используем версию без неё
    return noted_result

    return None  # не сошлось за разумное число итераций — пропускаем, не зависаем


def _parse_boost_specs(specs):
    """--boost-arg "tool_name:arg_name" -> [(tool_name, arg_name), ...].
    Общий, непривязанный к конкретным тулам механизм: usage-паттерн, который
    считается "хорошей привычкой" (например, replace_lines с заполненным
    expected_first_line — но с тем же успехом любой другой тул/параметр в
    любом другом проекте на этом же finetune_cli.py), задаётся конфигом
    при запуске, а не зашит в код."""
    result = []
    for spec in specs or []:
        tool_name, _, arg_name = spec.partition(":")
        if not tool_name or not arg_name:
            raise SystemExit(f"--boost-arg ожидает формат 'tool_name:arg_name', получено: {spec!r}")
        result.append((tool_name, arg_name))
    return result


def _example_weight(ex, boost_specs, weight):
    """weight, если целевой вызов — один из tool_name в boost_specs И у него
    заполнен (truthy) соответствующий arg_name; иначе обычный вес 1."""
    call = ex["target_tool_call"]
    call_args = call.get("args") or {}
    for tool_name, arg_name in boost_specs:
        if call["name"] == tool_name and call_args.get(arg_name):
            return weight
    return 1


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel

    examples = _load_dataset(args.dataset)
    print(f"Примеров в датасете: {len(examples)}")
    if not examples:
        raise SystemExit("Датасет пуст — сначала запустите `extract`.")

    tools = _build_tools_list(examples)
    print(f"Уникальных тулов: {len(tools)}")

    # Токенизатор — лёгкий, не грузит GPU. Всю CPU-работу (токенизация +
    # маскирование + скользящее окно по 1600+ примерам, это минуты) делаем
    # ДО загрузки модели на GPU. Раньше модель грузилась первой и потом
    # простаивала на GPU ~20 минут, ожидая конца этого CPU-шага — похоже,
    # именно из-за этого виртуальный CUDA-контекст в WSL "протухал" и первый
    # же реальный шаг обучения падал с CUDA error: device not ready.
    print(f"Загружаю токенизатор {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    boost_specs = _parse_boost_specs(args.boost_arg)
    if boost_specs:
        print(f"Усиление веса ({args.boost_weight}x) для: "
              + ", ".join(f"{t}.{a}" for t, a in boost_specs))

    import random
    tools_rng = random.Random(0)  # детерминированно — воспроизводимость между прогонами
    tools_by_name = {t["function"]["name"]: t for t in tools}

    n_exact_scope = 0
    prepared = []
    skipped = 0
    boosted = 0
    for ex in tqdm(examples, desc="Токенизирую и маскирую примеры"):
        if ex.get("tool_scope") is not None:
            # Точный набор из roles.py (см. extract_dataset.py) — то, что
            # эта роль РЕАЛЬНО видела в этот момент, не приближение.
            example_tools = [tools_by_name[n] for n in ex["tool_scope"] if n in tools_by_name]
            n_exact_scope += 1
        else:
            # Легаси-сессия без разметки роли — единственное, что осталось,
            # это приближение случайной подвыборкой (см. _tools_subset).
            example_tools = _tools_subset(tools, ex["target_tool_call"]["name"], args.max_tools_per_example, tools_rng)
        result = _render_and_mask(
            tokenizer, example_tools, ex["messages"], ex["target_tool_call"],
            max_length=args.max_length, per_tool_cap=args.per_tool_cap,
        )
        if result is None:
            skipped += 1
            continue
        weight = _example_weight(ex, boost_specs, args.boost_weight)
        boosted += weight > 1
        prepared.extend([result] * weight)
    print(f"Готово к обучению: {len(prepared)} примеров-повторов "
          f"(усилено {boosted} исходных примеров x{args.boost_weight}, "
          f"пропущено {skipped} — не влезли даже после обрезки и окна, или пуст шаблон)")
    print(f"Из них с точным tool_scope по роли: {n_exact_scope}, "
          f"со случайной подвыборкой (легаси/неизвестная роль): {len(examples) - n_exact_scope}")

    # CPU-offload (--gpu-mem/--cpu-mem) — для моделей, чьи 4-битные веса
    # целиком не влезают в VRAM (30B+ на карте вроде 6GB). max_memory
    # ограничивает, сколько ВЕСОВ device_map="auto" положит на GPU —
    # остальное уходит на CPU RAM. llm_int8_enable_fp32_cpu_offload=True
    # обязателен здесь: без него transformers/bitsandbytes падают с ошибкой
    # "You can't have any CPU or disk device when using 4-bit quantization"
    # при первой же попытке смешанного device_map (несмотря на "int8" в
    # названии флага, он же разрешает это и для 4-бит). Без --gpu-mem
    # (7B, целиком влезает) — max_memory=None, поведение НЕ меняется.
    max_memory = None
    if args.gpu_mem:
        if not args.cpu_mem:
            raise SystemExit("--cpu-mem обязателен вместе с --gpu-mem")
        max_memory = {0: args.gpu_mem, "cpu": args.cpu_mem}
        print(f"Загружаю базовую модель {args.base_model} (4-бит) с CPU-offload "
              f"(GPU<={args.gpu_mem}, CPU<={args.cpu_mem}) — это будет НАМНОГО "
              "медленнее, чем целиком на GPU (перенос данных между устройствами "
              "на каждый шаг)...")
    else:
        print(f"Загружаю базовую модель {args.base_model} (4-бит) на GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=bool(max_memory),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto",
        max_memory=max_memory,
    )

    if args.resume_from:
        # Продолжаем УЖЕ существующий адаптер (те же LoRA-веса, не заново
        # инициализированные) — для дообучения дополнительными эпохами на
        # ТОМ ЖЕ датасете. Не путать с последовательным ростом датасета
        # новыми данными — там мы всегда учим с нуля на всём накопленном,
        # чтобы не забыть старое (см. README/финетюн-обсуждение); здесь риска
        # забывания нет, это просто продолжение тех же градиентных шагов.
        print(f"Продолжаю обучение существующего адаптера {args.resume_from}...")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Без этого активации всех слоёв держатся в памяти до backward — на 6 ГБ
    # карте даже с 4-битной базой и LoRA не хватает на последовательности
    # в тысячи токенов. Пересчитываем часть активаций заново при backward
    # вместо хранения — медленнее, но радикально меньше памяти.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    device = next(model.parameters()).device

    def _save_checkpoint():
        # Тот же путь, что и финальное сохранение — --resume-from указывает
        # СЮДА ЖЕ, так что "прервали, потом продолжили" всегда подхватывает
        # САМЫЙ СВЕЖИЙ чекпоинт, а не какой-то один зафиксированный. Адаптер
        # — это только LoRA-веса (МБ, не ГБ базовой модели), так что частое
        # перезаписывание дёшево даже на медленном CPU-offload прогоне.
        Path(args.adapter_out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.adapter_out)
        tokenizer.save_pretrained(args.adapter_out)

    import random
    global_step = 0
    interrupted = False
    try:
        for epoch in range(args.epochs):
            random.shuffle(prepared)
            total_loss = 0.0
            n_ok = 0
            n_oom = 0
            pbar = tqdm(prepared, desc=f"epoch {epoch}")
            for input_ids, loss_mask in pbar:
                try:
                    x = torch.tensor([input_ids[:-1]], device=device)
                    y = torch.tensor([input_ids[1:]], device=device)
                    mask = torch.tensor([loss_mask[1:]], dtype=torch.float32, device=device)

                    # loss считается только на target_tool_call — это всегда
                    # хвост последовательности (см. _render_raw: full_text = prefix
                    # + целевой вызов, без add_generation_prompt после него).
                    # Поэтому лосс-логиты нужны только на последних k позициях, а
                    # не на всех T — logits_to_keep просит модель не проецировать
                    # через lm_head (T x vocab_size=151936, доминирующая статья
                    # памяти при росте длины) всё, что заведомо замаскировано.
                    k = int(mask.sum().item())
                    logits = model(x, logits_to_keep=k).logits
                    y_tail = y[:, -k:]
                    mask_tail = mask[:, -k:]
                    per_token = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y_tail.reshape(-1), reduction="none"
                    ).reshape(mask_tail.shape)
                    n = mask_tail.sum().clamp(min=1)
                    loss = (per_token * mask_tail).sum() / n

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_ok += 1
                    global_step += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}", oom=n_oom, step=global_step)

                    if args.save_every and global_step % args.save_every == 0:
                        _save_checkpoint()
                        pbar.write(f"[шаг {global_step}] чекпоинт сохранён в {args.adapter_out}")
                except torch.cuda.OutOfMemoryError:
                    # Единичный аномально длинный/тяжёлый пример не должен ронять
                    # весь прогон (часы обучения) — пропускаем его, чистим кэш,
                    # продолжаем. Если это происходит часто — снижайте --max-length.
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    n_oom += 1
                    pbar.set_postfix(oom=n_oom)
                    continue

            print(f"epoch {epoch} средний loss: {total_loss / max(n_ok, 1):.4f} "
                  f"(успешных шагов: {n_ok}, пропущено по OOM: {n_oom})")
    except KeyboardInterrupt:
        # Живой сценарий — CPU-offload прогон растягивается на дни, его
        # обязательно будут прерывать посреди (а не только по завершении
        # эпохи). Без этого Ctrl+C посреди медленного шага стирает весь
        # прогресс с прошлого --save-every.
        interrupted = True
        print(f"\nПрервано на шаге {global_step} — сохраняю то, что успели обучить...")

    _save_checkpoint()
    print(f"\nLoRA-адаптер сохранён в {args.adapter_out}"
          f"{' (прервано пользователем, шаг ' + str(global_step) + ')' if interrupted else ''}")
    if interrupted:
        print(f"Продолжить: --resume-from {args.adapter_out}")


def cmd_merge(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Загружаю базовую модель {args.base_model}...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Накладываю адаптер {args.adapter_out}...")
    model = PeftModel.from_pretrained(base, args.adapter_out)
    model = model.merge_and_unload()

    Path(args.merged_out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.merged_out, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_out)
    print(f"Смёрдженные веса сохранены в {args.merged_out}")


def cmd_gguf(args):
    llama_cpp_dir = Path(args.llama_cpp_dir)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "llama-quantize"
    if not convert_script.exists():
        raise SystemExit(
            f"Не найден {convert_script}. Нужен склонированный и собранный llama.cpp "
            f"(https://github.com/ggml-org/llama.cpp), путь передать через --llama-cpp-dir."
        )

    raw_gguf = f"{args.gguf_out}.f16.gguf"
    print("Конвертирую HF -> GGUF (f16)...")
    subprocess.run(
        [sys.executable, str(convert_script), args.merged_out, "--outfile", raw_gguf, "--outtype", "f16"],
        check=True,
    )

    if not quantize_bin.exists():
        print(f"ВНИМАНИЕ: {quantize_bin} не найден — оставляю f16-версию без квантования: {raw_gguf}")
        return

    print(f"Квантую в {args.quant_type}...")
    subprocess.run([str(quantize_bin), raw_gguf, args.gguf_out, args.quant_type], check=True)
    print(f"Готово: {args.gguf_out}")


def cmd_install(args):
    modelfile = Path(args.gguf_out).with_suffix(".Modelfile")
    modelfile.write_text(f"FROM {args.gguf_out}\n")
    print(f"Modelfile: {modelfile}")
    subprocess.run(["ollama", "create", args.ollama_tag, "-f", str(modelfile)], check=True)
    print(f"\nГотово — модель доступна как: ollama run {args.ollama_tag}")


def cmd_all(args):
    print("=== 1/5 extract ===")
    cmd_extract(args)
    print("\n=== 2/5 train ===")
    cmd_train(args)
    print("\n=== 3/5 merge ===")
    cmd_merge(args)
    print("\n=== 4/5 gguf ===")
    cmd_gguf(args)
    print("\n=== 5/5 install ===")
    cmd_install(args)
    print(f"\nВесь пайплайн завершён. Проверить: ollama run {args.ollama_tag}")


def add_common_paths(p):
    p.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    p.add_argument("--adapter-out", default=str(HERE / "lora_adapter"))
    p.add_argument("--merged-out", default=str(HERE / "merged_model"))
    p.add_argument("--gguf-out", default=str(HERE / "flowai-tuned.gguf"))
    p.add_argument("--ollama-tag", default="flowai-tuned")


def main():
    parser = argparse.ArgumentParser(description="LoRA-дообучение flowAI под свои тулы")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--db", default=None)
    p_extract.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    p_extract.add_argument("--include-rejected", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    p_train = sub.add_parser("train")
    add_common_paths(p_train)
    p_train.add_argument("--base-model", required=True,
                          help="HF-репозиторий базовой модели, напр. Qwen/Qwen2.5-Coder-7B-Instruct")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--resume-from", default=None,
                          help="путь к уже сохранённому LoRA-адаптеру (--adapter-out прошлого запуска) — "
                          "продолжить дообучение ТЕХ ЖЕ весов вместо инициализации новых "
                          "(для 'прогнать 1 эпоху, посмотреть, потом ещё одну' на том же датасете)")
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--lora-r", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--lora-target-modules", default="q_proj,v_proj",
                          help="через запятую; по умолчанию — проверенная на 6ГБ карте связка "
                          "(полный набор из 7 модулей упирается в ~18-20ГБ на seq_len~2000)")
    p_train.add_argument("--max-length", type=int, default=4096)
    p_train.add_argument("--per-tool-cap", type=int, default=800, help="макс. символов на один результат тула в истории")
    p_train.add_argument("--boost-arg", action="append", default=[],
                          help="tool_name:arg_name — усилить вес примеров, где у этого тула "
                          "заполнен этот параметр (можно указывать несколько раз, общий "
                          "механизм, не привязан к конкретным тулам)")
    p_train.add_argument("--boost-weight", type=int, default=3)
    p_train.add_argument("--max-tools-per-example", type=int, default=8,
                          help="сколько тулов давать модели на пример без точного tool_scope по роли "
                          "(целевой + случайные отвлекающие) — влияет только на легаси/неизвестные роли")
    p_train.add_argument("--gpu-mem", default=None,
                          help="потолок VRAM под ВЕСА модели (напр. '3GiB') — включает CPU-offload "
                          "(device_map='auto' сам раскидает то, что не влезло, на --cpu-mem). "
                          "Оставь пусто для модели, которая и так целиком влезает на GPU (7B) — "
                          "поведение не меняется. Нужно для 30B+ на маленькой карте: заведомо НАМНОГО "
                          "медленнее (перенос данных между GPU/CPU на каждый forward+backward), "
                          "но не падает по OOM.")
    p_train.add_argument("--cpu-mem", default=None,
                          help="потолок ОЗУ под то, что не поместилось на GPU (напр. '24GiB') — "
                          "обязателен, если задан --gpu-mem")
    p_train.add_argument("--save-every", type=int, default=50,
                          help="сохранять чекпоинт адаптера каждые N шагов (0 — не сохранять "
                          "промежуточные, только в конце/при прерывании). Позволяет остановить "
                          "обучение в ЛЮБОЙ момент (Ctrl+C тоже сохраняет то, что успели) и "
                          "продолжить позже через --resume-from <тот же --adapter-out>.")
    p_train.set_defaults(func=cmd_train)

    p_merge = sub.add_parser("merge")
    add_common_paths(p_merge)
    p_merge.add_argument("--base-model", required=True)
    p_merge.set_defaults(func=cmd_merge)

    p_gguf = sub.add_parser("gguf")
    add_common_paths(p_gguf)
    p_gguf.add_argument("--llama-cpp-dir", required=True, help="путь к склонированному и собранному llama.cpp")
    p_gguf.add_argument("--quant-type", default="Q4_K_M")
    p_gguf.set_defaults(func=cmd_gguf)

    p_install = sub.add_parser("install")
    add_common_paths(p_install)
    p_install.set_defaults(func=cmd_install)

    p_all = sub.add_parser("all")
    add_common_paths(p_all)
    p_all.add_argument("--db", default=None)
    p_all.add_argument("--include-rejected", action="store_true")
    p_all.add_argument("--base-model", required=True)
    p_all.add_argument("--epochs", type=int, default=2)
    p_all.add_argument("--lr", type=float, default=2e-4)
    p_all.add_argument("--lora-r", type=int, default=16)
    p_all.add_argument("--lora-alpha", type=int, default=32)
    p_all.add_argument("--lora-target-modules", default="q_proj,v_proj")
    p_all.add_argument("--max-length", type=int, default=4096)
    p_all.add_argument("--per-tool-cap", type=int, default=800)
    p_all.add_argument("--boost-arg", action="append", default=[])
    p_all.add_argument("--boost-weight", type=int, default=3)
    p_all.add_argument("--max-tools-per-example", type=int, default=8)
    p_all.add_argument("--llama-cpp-dir", required=True)
    p_all.add_argument("--quant-type", default="Q4_K_M")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
