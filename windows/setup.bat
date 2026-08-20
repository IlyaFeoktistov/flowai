@echo off
chcp 65001 >nul
title FlowAI — Установка
setlocal enabledelayedexpansion

echo.
echo  ==============================================
echo    FlowAI — Установка и настройка
echo  ==============================================
echo.
echo  Будет скачано около 3-4 ГБ данных.
echo  Время установки: 15-25 минут.
echo.
pause

:: ── Переходим в корень проекта (родитель папки windows\) ─────────────────────
cd /d "%~dp0.."
echo  [Папка проекта] %CD%
echo.

:: ── 1. Проверка winget ────────────────────────────────────────────────────────
echo  [1/7] Проверяю winget...
winget --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ОШИБКА] winget не найден.
    echo  Обновите Windows 10 до последней версии или
    echo  установите "App Installer" из Microsoft Store.
    echo.
    pause & exit /b 1
)
echo        OK.

:: ── 2. Python ─────────────────────────────────────────────────────────────────
echo.
echo  [2/7] Проверяю Python...

:: Проверяем PATH сначала
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    echo        Найден в PATH.
    goto :python_done
)

:: Стандартные пути winget-установки (user / system)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    echo        Найден: %LOCALAPPDATA%\Programs\Python\Python312\
    goto :python_done
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    echo        Найден: %LOCALAPPDATA%\Programs\Python\Python313\
    goto :python_done
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON=%ProgramFiles%\Python312\python.exe"
    echo        Найден: %ProgramFiles%\Python312\
    goto :python_done
)

echo        Python не найден, устанавливаю...
winget install Python.Python.3.12 ^
    --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo  [ОШИБКА] Не удалось установить Python.
    echo  Установите вручную с сайта python.org, затем запустите
    echo  этот скрипт снова.
    pause & exit /b 1
)
:: winget не обновляет PATH в текущей сессии — используем полный путь
set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
echo        Python 3.12 установлен.

:python_done

:: ── 3. Ollama ─────────────────────────────────────────────────────────────────
echo.
echo  [3/7] Проверяю Ollama...

ollama --version >nul 2>&1
if not errorlevel 1 (
    set "OLLAMA=ollama"
    echo        Найден в PATH.
    goto :ollama_done
)

if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    echo        Найден: %LOCALAPPDATA%\Programs\Ollama\
    goto :ollama_done
)

echo        Ollama не найден, устанавливаю...
winget install Ollama.Ollama ^
    --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo  [ОШИБКА] Не удалось установить Ollama через winget.
    echo  Скачайте вручную с сайта ollama.com, затем запустите
    echo  этот скрипт снова.
    pause & exit /b 1
)
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
echo        Ollama установлен.

:ollama_done

:: ── 4. Git for Windows (даёт agent'у настоящий bash — см. FlowAI's ────────────
::      bash_windows_server.py: без него тулы агента, завязанные на shell
::      (git/grep/find/&&-цепочки), не смогут выполняться вообще) ─────────────
echo.
echo  [4/7] Проверяю Git for Windows (нужен агенту как shell)...

:: Явные пути Git for Windows проверяем ПЕРВЫМИ, а не "where bash.exe" —
:: на машине с включённым WSL "where" находит C:\Windows\System32\bash.exe
:: раньше (это заглушка, запускающая WSL-дистрибутив, а не настоящий
:: автономный bash — совсем другая программа, и не то, что нам нужно здесь).
if exist "%ProgramFiles%\Git\bin\bash.exe" (
    set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
    goto :gitbash_found
)
if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" (
    set "GITBASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
    goto :gitbash_found
)

:: Фолбэк на PATH, но с фильтром против System32\bash.exe (WSL-заглушка,
:: см. комментарий выше) — берём первое совпадение, которое НЕ из System32.
for /f "delims=" %%b in ('where bash.exe 2^>nul') do (
    echo %%b | find /i "System32" >nul
    if errorlevel 1 (
        set "GITBASH=%%b"
        goto :gitbash_found
    )
)

echo        Git for Windows не найден, устанавливаю...
winget install Git.Git ^
    --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo  [ПРЕДУПРЕЖДЕНИЕ] Не удалось установить Git for Windows автоматически.
    echo  Скачайте вручную: https://git-scm.com/download/win, затем перезапустите
    echo  этот скрипт — без него агент не сможет пользоваться shell-тулами.
    set "GITBASH="
    goto :gitbash_done
)
if exist "%ProgramFiles%\Git\bin\bash.exe" set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
echo        Git for Windows установлен.

:gitbash_found
echo        Найден: %GITBASH%
:gitbash_done

:: ── 5. Python-зависимости ─────────────────────────────────────────────────────
echo.
echo  [5/7] Устанавливаю зависимости приложения (полный стек: LangChain/
echo        LangGraph/MCP, не облегчённая версия)...

if not exist ".venv" (
    "%PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo  [ОШИБКА] Не удалось создать виртуальное окружение.
        pause & exit /b 1
    )
)

.venv\Scripts\pip install --upgrade pip --quiet --disable-pip-version-check
.venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ОШИБКА] Не удалось установить зависимости.
    pause & exit /b 1
)
echo        Готово.

:: ── 6. Файл конфигурации (.env) ───────────────────────────────────────────────
echo.
echo  [6/7] Настройка конфигурации...

if not exist ".env" (
    copy .env.example .env >nul
    :: OLLAMA_MODEL — дефолтная модель flowAI (glm-4.7-flash) требует
    :: expert_streaming, отдельно собираемый из исходников форк llama.cpp
    :: (Linux/CUDA-toolchain-only, см. корневой README «Модели») — здесь не
    :: ставится, поэтому явно переключаем на модель, работающую на голом
    :: Ollama. .env.example не содержит строк OLLAMA_MODEL/
    :: EXPERT_STREAMING_ENABLED — дописываем, а не пытаемся заменить
    :: несуществующую строку.
    (
        echo OLLAMA_MODEL=qwen2.5:7b
        echo EXPERT_STREAMING_ENABLED=0
    ) >> .env
    if defined GITBASH (
        echo FLOWAI_WINDOWS_BASH=%GITBASH% >> .env
    )
    echo        Создан .env, выбрана модель qwen2.5:7b.
) else (
    echo        .env уже существует — не изменяю.
)

:: ── 7. Скачивание языковой модели ─────────────────────────────────────────────
echo.
echo  [7/7] Скачивание языковой модели (qwen2.5:7b, около 4.7 ГБ)...
echo        Пожалуйста, подождите. Это делается один раз.
echo.

:: Запускаем Ollama-сервер в фоне (на случай если он ещё не стартовал)
start "" /b "%OLLAMA%" serve
timeout /t 4 /nobreak >nul

"%OLLAMA%" pull qwen2.5:7b
if errorlevel 1 (
    echo.
    echo  [ПРЕДУПРЕЖДЕНИЕ] Не удалось скачать модель автоматически.
    echo  После установки запустите вручную:
    echo      ollama pull qwen2.5:7b
)

:: ── Создаём файл запуска ──────────────────────────────────────────────────────
echo.
echo  Создаю файл запуска...
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

(
echo @echo off
echo chcp 65001 ^>nul
echo title FlowAI
echo cd /d "%%~dp0"
echo.
echo rem Ollama на Windows обычно запускается автоматически.
echo rem Эта строка — страховка на случай если сервис остановлен.
echo if exist "%OLLAMA_EXE%" (
echo     tasklist /fi "imagename eq ollama.exe" 2^>nul ^| find /i "ollama.exe" ^>nul
echo     if errorlevel 1 start "" /b "%OLLAMA_EXE%" serve
echo )
echo timeout /t 2 /nobreak ^>nul
echo.
echo .venv\Scripts\python src\cli.py
echo pause
) > "Запустить FlowAI.bat"

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo  ==============================================
echo    Установка завершена!
echo.
echo    Для запуска откройте файл:
echo    "Запустить FlowAI.bat"
echo  ==============================================
echo.
if not defined GITBASH (
    echo  [ВНИМАНИЕ] Git for Windows не найден — shell-тулы агента ^(git,
    echo  grep, любые команды в чате^) работать не будут, пока он не
    echo  установлен. Поставьте https://git-scm.com/download/win и
    echo  перезапустите flowai — остальной чат при этом работает и так.
    echo.
)
pause
