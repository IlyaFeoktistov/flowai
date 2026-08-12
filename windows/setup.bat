@echo off
chcp 65001 >nul
title FlowAI — Установка
setlocal enabledelayedexpansion

echo.
echo  ==============================================
echo    FlowAI — Установка и настройка
echo  ==============================================
echo.
echo  Будет скачано около 2-3 ГБ данных.
echo  Время установки: 10-20 минут.
echo.
pause

:: ── Переходим в корень проекта (родитель папки windows\) ─────────────────────
cd /d "%~dp0.."
echo  [Папка проекта] %CD%
echo.

:: ── 1. Проверка winget ────────────────────────────────────────────────────────
echo  [1/6] Проверяю winget...
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
echo  [2/6] Проверяю Python...

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
echo  [3/6] Проверяю Ollama...

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

:: ── 4. Python-зависимости ─────────────────────────────────────────────────────
echo.
echo  [4/6] Устанавливаю зависимости приложения...

if not exist ".venv" (
    "%PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo  [ОШИБКА] Не удалось создать виртуальное окружение.
        pause & exit /b 1
    )
)

.venv\Scripts\pip install --upgrade pip --quiet --disable-pip-version-check
.venv\Scripts\pip install -r windows\requirements-lite.txt --quiet
if errorlevel 1 (
    echo  [ОШИБКА] Не удалось установить зависимости.
    pause & exit /b 1
)
echo        Готово.

:: ── 5. Файл конфигурации (.env) ───────────────────────────────────────────────
echo.
echo  [5/6] Настройка конфигурации...

if not exist ".env" (
    copy .env.example .env >nul
    powershell -Command ^
        "(Get-Content .env) -replace 'OLLAMA_MODEL=.*', 'OLLAMA_MODEL=qwen2.5:7b' | Set-Content .env"
    echo        Создан .env, выбрана модель qwen2.5:7b.
) else (
    echo        .env уже существует — не изменяю.
)

:: ── 6. Скачивание языковой модели ─────────────────────────────────────────────
echo.
echo  [6/6] Скачивание языковой модели (qwen2.5:7b, около 4.7 ГБ)...
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
echo .venv\Scripts\python cli.py
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
pause
