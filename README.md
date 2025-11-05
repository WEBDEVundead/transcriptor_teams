# Real-time System Audio Transcriber (MVP)

Цель:
- Захват системного звука в реальном времени (то, что слышно в колонки/Teams).
- Разбиение потока на чанки (по умолчанию 5 сек).
- Отправка каждого чанка в Pollinations STT (model: openai-audio, language: uk) и получение транскрипта.
- Запись транскриптов в transcript.txt с таймстампами и именами чанков.
- Логирование в консоль и app.log; сохранение WAV чанков в папку chunks/.

Внимание и приватность:
- Перед записью убедитесь, что у вас есть согласие всех участников разговора/урока.
- Данные отправляются на внешнее API (pollinations.ai), не отправляйте конфиденциальную аудиоинформацию без необходимости.

## Быстрый старт

1) Установка Python 3.9+ и зависимостей:
- Установите Python 3.9 или новее.
- Создайте виртуальное окружение (опционально) и установите зависимости:
  pip install -r requirements.txt

2) Настройка виртуального аудиоустройства (самый простой путь)
- Windows:
  - Вариант A (рекомендуется): VB-CABLE. Установите VB-CABLE, затем в настройках звука выберите:
    - Воспроизведение: ваш обычный выход (наушники/колонки).
    - Запись: CABLE Output (VB-Audio Virtual Cable) — это и есть «loopback» источник.
    - В скрипте укажите --device "CABLE Output" или оставьте по умолчанию, если это устройство выбрано как стандартное.
  - Вариант B: WASAPI loopback. На Windows можно захватывать системный звук с выходного устройства напрямую:
    - Узнайте название вашего выходного устройства (колонки/наушники).
    - Запустите с флагом --wasapi-loopback и укажите это устройство как --device "Название устройства".
- macOS:
  - BlackHole (2ch) или Loopback. Установите BlackHole 2ch, затем:
    - В Audio MIDI Setup создайте Aggregate/Мulti-Output Device при необходимости.
    - В качестве входа для записи выберите BlackHole 2ch. В скрипте укажите --device "BlackHole 2ch".
- Linux:
  - PulseAudio или PipeWire:
    - Для PulseAudio используйте «monitor» устройство вашего выхода, например: "alsa_output.pci-0000_00_1b.0.analog-stereo.monitor".
    - Для PipeWire аналогично: найдите monitor-устройство.
    - Посмотреть устройства можно опцией --list-devices.

3) Переменные окружения
- Создайте .env (или используйте .env.example как шаблон):
  POLLINATIONS_BASE_URL=https://text.pollinations.ai
  POLLINATIONS_API_TOKEN=   # опционально, можно пусто для анонимного режима
  CHUNK_SECONDS=5           # по умолчанию 5 секунд

4) Запуск (пример):
- Показать устройства:
  python transcribe_mvp.py --list-devices
- Запуск захвата с явным устройством и чанком 5 сек:
  python transcribe_mvp.py --device "CABLE Output" --chunk 5
- Windows WASAPI loopback (если без VB-CABLE):
  python transcribe_mvp.py --device "Наушники (Realtek(R) Audio)" --wasapi-loopback --chunk 5
- macOS c BlackHole:
  python transcribe_mvp.py --device "BlackHole 2ch" --chunk 5
- Linux PulseAudio monitor:
  python transcribe_mvp.py --device "alsa_output.pci-0000_00_1b.0.analog-stereo.monitor" --chunk 5

Параметры:
- --device: имя или индекс аудиоустройства (см. --list-devices).
- --chunk: длина чанка в секундах (по умолчанию 5).
- --samplerate: частота дискретизации захвата (по умолчанию 48000).
- --channels: 1 или 2 (по умолчанию 2).
- --target-sr: частота для вывода/отправки (по умолчанию 16000; поддерживается и 48000).
- --base-url: базовый URL Pollinations API (по умолчанию https://text.pollinations.ai).
- --token: токен авторизации (если требуется), можно не указывать.
- --language: код языка (uk).
- --model: модель распознавания (openai-audio).
- --test-wav: обработать локальный WAV без захвата и выйти.
- --wasapi-loopback: Windows-режим loopback c выходного устройства.

5) Отладка
- Каждый чанк сохраняется в папку chunks/ как chunk_0001.wav, chunk_0002.wav и т.д.
- В случае ошибок см. app.log и консоль. Можно отдельный WAV проверить так:
  python transcribe_mvp.py --test-wav chunks/chunk_0001.wav
- Если тишина/ничего не записывается:
  - Убедитесь, что выбрано правильное устройство.
  - Для Windows: попробуйте VB-CABLE или --wasapi-loopback.
  - Для macOS: проверьте, что BlackHole установлен как вход.
  - Для Linux: используйте monitor-устройство выхода (PulseAudio/PipeWire).

## Формат транскрипта

Каждая запись в transcript.txt (UTF-8):
[2025-11-05T12:34:56Z] chunk_0001.wav (0.0s - 5.0s)
Текст: <отриманий_транскрипт>

Пример (2 чанка):
[2025-11-05T10:00:00Z] chunk_0001.wav (0.0s - 5.0s)
Текст: Це приклад першого аудіофрагмента для перевірки транскрипції.

[2025-11-05T10:00:05Z] chunk_0002.wav (5.0s - 10.0s)
Текст: А це другий фрагмент. Система працює успішно.

## Примечания
- Формат отправки: WAV PCM16, моно, 16 кГц (по умолчанию). Ресемплинг выполняется в коде.
- В коде используется multipart/form-data с полем file, model=openai-audio и language=uk; при неуспехе повторяет попытку (3 раза, экспон. бэкофф). В заголовки можно добавить Authorization: Bearer <TOKEN>, если требуется.
- Для наилучшего результата убедитесь, что вход не перегружен и уровень громкости не слишком высокий.
- В целях приватности не загружайте конфиденциальные разговоры без явного согласия участников.
