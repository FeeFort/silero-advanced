# 👋 Приветствуем в Silero-Advanced!
Silero-Advanced - это улучшенный модуль Silero.
# 🤔 В чем отличие?
Оригинальная Silero не может озвучивать цифры и некоторые символы. В этом модуле данная проблема решена.
# ⬇️ Установка
```py
pip install silero-advanced
```
# 💻 Использование
Модуль имеет 3 функции:<br>

• Синтезировать<br>
• Синтезировать и воспризвести<br>
• Синтезировать и сохранить аудио в формате wav<br>

Пример 1 функции:
```py
import time
import sounddevice as sd
from silero_advanced import synthesis

synthesis = synthesis()
audio = synthesis.synthesis("Привет, Мир! Я умею озвучивать цифры и символы! Например: 2 + 2")

sd.play(audio, synthesis.sample_rate)
time.sleep(len(audio) / synthesis.sample_rate)
sd.stop()
```

Пример 2 функции:
```py
from silero_advanced import synthesis

synthesis = synthesis()
synthesis.synthesisAndPlay("Привет, Мир! Я умею озвучивать цифры и символы! Например: 2 + 2")
```

Пример 3 функции:
```py
from silero_advanced import synthesis

synthesis = synthesis()
synthesis.synthesisAndSaveWav("Привет, Мир! Я умею озвучивать цифры и символы! Например: 2 + 2")
# Файл будет сохранен в корневом каталоге
```
# 🙇 Спасибо всем авторам библиотек, что были использованы в данном проекте:
<a href="https://silero.ai/">Silero</a><br>
<a href="https://github.com/seriyps/ru_number_to_text">Sergey Prokhorov</a>