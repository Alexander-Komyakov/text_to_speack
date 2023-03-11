Text to speak

## Example:

```
Использование create_sounde_book:
    -t --text      ФАЙЛ     путь до текста.
    default text.txt
    -m --model     ФАЙЛ     путь до файла модели
    default: model.pt
    -g --gen_model ФАЙЛ     путь до сгенерированного
                            файла от основной модели
    default: ПУСТО
    -s --speaker   ИМЯ      имя спикера: aidar, baya,
                                         xenia, kseniya, eugene
    default: aidar
                            если используем gen_model: random
    -o --out_sound ФАЙЛ     конечный файл в формате wav
    default: sound.wav
    -l --ssml_true || --nossml_true
                            флаг указывающий использование
                            ssml разметки
    default: ПУСТО
Примеры:
    create_sound_book.py -t ssml_text.txt -o song.wav --ssml
    create_sound_book.py -t ssml_text.txt -o song.wav --ssm
```

# Created by Alexander Komyakov
For any kind of help, support, suggetion and request ask in me
## Follow on:
<p align="left">
<a href="https://github.com/Alexander-Komyakov"><img src="https://img.shields.io/badge/GitHub-Follow%20on%20GitHub-inactive.svg?logo=github"></a>
</p><p align="left">
<a href="https://vk.com/shurikkomyakov"><img src="https://img.shields.io/badge/VK-Follow%20on%20Vkontakte-blue?logo=vk&logoColor=white"></a>
</p><p align="left">
