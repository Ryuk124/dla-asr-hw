# HW1, ASR

В этом ДЗ написал и обучил ASR на LibriSpeech. Начал с бейзлайна из шаблона, потом перешел на Conformer, добавил аугментации, больше данных, BPE и LM декодирование. Финальный вариант - Conformer на 16 блоков с BPE-CTC, 4-gram и LM рескорингом.

У энкодера d_model=512, 8 attention heads, RoPE и RMSNorm. На вход идут 80 log-mel полос, на выходе 1024 BPE-токена и отдельный CTC blank. Полную историю экспериментов, графики и неудачные попытки собрал в [отчете](REPORT.md). Ниже команды, чтобы все это запустить.

## Что получилось

Финальный чекпоинт - `conformer_bpe1024_960h_01/model_best.pth`, 29-я эпоха последнего этапа. С полным декодером получил 4.17% WER на test-clean и 10.48% на test-other. Хотел добить 10% на other, но немного не хватило.

Все значения в таблице в процентах, WER/CER посчитаны по всему корпусу:

| Split | Greedy WER | 4-gram WER | Rescore WER | Rescore CER |
| --- | ---: | ---: | ---: | ---: |
| dev-clean | 7.45 | 4.96 | 3.98 | 1.68 |
| dev-other | 17.12 | 11.89 | 10.15 | 5.84 |
| test-clean | 7.45 | 5.24 | 4.17 | 1.71 |
| test-other | 17.26 | 12.26 | 10.48 | 5.81 |

Параметры декодирования подбирал на dev-other для предыдущей BPE-модели. На финальном чекпоинте оставил их без изменений, test для подбора не использовал. Полные числа и параметры лежат в [results/960h_metrics.json](results/960h_metrics.json).

В колонке 4-gram первая гипотеза из того же N-best поиска, который использую для рескоринга: beam 1000, до 100 гипотез, `prune_history=False`.

## Установка

Использовал Python 3.12. Обучение и полный инференс гонял на Linux с H100. Для инференса ключ Comet не нужен, он используется только при логировании обучения.

Сначала нужно поставить [uv](https://docs.astral.sh/uv/getting-started/installation/), потом склонировать репозиторий и установить зависимости:

```bash
git clone https://github.com/Ryuk124/dla-asr-hw.git
cd dla-asr-hw
uv sync --python 3.12 --locked --default-index https://pypi.org/simple
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Дальше все команды запускаются из корня проекта. После установки использую `--no-sync`, чтобы uv не пересобирал окружение при каждом запуске.

На девбоксе веса и декодеры проверил. Установку с нуля в чистом окружении и полный запуск в Colab еще нужно проверить.

## Демо в Google Colab

[Открыть demo.ipynb](https://colab.research.google.com/github/Ryuk124/dla-asr-hw/blob/main/demo.ipynb)

В ноутбуке нужно выбрать GPU, вставить ссылку на архив со своим датасетом и запустить ячейки по порядку. Поддерживаются ZIP и TAR, прямая HTTP-ссылка или публичная ссылка на файл Google Drive. Внутри архива должна быть папка `audio/`, а если есть правильные тексты - еще и `transcriptions/`. Формат подробнее показал ниже.

Если ссылку не указывать, ноутбук возьмет одну запись из dev-clean. По умолчанию включен полный декодер с LM и рескорингом, для быстрой проверки можно выбрать `greedy`.

Ноутбук сам ставит Python 3.12 в отдельное окружение, устанавливает зависимости, скачивает веса и запускает инференс. Если есть транскрипции, считает метрики. Результаты можно скачать одним ZIP. Для установки внутри используются эти команды:

```bash
uv venv --python 3.12 .venv-colab
uv pip install --python .venv-colab/bin/python -r requirements.txt \
  --default-index https://pypi.org/simple
```

Загрузку архива, инференс, расчет метрик и сборку ZIP проверил на девбоксе. В чистой сессии Colab пока не проверял. Сам ноутбук и пример датасета нужно отправить в GitHub, чтобы они были доступны по ссылкам.

## Веса и языковые модели

Финальные веса и BPE-токенизатор залил на [Hugging Face](https://huggingface.co/Ryuk1245/dla-asr-hw), скачивание без авторизации проверил. В командах ниже закрепил конкретную версию `4d0ba08499ecc0ba1c318bd02ed5ab23e6ac886a`.

| Ресурс | Скачать | Куда положить |
| --- | --- | --- |
| Финальный BPE-CTC, 419 МБ | [model_best.pth](https://huggingface.co/Ryuk1245/dla-asr-hw/resolve/4d0ba08499ecc0ba1c318bd02ed5ab23e6ac886a/model_best.pth) | `saved/conformer_bpe1024_960h_01/model_best.pth` |
| BPE-токенизатор | [librispeech_bpe_1024.model](https://huggingface.co/Ryuk1245/dla-asr-hw/resolve/4d0ba08499ecc0ba1c318bd02ed5ab23e6ac886a/librispeech_bpe_1024.model) | `tokenizers/librispeech_bpe_1024.model` |
| LibriSpeech 4-gram | [4-gram.arpa.gz](https://www.openslr.org/resources/11/4-gram.arpa.gz) | `data/lm/4-gram.arpa` после распаковки |
| Transformer LM | [lm.ckpt](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/resolve/bcf56f762f37877912578f8800bc2a4f5171e5e5/lm.ckpt) | `data/lm/transformer_lm/lm.ckpt` |
| Токенизатор Transformer LM | [tokenizer.ckpt](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/resolve/bcf56f762f37877912578f8800bc2a4f5171e5e5/tokenizer.ckpt) | `data/lm/transformer_lm/tokenizer.ckpt` |

ASR-чекпоинт сохранил только со `state_dict`, без состояния оптимизатора. BPE нужно брать именно от него: если обучить новый токенизатор, ID токенов могут поменяться и старая голова модели им уже не будет соответствовать.

SHA256 весов: `60f05337e330c45405ecaca14c665609644707afd0b002c6c423b344bd85c031`.

Скачать модель и BPE:

```bash
uv run --no-sync hf download Ryuk1245/dla-asr-hw model_best.pth \
  --revision 4d0ba08499ecc0ba1c318bd02ed5ab23e6ac886a \
  --local-dir saved/conformer_bpe1024_960h_01

uv run --no-sync hf download Ryuk1245/dla-asr-hw \
  librispeech_bpe_1024.model librispeech_bpe_1024.vocab librispeech_bpe_1024.json \
  --revision 4d0ba08499ecc0ba1c318bd02ed5ab23e6ac886a \
  --local-dir tokenizers
```

Для greedy и собственного beam без LM этого достаточно. Чтобы запустить полный декодер, нужно еще скачать обе языковые модели. Из репозитория SpeechBrain беру только текстовую Transformer LM и ее токенизатор, предобученную акустическую модель оттуда не использую.

```bash
mkdir -p data/lm/transformer_lm

curl --fail --location --retry 5 --connect-timeout 30 --continue-at - \
  https://www.openslr.org/resources/11/4-gram.arpa.gz \
  --output data/lm/4-gram.arpa.gz

gzip -dk data/lm/4-gram.arpa.gz

uv run --no-sync hf download speechbrain/asr-transformer-transformerlm-librispeech \
  lm.ckpt tokenizer.ckpt \
  --revision bcf56f762f37877912578f8800bc2a4f5171e5e5 \
  --local-dir data/lm/transformer_lm
```

Архив 4-gram весит примерно 1.36 ГБ, после распаковки - 4.40 ГБ. Если скачивание оборвалось, можно повторить ту же команду `curl`, она продолжит с места остановки.

## Инференс на своих данных

Датасет должен выглядеть так:

```text
dataset/
├── audio/
│   ├── utterance_001.wav
│   └── utterance_002.flac
└── transcriptions/
    ├── utterance_001.txt
    └── utterance_002.txt
```

Можно использовать WAV, FLAC и MP3. Папка с транскрипциями необязательна, без нее получится просто распознать аудио. Имена файлов без расширения должны быть уникальными, т.к. по ним сопоставляются аудио, правильный текст и предсказание.

Для полного декодера с 4-gram и LM рескорингом:

```bash
uv run --no-sync inference.py \
  input_dir=/path/to/dataset \
  inferencer.save_path=data/saved/submission
```

Вместо `/path/to/dataset` нужно подставить свою папку. Результат будет здесь:

```text
data/saved/submission/custom/
├── utterance_001.txt
├── utterance_002.txt
└── predictions.jsonl
```

В TXT сохраняется итоговый текст, в `predictions.jsonl` дополнительно есть greedy и reference, если он был. Выходная папка раздела должна быть пустой, поэтому для повторного запуска лучше указать новый `inferencer.save_path`.

По умолчанию устройство выбирается автоматически, batch size стоит 1. Можно явно передать `inferencer.device=cuda` или `inferencer.device=cpu`, а размер батча поменять через `dataloader.batch_size=8`. Полный рескоринг на CPU будет работать медленнее.

Если нужен только greedy без загрузки LM:

```bash
uv run --no-sync inference.py \
  input_dir=/path/to/dataset \
  inferencer.save_path=data/saved/greedy \
  inferencer.decoding.method=greedy \
  text_encoder.use_lm=false \
  text_encoder.rescorer=null
```

Собственный CTC prefix beam search без LM:

```bash
uv run --no-sync inference.py \
  input_dir=/path/to/dataset \
  inferencer.save_path=data/saved/beam \
  inferencer.decoding.method=beam \
  inferencer.decoding.beam_size=50 \
  inferencer.decoding.token_top_k=40 \
  text_encoder.use_lm=false \
  text_encoder.rescorer=null
```

Beam search с 4-gram, но без Transformer LM рескоринга:

```bash
uv run --no-sync inference.py \
  input_dir=/path/to/dataset \
  inferencer.save_path=data/saved/beam_lm \
  inferencer.decoding.method=beam_lm \
  text_encoder.rescorer=null
```

Свой beam написал в [CTCTextEncoder](src/text_encoder/ctc_text_encoder.py), для LM-декодирования использую `pyctcdecode`. У `beam_lm` и рескоринга отличается режим pruning, поэтому первая гипотеза может немного отличаться.

## Расчет WER и CER

Если есть правильные транскрипции, после инференса можно посчитать метрики:

```bash
uv run --no-sync calc_metrics.py \
  --references /path/to/dataset/transcriptions \
  --predictions data/saved/submission/custom \
  --output data/saved/submission/metrics.json
```

Скрипт находит TXT с одинаковыми ID и считает корпусные метрики: суммарное кол-во ошибок делит на суммарную длину правильных текстов. Пробелы входят в CER. Текст привожу к нижнему регистру, оставляю английские буквы и пробелы, повторные пробелы схлопываю.

Если ID не совпадают, скрипт выдаст ошибку. Для расчета нужны правильные тексты для всех переданных предсказаний. В консоли метрики выводятся в процентах, в JSON сохраняются доли.

## Проверка на LibriSpeech

Для оценки сразу на четырех частях LibriSpeech есть готовый конфиг:

```bash
uv run --no-sync inference.py --config-name inference_bpe_rescored \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  inferencer.save_path=data/saved/librispeech_final
```

Он запускает dev-clean, dev-other, test-clean и test-other. Данные ищет в `data/datasets/librispeech`, недостающие части скачивает сам. Предсказания складывает в подпапки `val`, `val_other`, `test` и `test_other`.

Метрики, которые печатает сам инференсер, усредняются по записям. Поэтому они отличаются от корпусных значений в таблице выше и от результата `calc_metrics.py`. Исходную полную оценку гонял с batch size 16, при другом разбиении на батчи возможны небольшие численные отличия.

## Как обучал

Финальная модель получилась в три этапа:

| Этап | Данные | Эпохи × шаги | Batch size | Max LR | Инициализация |
| --- | --- | --- | --- | --- | --- |
| Char-CTC Conformer | clean-100 + other-500 | 90 × 1000 | 16 | 3e-4 | С нуля |
| BPE-CTC | clean-100 + other-500 | 30 × 1000 | 16 | 1e-4 | Энкодер Char-CTC |
| BPE-CTC 960h | clean-100 + clean-360 + other-500 | 30 × 1000 | 16 | 1e-4 | Полный BPE-чекпоинт |

Использовал AdamW, OneCycleLR и gradient clipping 5.0. Эпоха здесь - фиксированные 1000 батчей, а не полный проход по датасету. В последнем запуске выбрал чекпоинт 29-й эпохи. Веса clean-100, clean-360 и other-500 задал как 1:3.6:5, Common Voice в финальную смесь не добавлял.

BPE обучил на 177 227 транскрипциях из train-clean-100 и train-other-500. Dev и test для токенизатора не использовал. После долива clean-360 оставил тот же BPE.

Если хочется повторить обучение токенизатора, можно сохранить его в отдельный файл:

```bash
uv run --no-sync train_bpe.py \
  --data-dirs data/datasets/librispeech/train-clean-100 \
              data/datasets/librispeech/train-other-500 \
  --vocab-size 1024 \
  --output-prefix tokenizers/retrained_bpe_1024
```

Для логов обучения нужно задать `COMET_API_KEY` в окружении. Если онлайн-логи не нужны, можно добавить к команде `writer.mode=offline`.

Первый этап, символьный Conformer:

```bash
uv run --no-sync train.py \
  model=conformer_extra_big datasets=librispeech \
  ++text_encoder.use_lm=false \
  trainer.device=cuda dataloader.batch_size=16 dataloader.num_workers=4 \
  trainer.n_epochs=90 trainer.epoch_len=1000 trainer.save_period=20 \
  ++trainer.max_grad_norm=5.0 \
  lr_scheduler.max_lr=3e-4 lr_scheduler.pct_start=0.06 \
  ++loss_function.zero_infinity=true \
  writer.run_name=conformer_16blocks_90ep_01
```

Потом перенос энкодера в BPE и дообучение на 960 часах:

```bash
uv run --no-sync train.py --config-name bpe
uv run --no-sync train.py --config-name bpe_960
```

Для этих двух команд нужны чекпоинты предыдущих этапов: `saved/conformer_16blocks_90ep_01/model_best.pth` и `saved/conformer_bpe1024_01/model_best.pth` соответственно. На HF лежит финальный чекпоинт для инференса без optimizer state.

Из аугментаций использовал speed perturbation, gain, colored noise, frequency masking и time masking. Аудио-аугментации применяются до log-mel, спектрограммные - после.

Логи всех трех этапов:

- [Char-CTC, 90 эпох](https://www.comet.com/konstantin-stakheev/dla-asr-hw/of2rs4kdm1e064kvrv2y9p52j1x30rtr)
- [BPE, 600 часов](https://www.comet.com/konstantin-stakheev/dla-asr-hw/hpk3bgbxoxfv3h55loph17qwdh908oog)
- [BPE, 960 часов](https://www.comet.com/konstantin-stakheev/dla-asr-hw/k514yjd81hafadiwj9jx0qq1wvepn96j)

## Проверки

Тесты запускаются так:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

Проверял CTC/BPE decoding, свой prefix beam сравнением с полным перебором, инференс без транскрипций, сохранение ID, переиспользование LM-предсказаний и корпусные метрики. Еще прогонял настоящий чекпоинт: greedy на CPU и LM с рескорингом на GPU.

## Что осталось перед сдачей

По [требованиям домашки](https://github.com/markovka17/dla/blob/2026/hw1ASR/README.md) осталось проверить несколько вещей:

- Установку из `requirements.txt` и весь `demo.ipynb` в чистом Google Colab на своей ссылке с аудио.
- Перечитать [отчет](REPORT.md), графики и примеры туда уже добавлены.
- Проверить доступ к полным логам обучения и отдельным прогонам с аугментациями, собственным beam и LM.
- Отправить в GitHub код, конфиги, тесты, токенизатор, ноутбук, отчет и его картинки.
- Проверить доступность репозитория по правилам курса.
