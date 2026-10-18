# Сторонние компоненты и лицензии

Версии и лицензии взяты из `.venv/lib/python3.13/site-packages/*.dist-info/METADATA` на дату 2026-10-17 (pip в `.venv` не установлен, `pip show` недоступен). Для весов модели данные взяты из `app/detector.py` и `tools/fetch_weights.py` и из записи `/api/v1/model`.

Лицензия здесь — то, что заявлено в метаданных пакета. Это не юридическое заключение. Перед внешним распространением лицензии нужно проверить.

## Зависимости из `requirements.txt` и их транзитивные пакеты

| Компонент | Версия | Лицензия | Источник | Статус |
|---|---|---|---|---|
| fastapi | 0.143.3 | MIT | https://github.com/fastapi/fastapi | проверено (метаданные venv) |
| starlette | 1.8.0 | BSD-3-Clause | https://github.com/Kludex/starlette | проверено (метаданные venv; транзитивная зависимость FastAPI) |
| uvicorn | 0.54.0 | BSD-3-Clause | https://github.com/Kludex/uvicorn | проверено (метаданные venv) |
| python-multipart | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart | проверено (метаданные venv) |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (поле метаданных: «Apache 2.0») | https://github.com/opencv/opencv-python | проверено (метаданные venv) |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://github.com/numpy/numpy | проверено (метаданные venv) |
| onnxruntime | 1.31.0 | MIT | https://github.com/microsoft/onnxruntime | проверено (метаданные venv) |
| certifi | 2026.7.22 | **MPL-2.0** | https://github.com/certifi/python-certifi | проверено (метаданные venv). MPL-2.0 — copyleft уровня файла; certifi в проекте не модифицируется |
| openpyxl | 3.1.5 | MIT | https://openpyxl.readthedocs.io | проверено (метаданные venv). Используется только в локальных инструментах импорта меток |
| uvicorn[standard] (extras) | не разобраны | не проверено | — | дополнительные пакеты extras в этом документе не разобраны (не измерено) |

## Веса модели детекции людей

| Компонент | Версия / файл | Лицензия | Источник | Статус |
|---|---|---|---|---|
| YOLOX (код и веса yolox_s / yolox_tiny / yolox_nano) | релиз 0.1.1rc0; `yolox_s.onnx`: SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, 35 858 002 Б | Apache-2.0 (код; лицензия YOLOX) | https://github.com/Megvii-BaseDetection/YOLOX (лицензия: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) | код: LICENSE URL в `app/detector.py`; SHA-256 и размер yolox_s: `app/detector.py` (`MODELS`), проверяются `tools/fetch_weights.py`; лицензия самих весов — см. примечание ниже |

Примечание о весах:
- Веса обучены на датасете **COCO**. Условия датасета COCO (аннотации и изображения из разных источников) в этом документе **не проверялись**. Перед коммерческим использованием или внешним распространением весов нужна отдельная проверка.
- Для `yolox_tiny` (20 219 662 Б, SHA-256 `427cc366…`) и `yolox_nano` (3 659 407 Б, SHA-256 `c789161e…`) размеры и хеши записаны в `app/detector.py` (`MODELS`). Полные значения — там.
- Файлы весов в репозиторий не коммитятся (`weights/` в `.gitignore`). Скачиваются через `python -m tools.fetch_weights`.

## Инструменты разработки (не входят в `requirements.txt`)

| Компонент | Версия | Лицензия | Источник | Статус |
|---|---|---|---|---|
| playwright (python) | 1.64.0 | Apache-2.0 | https://github.com/microsoft/playwright-python | проверено (метаданные venv). Только для разработки и проверки интерфейса; в продукт не входит. Браузер Chromium, установленный playwright, лицензируется отдельно |

## Что не проверено в этом документе

- Лицензии extras `uvicorn[standard]` и прочих транзитивных пакетов `.venv`, которых нет в таблице.
- Лицензия Chromium, поставляемого playwright.
- Условия датасета COCO для весов YOLOX.
- Юридическая оценка лицензий (это не делалось).

## Research stack (stage 3, not in production)

Компоненты исследовательского стека находятся только в `requirements-research.txt` и `.venv-research/`. Они не входят в `requirements.txt` и не используются продуктовым сервисом. Источник: проверка `docs/REVIEW_CROSS.md` («Stage 3 licences», 2026-10-17). Это не юридическое заключение.

| Компонент | Версия / ревизия | Лицензия (как заявлено) | Источник | Коммерческое использование | Примечание |
|---|---|---|---|---|---|
| Qwen/Qwen2-VL-2B-Instruct (веса) | rev `895c3a49` | `license: apache-2.0` в карточке модели; файл LICENSE = «Apache License Version 2.0, January 2004» | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/895c3a49/LICENSE | да | В LICENSE нет текста «Tongyi Qianwen» и условий по активным пользователям. Условия Qwen для 72B к этому репозиторию не относятся |
| mlx-community/Qwen2-VL-2B-Instruct-4bit (конвертация) | rev `01af461c` | `license: apache-2.0` в карточке (наследуется от базовой модели); LICENSE-файла в репозитории нет | https://huggingface.co/mlx-community/Qwen2-VL-2B-Instruct-4bit/resolve/01af461c/README.md | да (наследуется) | `model.safetensors`: 1 245 868 907 Б, LFS oid префикс `3534d3dc…`, совпадает с `tools/fetch_weights.py` |
| mlx | 0.32.4 | «MIT License … Copyright © 2023 Apple Inc.» | https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/LICENSE | да | |
| mlx-vlm | 0.7.8 | «MIT License … Copyright © 2025 Prince Canuma» | https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/LICENSE | да | |
| transformers | 5.20.1 | «Apache License Version 2.0» | https://raw.githubusercontent.com/huggingface/transformers/v5.20.1/LICENSE | да | |
| huggingface-hub | 2.2.0 | «Apache License Version 2.0» | https://raw.githubusercontent.com/huggingface/huggingface_hub/v2.2.0/LICENSE | да | |
| safetensors | 0.8.0 | «Apache License Version 2.0» | https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/LICENSE | да | В метаданных только classifier Apache |
| tokenizers | 0.23.3 | «Apache License Version 2.0» | https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.3/LICENSE | да | Использован LICENSE из корня тега |
| pillow | 13.0.0 | `License-Expression: MIT-CMU`; текст LICENSE в теге — HPND-подобный | https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/LICENSE | да | Побуквенное сравнение с текстом SPDX MIT-CMU не выполнялось |
| Данные обучения Qwen2-VL (training-data caveat) | — | **UNVERIFIED**: вендор не публикует источники и условия данных в карточке | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct | не установлено | Лицензия Apache-2.0 на веса не сертифицирует обучающие данные. Та же оговорка, что и для COCO у YOLOX |

Непроверено: небольшие файлы модели, кроме `model.safetensors`, не сверялись по хешу независимо (см. `REVIEW_CROSS.md`).

### Stage 5 — frozen vision encoder (research-only, not shipped in production)

Статус: исследовательский компонент. Не входит в `requirements.txt`, продуктовым сервисом не используется. Пакеты закреплены только в `requirements-research.txt` и установлены только в `.venv-research`. Источники: `docs/REVIEW_CROSS.md` («Stage 5 — Part A», 2026-10-18), `docs/PROTOCOL_STAGE5.md` §10 (retrieved 2026-10-17/18), `docs/COORDINATION.md` (ONNX-экспорт). Это не юридическое заключение.

| Компонент | Версия / ревизия | Лицензия (как заявлено) | Источник | Коммерческое использование | Примечание |
|---|---|---|---|---|---|
| google/siglip2-base-patch16-224 (веса) | rev `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | `license: apache-2.0` в карточке модели; файла LICENSE в репозитории нет | https://huggingface.co/google/siglip2-base-patch16-224/resolve/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md | да по заявленной лицензии весов (не сертифицирует данные обучения) | Не gated; загружается нативно, без `trust_remote_code`. 8 файлов с SHA-256 в `tools/fetch_weights.py`; `model.safetensors` 1 500 800 904 Б, sha256 `612923381c76ec5a…` |
| vision.onnx (производный экспорт SigLIP2) | 354.6 MB, sha256 `4e95b080c23eff44…` | производный от весов; отдельной проверки лицензии не проводилось | получен локально скриптом экспорта (`tools/stage5_encoder.py`) | не установлено | Производный артефакт, в `.gitignore`, в репозиторий не коммитится |
| Данные обучения SigLIP2 (WebLI) | — | **UNVERIFIED**: набор не публикуется (по §10 — unreleased, Google-internal); условия данных не проверены | карточка модели, см. выше | не установлено | Apache-2.0 на веса не сертифицирует данные. Та же оговорка, что для Qwen2-VL и COCO |
| big_vision (справочный код SigLIP2) | commit `0127fb6b337ee2a27bf4e54dea79cff176527356` | Apache-2.0 (текст LICENSE) | https://raw.githubusercontent.com/google-research/big_vision/0127fb6b337ee2a27bf4e54dea79cff176527356/LICENSE | да | Только как справочный код, не выполнялся |
| transformers | 5.20.1 | Apache-2.0 | см. таблицу Stage 3 выше | да | SigLIP реализован нативно (`models/siglip`) |
| torch | 2.14.1 | BSD-3-Clause (текст — 3-clause BSD; в файле вариант не назван) | https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/LICENSE | да | Только экспорт и smoke-проверка в research-venv |
| onnx | 1.23.1 | Apache-2.0 | raw LICENSE, тег v1.23.1 | да | Только экспорт в research-venv |
| onnxscript | 0.7.2 | MIT | raw LICENSE, тег v0.7.2 | да | Только экспорт в research-venv |

Проверено: 8 файлов SIGLIP2 совпадают по размеру и SHA-256 (`docs/REVIEW_CROSS.md`, пункт 3); лицензии big_vision, transformers, torch, onnx и onnxscript сверены по raw LICENSE на зафиксированных тегах (пункт 2); версии совпадают с `requirements-research.txt` (пункт 5).

Отклонённые альтернативы:
- **openai/clip-vit-base-patch32** — в карточке нет лицензии; сказано: «Any deployed use case of the model — whether commercial or not — is currently out of scope».
- **laion/CLIP-ViT-B-32-laion2B** — карточка MIT, но содержит такую же оговорку об out-of-scope использовании и отговаривает от «ready-to-go industrial products»; есть вопросы к данным LAION-2B.
- **Florence-2** — требует `trust_remote_code`.
- **DINOv2-S/B** (Apache-2.0, HF rev `ed25f3a3` / `f9e44c81`; GitHub LICENSE Apache-2.0) — приемлемо, но не выбрано: самообучение на LVD-142M, нет языкового выравнивания (`docs/PROTOCOL_STAGE5.md` §12).

Непроверено: данные WebLI; юридическая оценка лицензий (не делалась); лицензия результатов экспорта.
