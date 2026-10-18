# Forte: локальный сервис анализа видео (статус: НЕ готов выносить вердикты)

Сервис принимает видеоклип по HTTP, локально прогоняет его через детектор людей (YOLOX-S, ONNX Runtime) и правила, и возвращает результат задания.

**Актуально после stage13 (2026-10-18):** по явному разрешению пользователя все **40 уникальных клипов** включены в обучение и exploratory CV. Прежние 23 holdout-клипа теперь использованы; **независимого holdout больше нет**. Исторические упоминания закрытого holdout ниже относятся к соответствующим прежним этапам. Исходный split сохранён на диске для аудита, но больше не задаёт независимый тест. Production по-прежнему fail-closed.

> **Главный вывод, который нужно знать до использования.** Без валидированных правил сервис возвращает `failed` с кодом `no_validated_rule`. Он **не готов выносить вердикты**. Работающий локальный инференс ≠ работающее обнаружение нарушений: пайплайн запускается и отдаёт наблюдения, исследовательские метрики на реальных данных приведены ниже, но независимое качество не подтверждено и ни одно правило не валидировано. Флаг `ALLOW_UNVALIDATED_RULES=1` предназначен только для демонстрации. Он не является продуктовым вердиктом и не означает никакого заявления о качестве.

**Итог измерения на holdout C02 (24 файла, 23 sha16, метки XLSX без ручной проверки):** детектор людей даёт покрытие (23/23 клипов прошли гейты), но кандидатные паттерны не соответствуют нарушениям: recall 0.0, ни одно правило не валидировано, продакшн fail-closed (`failed/no_validated_rule`). Holdout уже был однажды оценён на этапе 1 и не является полностью независимым. Подробности и следующий шаг — в [docs/QUALITY.md](docs/QUALITY.md).

**Этап 7 (внутриклиповый контраст с неразмеченным фоном, протокол `ab2c71c974fa9d7d`, один прогон на train-17, holdout не открывался): NO-GO для обеих семей признаков.** Клип F1 0.300 (SigLIP2 zone) и 0.267 (motion/geometry) против базовых 0.4545 (всё положительно) и 0.600 (только длительность); локализация не лучше позиционного приора. Продакшн fail-closed, ни одно правило не валидировано. Подробности: [QUALITY.md §3f](docs/QUALITY.md).

Подробности: [архитектура](docs/ARCHITECTURE.md), [качество и что именно измерено](docs/QUALITY.md), [развёртывание](docs/DEPLOY.md), [лицензии](docs/LICENSES.md), [нормативные кандидаты](docs/NORMS.md).

## Актуальное состояние: платформа и исследовательский детектор

Финальные проверки secure upload: **251 passed**, включая реальный pinned encoder/saved stage15 на новом синтетическом MOV и19 supervisor tests; одно существующее Starlette/httpx warning. Никаких real fits.

**Актуальный технический handoff: защищённый stage15 upload.** Stage15 остаётся выбранным research candidate; stage16 хуже и не продвигается. Новый MOV/MP4 проходит decode → pinned SigLIP2 full-frame → исторические float16 cache semantics/нормализацию → сохранённые stage15 classifier/localizer. Lookup по known clip/hash и обучение не используются. Настоящий synthetic MOV E2E с реальным encoder/весами совпал с независимым прежним extractor.

Локальная UI: `http://127.0.0.1:8000/`. Raw binary upload: POST `/research/stage15/analyze`, Content-Type `video/quicktime` / `video/mp4` / `application/octet-stream`, заголовок `X-API-Key`. Это raw body, не multipart. Ответ только `research_only=true`, `production_enabled=false`, `status=not_validated`, `human_review_required=true`, logit и максимум один кандидат. Нет verdict/type/norm. Key не сохраняется UI в browser storage; field очищается при отправке.

Auth использует общий constant-time helper платформы и выполняется до body. Admission: один research request до чтения body, остальные429; release в finally. Upload ≤60MiB, приём ≤60с; JSON route ≤1024bytes/5с. MOV/MP4 ≤150с, ≤1920×1080, ≤120fps, минимум3 sample points. Файлы называются сервером, filename/пути клиента не используются; плейлисты отклоняются, FFmpeg external data references запрещены. Child runtime ≤120с с SIGKILL process group/cleanup, sampled group RSS ≤2048MiB для video, CPU/NOFILE64/FSIZE1MiB/CORE0 caps; Linux дополнительно RLIMIT_AS. Job dir0700, файлы0600, child environment не содержит API_KEY. Ошибки статические:401/408/413/415/422/429/503/504. Это trusted-code isolation, не полный OS filesystem/network sandbox; RSS cap проверяется периодически, возможен краткий overshoot. Linux runtime не проверен.

Запуск research UI/API из корня проекта с заранее безопасно заданным API_KEY:

```bash
.venv/bin/python -B -m uvicorn tools.research_video:create_video_app --factory --workers 1 --no-proxy-headers --host 127.0.0.1 --port 8000 --no-access-log
```

`tools.research_supervisor --start-approved` предназначен только для явно разрешённого24h доступа с существующим API_KEY: максимум86400с по wall+monotonic, без restart, shutdown/childexit/expiry останавливают обе группы. Состояние/PID/expiry: `local_private/research_access_state.json`; key туда не попадает. Подготовлен проверенный Homebrew cloudflared2026.10.1, binary+manifest в `local_private/bin/`, без новой установки. Cloudflare URL временный, без uptime guarantee; трафик проходит Cloudflare. Внешний watchdog дополнительно контролирует смерть supervisor. Никакого GO модели это не означает. [Quick Tunnel limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

**Stage16 — последний фиксированный эксперимент, отрицательный результат:** 12 признаков распределения и lag-1 persistence четырёх temporal-change каналов; линейная logistic-head L2=1, баланс классов, scaler только train-fold, порог 0. Локализаторы stage15 заморожены. Все 40 клипов (21+/19−), те же 5 folds, все 22 события. Пять CV-fit и final fit сошлись за 0.498 с, reload точный. Никакого подбора или автоматической замены stage15.

OOF clip: **F1 0.4211, macro-F1 0.4486, precision 0.4706, recall 0.3810, AUC 0.3634**, TP8/FP9/FN13/TN10. Stage15: F1 0.5854, AUC 0.5514; duration baseline F1 0.7347. End-to-end events ±5 с: **F1 0.2564, precision 0.2941, recall 0.2273, TP5/FP12/FN17** против stage15 F1 0.3333. Always-open gate + тот же localizer: F1 0.3548, но 19/19 false alarms на negative. Stage16: false alarms 9/19; gate misses 13 positive-клипов / 14 событий. Локализация по всем 21 positive без gate неизменна: 11/21, exact interval 0/2. Event AUC для этого matching-контракта не определена; AUC выше относится к clip logits.

Результат сохранён отдельно: `local_private/research_models/stage16_persistence/` (weights/CV weights/config/manifests/summary/verification), SHA-256 весов `3278d520511e0937f8243701f9742a7e596e231d95d1cf26bd944399fd90c4b1`. [Протокол](docs/PROTOCOL_STAGE16.md) зафиксирован до fit. Stage15 и все прежние модели/архивы сохранены. **Stage16 не улучшила качество; дальнейших экспериментов нет. NO-GO, adaptive reused CV, независимого holdout нет.**

**Stage15 — фиксированный classifier stage14 + отдельный temporal localizer:** 40 уникальных клипов (21 positive / 19 negative), те же 5 целоклиповых folds. Восемь скалярных temporal-change признаков; линейный локализатор L2=1 обучен только на vendor timestamps train-fold (окно ±2 с), баланс классов и scaler — только train-fold. Порог classifier 0; максимум один кандидат на клип. Метод и matching зафиксированы [до fit](docs/PROTOCOL_STAGE15.md).

**OOF end-to-end события, однозначное matching ±5 с, все 22 GT события:** TP7/FP13/FN15, precision **0.350**, recall **0.3182**, F1 **0.3333**. Исходный peak stage14 с тем же classifier и matching: F1 **0.0952** (TP2/FP18/FN20). Локализация независимо от gate по всем 21 positive: **11/21** против 2/21; медианная ошибка 4.530 с. Gate пропустил 9 positive clips / 9 событий; ложные срабатывания на negative: **8/19**. Exact interval: 0/2, сетка 1 fps не покрывает эти интервалы. Ограничение в один кандидат не позволяет обнаружить оба события одного клипа.

Clip-метрики не изменены: positive F1 **0.5854**, macro-F1 **0.5747**, precision 0.600, recall 0.5714, AUC 0.5514, TP12/FP8/FN9/TN11. Duration baseline F1 0.7347 / macro-F1 0.6577 сильнее. Все оценки adaptive/exploratory на повторно использованных folds; **независимого holdout нет, NO-GO**. Улучшение локализации не подтверждает production-качество.

Артефакты: `local_private/research_models/stage15_two_component/` — combined final/CV weights, config, input/artifact manifests, summary и verification. SHA-256 весов `44731138495b1cd1c75021d424d38af7c64a227018c5befeadde0fcddc91bd32`. Пять CV-fit и final fit локализатора сошлись (0.51 с); classifier-модели побитово сохранены. Reload точный, независимый event matcher подтвердил counts, восстановление train-fold scaler дало нулевую разницу. Старые модели и архивы сохранены.

Исторический opaque-ID API теперь защищён auth/worker caps: `tools.stage15_api.create_research_app(predictor)`; picklable provider создаётся через `tools.research_provider.make_model_provider(model_dir, samples)`, где `samples` — локальный реестр opaque ID → timestamps/centered cached features. POST `/research/stage15/predict` принимает только `{"sample_id":"local_id"}`. Ответ содержит `research_only=true`, `production_enabled=false`, `status=not_validated`, некалиброванный logit и максимум один кандидат; отсутствие кандидата не означает отсутствия нарушения. Нет verdict/type/norm. Provider отсутствует или некорректен — статический 503; неверный запрос — 422 без отражения ввода. Адаптер проверен синтетически и не подключён к production; актуальный upload flow описан выше.

Полный регрессионный прогон после stage15: **210 passed**, одно существующее предупреждение Starlette/httpx; реальные данные тестам не нужны. Команда: `python -B -m pytest -q -p no:cacheprovider`. Исторический исходный пакет stage15 сохранён; текущий пакет stage16 проверяется командами ниже; он не включает обученные веса и приватные входы.

**Stage14 — один фиксированный эксперимент на temporal change:** все 40 уникальных клипов, те же 5 folds stage13. Четыре скалярных канала изменений SigLIP относительно собственного фона (не optical flow и не appearance-координаты), сильная L2=1, MIL pooling, порог 0; без перебора моделей/порогов. **OOF positive F1 0.5854, macro-F1 0.5747, precision 0.600, recall 0.5714, AUC 0.5514; TP12/FN9/FP8/TN11.** Stage13 на тех же folds: positive F1 0.400, macro-F1 0.4667, AUC 0.4085. Duration baseline сильнее: positive F1 0.7347, macro-F1 0.6577; always-positive F1 0.6885. Локализация ухудшилась: 2/21 endpoint-hit против 11/21 stage13; exact 0/2 при нулевом покрытии сеткой. Поэтому это адаптивное частичное улучшение классификации, не GO. Независимого holdout нет, production fail-closed.

Stage14 сохранена отдельно в `local_private/research_models/stage14_change/`: веса, CV weights, config, manifests, сравнение и проверка. SHA-256 весов `3a549069bd6c65ec74d93ad23539ab2e8667f198a56cff1e907225b951bfda91`. Все шесть fit сошлись, reload точный; fold-only scaler воспроизведён с нулевой разницей. Stage13 и её пакет не изменены. См. [протокол stage14](docs/PROTOCOL_STAGE14.md).

**Техническая платформа:** локальный HTTP API, очередь, отдельные процессы заданий, таймауты, проверка входов и fail-closed правила. Полный локальный прогон 2026-10-18 после stage14: **185 passed**, одно существующее предупреждение Starlette/httpx (`python -B -m pytest -q -p no:cacheprovider`). Добавлены синтетические проверки градиента и инвариантности к статическому фону. Это не подтверждает качество обнаружения нарушений или соответствие условиям кейса.

Проверен путь HTTP upload → очередь → настоящий дочерний процесс → OpenCV → синтетический детектор → два события, строгий ответ и локальные JPEG evidence; проверены авторизация и очистка временных файлов. Отдельный тест запускает закреплённый YOLOX-S ONNX на синтетическом пустом клипе и получает `failed/too_few_detections`, без ложного `no_violation`. В исходном пакете без весов этот ONNX-тест намеренно **skip**; остальные тесты не требуют приватного датасета. Эти проверки не используют реальные видео.

**Stage13 — all40, предыдущее сохранённое обучение:** 40 уникальных реальных клипов, **21 positive / 19 negative**, 1300 строк SigLIP; два точных интервала дают 5 дополнительных положительных кадров. 42 физических файла не считаются независимыми примерами. Для 17 клипов переиспользован существующий full-frame cache, для 23 извлечены новые признаки тем же закреплённым encoder/preprocess. Дополнительные точные кадры также пересчитаны в full-frame view. Единый full-frame view выбран до результатов, чтобы не смешивать старый zone-view с fallback новых клипов; параметры головы и порог stage12 не изменены. Extraction занял **45.941 с**, пять CV-fit и один final fit с оценкой — **0.830 с**; все fit сошлись.

Стратифицированная CV по целым уникальным клипам, 5 folds, seed13; projection/scaler обучены только на train-fold. **OOF F1 0.400; AUC 0.4085; TP7/FN14/FP7/TN12**. Baselines на тех же folds: always-positive F1 0.6885; duration-only F1 0.7347. Локализация endpoint ±5 с: 11/21 против случайной 0.4837. Точные интервалы: 0/2, покрытие их сеткой кандидатов — 0/2. Обучающая F1 0.8947 (TP17/FN4/FP0/TN19) — **resubstitution, не валидация**. Результат не превосходит baselines и не даёт production GO. Его нельзя напрямую сравнивать со stage12 как независимое улучшение: состав данных, view и CV изменились, данные использованы адаптивно.

Новые веса/config/input manifest/CV weights/summary сохранены отдельно в `local_private/research_models/stage13_all40/`; SHA-256 весов `34fd44c2332631388532e1e7c22b83677bbcb48bbc01331ff67e0fd1075e65dc`. Reload финальной и всех пяти CV-моделей совпадает точно. Дополнительная проверка восстановила scaler/projection только по соответствующим train-folds с нулевой разницей; независимый расчёт pooling отличается не более чем на 3.56e-15. Stage12, прежние архивы и разметка не изменены. См. [протокол all40](docs/PROTOCOL_STAGE13.md).

**Stage12 — предыдущее обучение на 17 train-клипах:** 5 vendor-positive и 12 vendor-negative из исходной таблицы; локальные SigLIP-кэши реальных данных ForteBank, 656 строк, 768 конечных признаков, coverage 17/17. Хеши исходных train-файлов проверены. Два уточнённых интервала дают 5 дополнительных точных положительных кадров; остальные 3 positive с неизвестным временем участвуют в положительном clip-level loss. Синтетика в реальное обучение не входит.

Обучена новая регуляризованная temporal MIL-head поверх замороженного SigLIP2: loss по всем клипам через logmeanexp pooling плюс дополнительный loss точных кадров. Это не дообучение энкодера. Один фиксированный подход, 17 leave-one-clip-out fit и финальный fit; все сошлись. Нормализация и проекция обучены только на клипах соответствующего train-fold. **OOF AUC 0.3333; F1 0; TP0/FN5/FP2/TN10**. Baselines: always-positive F1 0.4545; duration-only LOO F1 0.5455 (знак и порог выбираются только на обучающих клипах; это не прежний nested baseline). Локализация импортированного endpoint ±5 с: 2/5, случайный выбор кандидата — 0.3856. Точная локализация: 0/2; исходная сетка 1 fps не содержит кадров внутри этих двух интервалов, поэтому точная локализация на ней невозможна. Дополнительные точные кадры используются только в обучении и не меняют оценочные кандидаты.

На обучающих клипах финального fit F1 0.8889 (TP4/FN1/FP0/TN12): **resubstitution, не валидация**. Сохранены новые `weights.npz`, веса 17 CV-folds, config, input/artifact manifests, coverage, summary и отдельная проверка reload в `local_private/research_models/stage12/`. Повторный inference сохранённых моделей на 17 train-клипах воспроизводит метрики; хеши входов и stage11 совпадают. SHA-256 новых весов: `78e4c9b5629274a1ac68b9cb683a617db3dfadb65d6d15873b20d3b6eb5e6d37`. Итог **NO-GO / exploratory only**: перенос не подтверждён, baseline не превзойдён. Train использовался повторно; permutation controls и независимая проверка в этом запуске не проводились, старые GO-требования не снижены. Holdout закрыт, production fail-closed. См. [зафиксированный протокол stage12](docs/PROTOCOL_STAGE12.md).

**Stage11 (предыдущая сохранённая модель):** локально обучена logistic head поверх фиксированного SigLIP2 encoder. Два положительных клипа, пять кадров внутри записанных интервалов; двенадцать отрицательных клипов имеют слабые provider-метки, не подтверждённые нормальные интервалы. Три неизвестных задания исключены. В 24 проверках с разделением по клипам: **AUC 0.6667; F1 0; TP0/FN2/FP2/TN10; точная локализация 0/24**. Duration baseline separability 0.875. Все fit сошлись, финальные веса сохранены локально; качество недостаточно для GO. Данные train использовались неоднократно, это не независимая валидация. На stage11 holdout не открывался; историческая оценка stage1 описана выше.

Исследовательская голова **не подключена к HTTP detector**. Production остаётся fail-closed; ни одно правило не валидировано. Подробности: [протокол stage11](docs/PROTOCOL_STAGE11.md) и [результаты](docs/QUALITY.md).

**Техническая диагностика stage11:** подтверждённой ошибки preprocessing, sampling, баланса классов, знака оценки или inference не найдено. На обучающих строках финальная голова классифицирует 5/5 положительных и 600/600 отрицательных; это resubstitution, не валидация. Суммарные веса классов равны (302.5 каждый), максимальный модуль градиента около 1.58e-14, по 50 кандидатов на клип. Результат согласуется с переобучением и слабым переносом при двух положительных группах; оснований для исправляющего fit нет. `tools.stage11_diagnose` выполняет только локальную проверку без fit, декодирования видео и сети. Проверка целостности исправлена: текущие входы сверяются с сохранённым manifest обучения, включая исходный код, а не только с заново вычисленными хешами. Веса, пороги и входы обучения не изменены; holdout на этом этапе закрыт.

**Исходный пакет:** содержит код, тесты, конфигурацию без секретов и документацию. Не содержит видео, кадров, датасета, сырых меток, весов (включая stage11), признаков, кэшей, API-ключей или рабочего окружения. Исследовательские скрипты требуют оригинальных приватных входов и намеренно не воспроизводят эксперимент из одного исходного пакета; автоматический повтор запрещён. Старый архив stage6 и demo не обновлялись.

Новый архив формируется по `tools.release_check.build(False)` и содержит manifest SHA-256 каждого исходного файла. Для отката его следует распаковать в отдельный каталог и проверить хеши; приватные данные и веса восстанавливаются отдельно из сохранённых локальных артефактов. `release_check` остаётся **NOT READY**: 0 tracked files, 26 review findings в исходниках (хеши целостности и статические UI/test-строки) и неклассифицированные рабочие файлы вне allowlist. Эти рабочие файлы в архив не входят; блокирующих находок сканера в allowlist нет. Это локальный исходный пакет, не опубликованный релиз.

Для сдачи технического API нужны отдельно согласованные URL размещения, безопасно переданный API-ключ и доступ проверяющих к Git. Пакет сам по себе не создаёт endpoint и не подтверждает соответствие кейсу.

## Проверка текущего secure-upload исходного пакета без обучения

Архив `forte_source_stage15_secure_upload_delivery_20261018.tar.gz` содержит исходники и `SOURCE_MANIFEST.json` с SHA-256 каждого файла. Рядом находятся `.sha256` и отдельная `.verification.json` с результатом тестов свежераспакованного пакета. Все веса, приватные manifests обучения, кэши и данные остаются отдельно локально; исходный архив сам по себе не запускает inference обученной головы. Stage14 не подключена к HTTP detector.

Проверка архива и распаковка в новый каталог:

```bash
shasum -a 256 -c forte_source_stage15_secure_upload_delivery_20261018.tar.gz.sha256
mkdir forte-stage15-secure-source
tar -xzf forte_source_stage15_secure_upload_delivery_20261018.tar.gz -C forte-stage15-secure-source
cd forte-stage15-secure-source
```

Проверка исходников по manifest стандартной библиотекой Python:

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("SOURCE_MANIFEST.json").read_text())
for name, expected in manifest["files"].items():
    path = Path(name)
    assert not path.is_absolute() and ".." not in path.parts
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print("Source manifest verified:", len(manifest["files"]), "files")
```

Из корня распакованного пакета, существующим Python-окружением проекта с pytest и зависимостями:

```bash
python -B -m pytest -q -rs -p no:cacheprovider
```

Веса для этой проверки не добавляются: optional YOLOX-S ONNX и real-SigLIP/stage15 upload E2E будут пропущены. В рабочем окружении оба реально выполнялись. Тесты создают только синтетические fixtures и не запускают обучение на реальных данных. Отдельная чистая установка зависимостей не проверялась. Не запускать training/extraction entrypoints `tools.stage16_persistence`, `tools.stage15_two_component`, `tools.stage14_change`, `tools.stage13_all40` или `tools.stage13_extract` для проверки пакета: они требуют приватных данных и отдельного разрешения на новый эксперимент. Упаковка не меняет сохранённые метрики, модели, split или production.

## Что это

- HTTP API (FastAPI, ASGI): загрузка клипа, опрос статуса задания, модельная информация.
- Пайплайн: этап 1 (выборка кадров, детекция людей, зона оператора, сетка движения), этап 2 (уточнение вокруг границ кандидатов), наблюдения, правила, вердикт с fail-closed поведением.
- Изоляция: каждое задание выполняется в отдельном дочернем процессе с жёстким лимитом времени, есть очередь, admission control и журнал статусов на диске.
- Локальные инструменты разметки (`tools/`): инвентаризация каталогов по анонимным ID, импорт меток, разметка в браузере на 127.0.0.1, локальный прогон и оценка на holdout. Они печатают только агрегаты (счётчики, метрики) и не выводят имена файлов, пути или текст сцен.

## Оборудование, на котором измерено

- Apple M5, 16 ГБ RAM. Путь выполнения: CPU (ONNX Runtime, `CPUExecutionProvider`).
- Число ядер: в исходных данных проекта указано 8; в обзоре `docs/REVIEW_CRITICAL.md` (пункт L12) указано 10 ядер, из них 4 производительных. Расхождение не разрешено.
- CoreML: опционально, через `DETECTOR_PROVIDERS`. Замер покадрового инференса YOLOX: 15 мс/кадр (см. [QUALITY.md](docs/QUALITY.md)). Сквозной прогон с CoreML не измерен.

## Установка

Требуется Python 3.13 (окружение `.venv` собрано на 3.13).

```bash
uv venv                                  # по данным проекта; uv не проверялся в отдельной установке
uv pip install -r requirements.txt
python -m tools.fetch_weights            # по умолчанию yolox_s; нужна сеть (GitHub release)
```

`tools/fetch_weights.py` принимает модели `yolox_s` (по умолчанию), `yolox_tiny`, `yolox_nano` или `all`, флаг `--force` перекачивает файл. Каждый файл проверяется по SHA-256 из `app/detector.py`; при несовпадении файл не сохраняется и код возврата не равен 0.

Без весов задания завершаются `failed` с кодом `model_missing`.

`pytest` в `requirements.txt` не входит. Для тестов его нужно установить отдельно: `uv pip install pytest`.

## Финальный технический handoff

Платформа, research-модели и production-качество — разные вещи. HTTP schema/auth/static errors/upload timeouts/job budgets покрыты синтетическими тестами; production-правила не валидированы и остаются fail-closed. Stage15/16 не подключены к production detector. Research API имеет проверенный синтетический путь saved bundle → provider → HTTP, теперь имеет общую API-key auth до body, bounded admission, hard child timeout/resource caps и cleanup; ограничения описаны выше. Для stage16 используется тот же исследовательский контракт `/research/stage15/predict` и provider `tools.stage16_persistence.research_predictor`; это не новая production route.

Из корня проекта, после безопасного задания `API_KEY` в окружении (значение здесь не приводится), локальный запуск платформы:

```bash
: "${API_KEY:?Set API_KEY in the local environment}"
export API_KEY
DATA_DIR="$(mktemp -d /tmp/forte-local.XXXXXX)" REQUIRE_EMPTY_DATA_DIR=1 \
  env -u ALLOW_UNVALIDATED_RULES -u SYNTHETIC_DETECTOR -u ALLOW_SYNTHETIC_DETECTOR \
  .venv/bin/python -m uvicorn app.asgi:app --workers 1 --no-proxy-headers --host 127.0.0.1 --port 8000
```

После запуска: UI `http://127.0.0.1:8000/`, health `http://127.0.0.1:8000/healthz`, API `/api/v1/analyze`. Это адреса команды выше, **не утверждение о сейчас работающем сервере**. Команда/ASGI entrypoint сверены с кодом; HTTP-пути проверены TestClient, отдельный сетевой запуск в финальном handoff не выполнялся. Обязательна только `API_KEY`; команда задаёт новый `DATA_DIR` и `REQUIRE_EMPTY_DATA_DIR=1`. Остальные defaults ниже. API использует `X-API-Key`; upload timeout 60 с, stall timeout 15 с, job budget 230 с.

Финальный regression после stage16: **213 passed**, одно существующее предупреждение Starlette/httpx. Все 8 требований платформы и зависимости объявленных extras совпадают с установленным окружением.

Зависимости: установленный environment проверен через package metadata (38 обязательных dependency edges без missing/version conflicts); `pip check` недоступен, поскольку pip в venv отсутствует. Исправлен реальный pin `opencv-python-headless`: `5.0.0` → установленный и протестированный `5.0.0.93`. Ничего не устанавливалось. Чистая установка и полный optional research stack не проверялись. Существующее предупреждение Starlette/httpx сохраняется.

## Запуск

```bash
API_KEY=<секрет> uvicorn app.asgi:app --host 127.0.0.1 --port 8000
```

`API_KEY` обязателен: без него приложение не стартует (`RuntimeError`). Запускать из корня проекта. Пути `DATA_DIR` и `RULESET` по умолчанию привязаны к корню проекта, а не к текущему каталогу.

Веб-интерфейс статуса доступен на `/` (статический `static/index.html`). Баннер показывает статус модели и правил.

Запуск за туннелем или прокси (подробности и ограничения — в [DEPLOY.md](docs/DEPLOY.md)):

```bash
API_KEY=<секрет> REQUIRE_EMPTY_DATA_DIR=1 DATA_DIR=<пустой_каталог> \
  uvicorn app.asgi:app --workers 1 --no-proxy-headers --host 127.0.0.1 --port 8000
```

`--no-proxy-headers` нужен за любым прокси на loopback. По умолчанию uvicorn доверяет `X-Forwarded-For` от `127.0.0.1` и `::1` и подменяет адрес клиента, поэтому лимиты частоты можно было бы обойти подменой заголовка. Цена флага: все клиенты туннеля делят один bucket лимитов.

## API (контракт)

Аутентификация: заголовок `X-API-Key: <API_KEY>`. Сравнение через `hmac.compare_digest`. Ключ проверяется **до чтения тела запроса**, при неверном ключе тело не читается (401).

### `POST /api/v1/analyze`

`multipart/form-data`, поле `file` (видео; `Content-Type` должен быть `video/*` или `application/octet-stream`, иначе 415).

Ответ 200:

```json
{"job_id": "<32 hex>", "status": "processing"}
```

Ошибки до постановки в очередь: 400 `empty file`, 413 `file too large`, 415 `unsupported media type`, 429 `queue full, retry later`, 429 `too many requests` (лимит частоты на IP), 429 `too many concurrent uploads`, 401 неверный ключ, 408 таймаут приёма тела, 422 `{"detail":"invalid request"}`. Тела ошибок статические и не содержат пользовательского ввода.

Не каждый клип завершается `failed/no_validated_rule`. Другие статические отказы: 400 (пустой файл), 413 (слишком большой), 415 (тип), 408 (медленная загрузка), 422 (некорректный запрос), 429 (лимиты). В задании: `corrupt_video`, `too_long` (длиннее `MAX_DURATION_SEC`), `time_budget_exceeded`. Какой код вернётся, зависит от входа и загрузки машины.

### `GET /api/v1/jobs/{job_id}`

- Задание в работе: `{"job_id": "...", "status": "processing"}`.
- Успех: `{"job_id": "...", "status": "done", "verdict": "violation" | "no_violation", "violations": [...]}`.
  - `no_violation`: `violations` пустой; `violation`: минимум одно нарушение, максимум 50.
  - Элемент `violations`: `{"type": str, "timecode_sec": число >= 0, "description": str, "evidence_frames": [1..3 элементов]}`.
  - Элемент `evidence_frames`: `{"timestamp_sec": число >= 0, "image_base64": "<JPEG в base64>"}`.
- Ошибка: `{"job_id": "...", "status": "failed", "error": "<код>"}`.
- Неизвестный или некорректный `job_id`: 404 `{"detail": "job not found"}`.

Наблюдения, метаданные и внутренние поля в публичный ответ не попадают. Результат проходит строгий валидатор `app/worker.py:validate_result` (JPEG декодируется через `cv2.imdecode`, размеры 16–4096 px, не более 2 МБ на кадр). Невалидный результат дочернего процесса даёт `failed/invalid_result`.

Коды ошибок (закрытый список `ALLOWED_ERROR_CODES`, `app/worker.py`):

| Группа | Коды |
|---|---|
| Пайплайн | `model_missing`, `model_load_failed`, `model_invalid`, `invalid_config`, `corrupt_video`, `too_long`, `too_few_frames`, `decode_error`, `no_validated_rule`, `too_few_detections`, `zone_unreliable`, `jpeg_failed` |
| Очередь и воркер | `time_budget_exceeded`, `worker_crashed`, `worker_crashed_oom_suspected`, `job_lost`, `invalid_result`, `internal_error`, `service_shutting_down`, `analysis_failed` |

`no_validated_rule` означает, что ни одно правило не активно (нет валидированных правил и не включён демонстрационный флаг). Это не результат обработки видео.

### `GET /api/v1/model`

Требует `X-API-Key`. Ответ:

```json
{
  "model": {"name", "license", "license_url", "weights_sha256_prefix", "validated",
            "providers", "input", "rules": [...], "allow_unvalidated_rules", "available"},
  "queue": {...},
  "limits": {"max_upload_mb", "max_duration_sec", "job_budget_sec"}
}
```

Поле `available` добавлено аддитивно (без изменения остальных полей). Если модель недоступна: `{"available": false, "error": "model_unavailable"}`.

### `GET /healthz`

Без ключа: `{"ok": true}`.

## Защита API (этап 6)

- Все ответы `/api/*` несут `Cache-Control: no-store`, `Pragma: no-cache`, `X-Content-Type-Options: nosniff`. Это касается и ранних ответов `_Guard` (401, 400, 408, 413, 415, 429) и обработчиков 422 и 500. `/healthz` и `/static` этих заголовков не получают.
- `/docs`, `/redoc` и `/openapi.json` отключены (404).
- Лимиты в `_Guard` (`app/main.py`) хранятся в памяти одного процесса:
  - `MAX_INFLIGHT_UPLOADS` = 4: одновременные загрузки, сверх них 429;
  - `RATE_LIMIT_PER_MIN` = 30: загрузки в минуту на IP, сверх них 429;
  - `POLL_LIMIT_PER_MIN` = 600: опросы `GET /api/v1/jobs/*` в минуту на IP. Проверка идёт до проверки ключа.
- Адрес клиента для лимитов — сокет (`scope["client"]`). `X-Forwarded-For` и `X-Real-IP` игнорируются, пока не задано `TRUST_PROXY=1` (по умолчанию выключено). Тесты проверяют, что подмена этих заголовков не обходит лимит.
- **Граница доступа к заданиям.** Ключ один и общий, поэтому любой держатель `API_KEY` читает любое задание. `BIND_JOBS_TO_KEY=1` (по умолчанию) запоминает в памяти хеш ключа, который отправил задание, и отдаёт его только тому же ключу. Это не разделение пользователей. Запись не durable: после перезапуска её нет, и задание читается любым ключом.
- `REQUIRE_EMPTY_DATA_DIR=1` (opt-in): сервис не стартует, если `DATA_DIR` не пуст. Файлы он не удаляет. Рекомендуется пустой каталог на каждый запуск.

## Переменные окружения

Все значения и значения по умолчанию взяты из кода (`app/config.py`, `app/detector.py`, `app/pipeline.py`, `app/detector_rules.py`, `app/worker.py`).

### Публичные (оператор сервиса)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_KEY` | — (обязательна) | Ключ `X-API-Key` |
| `MAX_UPLOAD_MB` | 60 | Максимальный размер загрузки |
| `MAX_DURATION_SEC` | 150 | Максимальная длительность клипа (код `too_long`) |
| `JOB_BUDGET_SEC` | 230 | Бюджет задания от момента постановки (включая ожидание в очереди). Должен быть в (0, 300) |
| `UPLOAD_TIMEOUT_SEC` | 60 | Общий таймаут приёма загрузки (408). Проверка: `UPLOAD_TIMEOUT_SEC + JOB_BUDGET_SEC < 300` |
| `RECEIVE_TIMEOUT_SEC` | 15 | Максимальная пауза между чанками тела (408) |
| `WORKERS` | 2 | Число параллельных дочерних процессов-воркеров |
| `QUEUE_LIMIT` | 20 | Дополнительная ёмкость очереди сверх `WORKERS` (лишнее → 429) |
| `DATA_DIR` | `data/jobs` | Каталог загрузок и статусов (относительный путь — от корня проекта) |
| `DETECTOR_MODEL` | `yolox_s` | Имя модели из `app/detector.py:MODELS`: `yolox_s`, `yolox_tiny`, `yolox_nano` |
| `DETECTOR_THREADS` | 4 | Потоки ONNX Runtime |
| `DETECTOR_PROVIDERS` | `CPUExecutionProvider` | Список провайдеров через запятую, например `CoreMLExecutionProvider,CPUExecutionProvider` |
| `DETECTOR_CONF` | 0.30 | Порог уверенности детектора |
| `RULESET` | `app/rules_default.json` | Путь к набору правил |
| `OPERATOR_ZONE` | — | Ручная зона оператора `x0,y0,x1,y1` в нормализованных координатах (переопределяет автоматическую оценку) |
| `ONNX_MODEL` | — | Путь к внешней модели (проверяется по SHA-256 только для известных имён) |
| `ALLOW_UNVALIDATED_RULES` | — | `1` включает правила без валидации. **Только демонстрация.** Ответ остаётся с маркером «НЕ валидировано», это не продуктовый вердикт |
| `MAX_INFLIGHT_UPLOADS` | 4 | Одновременные загрузки в `POST /api/v1/analyze`; сверх них 429 (`app/main.py`, `_Guard`) |
| `RATE_LIMIT_PER_MIN` | 30 | Загрузки в минуту на IP (token bucket); сверх них 429 |
| `POLL_LIMIT_PER_MIN` | 600 | Опросы `GET /api/v1/jobs/*` в минуту на IP; сверх них 429 |
| `BIND_JOBS_TO_KEY` | 1 | Задание привязано к хешу ключа в памяти; `0` отключает привязку |
| `REQUIRE_EMPTY_DATA_DIR` | — | `1`: сервис не стартует, если `DATA_DIR` непуст. Никогда не удаляет файлы |
| `TRUST_PROXY` | — | `1` доверяет первому `X-Forwarded-For`. По умолчанию заголовки игнорируются |

### Тестовые и служебные (не для продукции)

| Переменная | Назначение |
|---|---|
| `SYNTHETIC_DETECTOR` | Синтетический детектор для тестов. `app/asgi.py` отказывает в старте, если он задан, кроме случая `ALLOW_SYNTHETIC_DETECTOR=1` |
| `ALLOW_SYNTHETIC_DETECTOR` | Явное разрешение на синтетический детектор (только при тестах и демо) |
| `DETECTOR_SPIN` | `1` оставляет busy-wait потоков ONNX Runtime (по умолчанию выключен) |
| `SAMPLE_FPS` (4), `JOB_ESTIMATE_SEC` (60), `ADMISSION_MARGIN_SEC` (20), `KILL_GRACE_SEC` (2), `JOB_TTL_SEC` (7200), `ORPHAN_MAX_AGE_MIN` (10), `WORKER_MEM_MB` (0, только Linux) | Параметры очереди и выборки |
| `LOCAL_PRIVATE_DIR` | Переопределяет `local_private/` для инструментов. **Известное ограничение:** `tools/import_labels.py` эту переменную не учитывает |
| `JOB_SETTINGS`, `JOB_DEADLINE`, `JOB_PARENT_PID` | Внутренние переменные для дочернего процесса воркера. Задаёт родитель, оператор их не выставляет |

## Тесты и проверки

```bash
.venv/bin/python -m pytest -q tests -p no:cacheprovider
```

- Результат: **172 passed, 0 failed, 0 skipped** (прогон 2026-10-18). В него входят 38 тестов `tests/test_hardening.py`: заголовки, отключённые `/docs`, лимиты, пути вне API, граница заданий, `REQUIRE_EMPTY_DATA_DIR`, подмена `X-Forwarded-For`.
- Самотест интерфейса разметки `python -m tools._selftest_ui`: 16/16; `python -m tools._selftest_audit`: 1/1.
- Самотест этапа 4: 3/3. Самотест этапа 5 (`tools/stage5_selftest.py`): 5/5.
- Самотест этапа 7 (`tools/stage7_selftest.py`, синтетика): 10/10. Итог регрессии этапа 7: pytest 172/0/0, самотесты 16/16, 1/1, этап 4 3/3, этап 5 5/5, этап 7 10/10.
- Стресс-тест очереди (синтетика, тестовый детектор) — PASS:
  ```bash
  SYNTHETIC_DETECTOR=1 python -m tools.stress_queue 24 8
  ```
  Аргументы: число заданий, бюджет в секундах. Подробности: [QUALITY.md](docs/QUALITY.md).
- Живой smoke-тест из другого рабочего каталога (CWD) — PASS (данные регрессионной проверки координатора).

## Локальная разметка на собственных данных

Это ручной сценарий. Все шаги выполняются локально. Ни один шаг не выводит имена клипов, пути файлов, метки или текст сцен. Список кандидатов на месте не публикуется.

1. **Инвентаризация (только метаданные).**
   ```bash
   python -m tools.inventory ~/Downloads
   ```
   Печатает анонимные ID кандидатов (например, `C01`, `C02`) и счётчики по расширениям (allow-list, остальное в `other`). Соответствие «ID → каталог» записывается в `local_private/inventory_map.json` (права 0600, в `.gitignore`).
   - Посмотреть соответствие можно только у себя локально: `cat local_private/inventory_map.json`. **Никому не передавать этот файл и его содержимое.**
2. **Проверка кандидата (изолированный probe).**
   ```bash
   python -m tools.inventory ~/Downloads --probe C02
   ```
   Печатает агрегаты: число клипов, fps, длительности, разрешения, размеры.
3. **Импорт меток.** Профиль (только счётчики по колонкам) и импорт:
   ```bash
   python -m tools.import_labels --candidate C02 --profile
   python -m tools.import_labels --candidate C02 --import --auto-ru
   ```
   `--auto-ru` включает фиксированный разбор русских формулировок (`нарушение` / `нет нарушения`, секунды). Импортированные метки получают статус **`imported_unconfirmed`**. Они не являются эталоном и ничего не валидируют. Ячейки, которые не распознаны однозначно, идут на ручной разбор.
4. **Разметка в браузере.**
   ```bash
   python -m tools.annotate            # http://127.0.0.1:8790 (порт: --port)
   ```
   Сервер слушает только `127.0.0.1` и отвергает неверный `Host` (403). В интерфейсе:
   - подтвердить или исправить импортированные события и статусы;
   - для каждого клипа отметить статус: `unreviewed` (не проверен), `reviewed_no_violation` или `reviewed_violation`. Пустой список событий на непроверенном клипе **не является** отрицательной меткой;
   - создать разбиение «клип целиком» на train / holdout (детерминированно по хешу, `local_private/splits/C02.json`).
5. **Локальный прогон** (тот же путь, что и API, с теми же правилами и лимитами):
   ```bash
   python -m tools.run_local --candidate C02 --split holdout --budget-sec 230 [--dry-run]
   ```
   `--dry-run` печатает только число клипов без запуска анализа. `--budget-sec` допускает значения 5..3600.
   Пишет `local_private/predictions/<sha16>.json` и `local_private/observations/<sha16>.json`. Печатает только счётчики и статистику времени. Каждый клип выполняется в своём дочернем процессе с жёстким бюджетом.
   Флаг `--split holdout` берёт только holdout-клипы; без него прогоняются все. Время для реального набора **не измерено**.
5a. **Минимальный аудит (67 решений, около 7 минут).** В `python -m tools.annotate` откройте кандидата C02 и карточку «Аудит»:
   - «Загрузить моменты» — 10 слепых моментов: 5 с меткой и 5 контрольных. Теги скрыты. Идентификаторы моментов стабильны (хеш от sha, t и pid); ответы привязываются к паре (sha, t) и не переписываются. На каждом моменте 5 полей: q1 (людей в рабочей зоне: 0 / 1 / 2+), q3 (банкноты на стойке видны), q7 (клиент у стойки), T1 «Оцените момент» (начало / середина / конец события / не событие / не видно), V1 «Нарушение видно в кадре» (видно / частично / не видно / не могу сказать). Клавиши 1–9 выбирают вариант.
   - Контрольные моменты (стратегия в docstring `tools/annotate_audit.py`): a — тот же клип, не ближе 20 с к метке (парный); b — другой train-клип с тем же статусом импорта (непарный); c — другой train-клип с отрицательным статусом (непарный). Парный контроль типа a есть только у 1 из 5 меток, поэтому парный знаковый тест по этим данным **невозможен**. Контроли b и c — непарный фоновый материал без статистического вывода.
   - «Роль в клипах» — R1: кассир в нижней / верхней / левой / правой части кадра, или не видно, для 17 train-клипов. Это грубый квадрантный вариант: поля зоны оператора в экспорте наблюдений нет.
   - Итого 10 × 5 + 17 = 67 решений. Аудит **завершён: 67 из 67**, все ответы валидны. Агрегаты и вывод — в [QUALITY.md](docs/QUALITY.md#3c-минимальный-аудит-c02--завершён-2026-10-18).
   - Сводка только счётчиками: `GET /api/c/C02/attrs/audit_summary` (поля `progress`, `r1`, `pairing`). Подготовка моментов и R1 (только счётчики): `.venv/bin/python -m tools.annotate_audit C02`; уже сохранённые ответы при этом не меняются. После обновления кода перезапустите `python -m tools.annotate` и перезагрузите открытую вкладку браузера.
   - Правило решения (координатор): если ≥ 2 из 5 меток отвечают «не видно» или «частично», либо T1 не соответствует началу события у ≥ 2 из 5, либо R1 показывает, что зона не совпадает с кассиром у большинства клипов, то это было бы указанием на семантику меток и роли (условие правила, не вывод). Иначе выбирается целевой атрибут. По итогу аудита выполнены условия T1 (не «начало» у 5 из 5 меток) и V1 («не видно» или «частично» у 5 из 5). Эти агрегаты ограничивают интерпретацию, но не устанавливают причину неудачи: ни семантика меток, ни признаки этим не доказаны. Результат описательный и статистического вывода не даёт.
   - Полная панель из 35 подсказок × 8 вопросов («Атрибуты момента») сохранена как необязательная: её можно использовать, если аудит покажет, что нужна большая проверка.
5b. **Этап 2 — пробный анализ признаков на train (только train):**
   ```bash
   python -m tools.train_probe --candidate C02
   ```
   Отказывается работать с holdout. Печатает только агрегаты. Результаты и методика — в [QUALITY.md](docs/QUALITY.md#3a-этап-2--анализ-на-train-только-train-holdout-не-расходовался).
5c. **Этап 5 — SigLIP2 (замороженный энкодер) + временные признаки + регуляризованная логистика, только train: NO-GO** (клип F1 0.300, событие F1 0.000, hit 0.200; базовые always-positive 0.4545, длительность 0.600 как базовый уровень). Протокол зафиксирован, самопроверка 5/5, независимое ревью пройдено, holdout не одобрен. Подробности и ограничения: [QUALITY.md](docs/QUALITY.md), раздел 3e.
6. **Оценка** на holdout (только числа, правила не включаются и не валидируются). Два режима:
   - по ручной разметке (эталон — записи проверки):
     ```bash
     python -m tools.evaluate --labels local_private/labels/C02.jsonl \
         --split local_private/splits/C02.json --predictions local_private/predictions \
         [--tolerance-sec 5] [--min-holdout 5]
     ```
   - по поставленной таблице (только оценка, без ручной проверки; заголовок вывода: «по поставленной XLSX, без ручной проверки»; человеческие записи игнорируются; дубликаты по sha16 схлопываются; правило не валидируется):
     ```bash
     python -m tools.evaluate --source imported_dataset --cid C02
     ```
   Непроверенные клипы исключаются из знаменателей. Наблюдения (accept/reject) не считаются эталоном.

Подробные условия, при которых правило можно пометить как валидированное, описаны в [docs/QUALITY.md](docs/QUALITY.md).

**Этап 8 (2026-10-18):** локальная интервальная разметка (`tools/annotate_intervals*`, 96/96) и предрегистрированный гейт достаточности дали лишь малый объём данных; измеренный результат единственного gate-only запуска (lock 5c94785af7be6620): `STOP_INSUFFICIENT_DATA`, P = 2 < 4, C = 0 < 3; обучение по протоколу не разрешено. Подробности: [QUALITY.md, раздел 3g](docs/QUALITY.md).

**Этап 10 (2026-10-18):** предрегистрированный exploratory weakly-supervised fit на train, один запуск, не GO и не независимая валидация; измеренный результат: `STOP_NOT_EVALUABLE`, обучения не было, весов нет; причина в несовпадении временной выборки меток и кэша 1 fps (см. раздел 3i). Дизайн и ограничения: [QUALITY.md, раздел 3i](docs/QUALITY.md).

**Этап 9 (2026-10-18):** попытка exploratory дообучения малой головы не выполнена: `NO_TRAINABLE_SUPERVISION` (0 негативных строк при утверждённой маске, 2 позитивных интервала). Код DRAFT / NOT RUN, метрик и весов нет. См. [QUALITY.md, раздел 3h](docs/QUALITY.md).

## Перед демонстрацией

- Время одного клипа зависит от загрузки машины. Координатор замерил этап 1 для клипа 145 с 1080p: 131 с под нагрузкой против 66 с на тихой машине.
- Рекомендация: `WORKERS=1` или `DETECTOR_THREADS=2`, либо `DETECTOR_MODEL=yolox_tiny`, либо `SAMPLE_FPS=3`. Перед демонстрацией один прогон засечь на тихой машине.
- Бюджет: загрузка 60 с + задание 230 с = 290 с < 300 с (проверка в `app/config.py`).
- Демо-видео `local_private/demo/forte_demo.mp4` (128 с, 1280×720, 30 fps): пояснительный синтетический клип с реальным локальным smoke через API. Это не запись приложения. Файл в `.gitignore`, прикладывается к сдаче отдельно. Создан `tools/demo_video.py`.

## Что сделано автономно и что заблокировано на пользователе

### Сделано автономно (подтверждено кодом и журналом проекта)

- Пайплайн на YOLOX-S (ONNX, декодирование проверено по официальной логике демо), без HOG; наблюдения отделены от правил; вердикт только из активных правил; при отсутствии активного правила задание `failed/no_validated_rule`.
- Изоляция заданий: дочерний процесс, SIGKILL группы процессов на `submit + budget`, watchdog, admission control (429), очистка только собственных файлов, журнал статусов на диске, строгий валидатор результата.
- ASGI-защита: ключ до тела, потолок байт, 408/413 со статическим телом.
- Инструменты `inventory`, `import_labels`, `annotate`, `run_local`, `evaluate`, `fetch_weights`, `stress_queue`, `make_synthetic`.
- Нормативные кандидаты: редакция текста проверена по официальному источнику на дату 2026-10-17 (см. [NORMS.md](docs/NORMS.md)). Все пункты остаются **кандидатами**.

### Заблокировано на пользователе (видео не может это установить)

1. **Подтвердить, что C02 — набор хакатона.** Пользователь указал C02 как наиболее вероятный набор. По метаданным: 42 клипа, одно разрешение 464x848, длительности 11.6–142.5 с, fps не ниже 29; рядом лежит одна таблица, в которой 42 строки совпадают с именами клипов. Окончательное подтверждение остаётся за пользователем.
2. **Подтвердить или исправить импортированные метки в интерфейсе разметки.** Для каждой импортированной записи: подтвердить, исправить или отклонить. Пока этого нет, метки остаются `imported_unconfirmed`.
3. **Подтвердить, что правила применимы.** Видео не может установить роль людей (оператор, клиент), тип операции и момент окончания обслуживания. Эти условия для кандидатных норм нужно подтвердить вручную. Без подтверждения правила остаются кандидатами.
4. **Разметить каждый клип** (`reviewed_*`) и создать разбиение train / holdout до любой настройки правил.
5. **Минимальный аудит выполнен (67 из 67; см. шаг 5a).** Следующий шаг (этап 4) ведёт координатор.

Ничего в продакшне не меняется, пока правило не валидировано на независимом наборе. Текущий итог: см. [QUALITY.md](docs/QUALITY.md).

До выполнения шагов 1–4 и шага 5 (`run_local`) с последующей оценкой сервис не может выносить вердикты.
