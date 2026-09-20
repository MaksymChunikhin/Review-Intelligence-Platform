# Aspect Extraction Gold Annotation Guide

## Queue Status — 2026-08-18

The current queue is
`notebooks/03_aspects/hair_conditioners_extraction_eval_v2_review_queue.csv`.
It contains 240 rows and all 240 are still `pending`; no gold metrics exist yet.

| Component | Rows | Design | Population-weight use |
|---|---:|---|---|
| Representative | 160 | review-level stratified SRSWOR; no product cap | valid; weights sum to 112,525 |
| Targeted | 80 | deterministic alias-coverage diagnostic; cap of 2 reviews per product | none; rows are unweighted |

The representative population has 112,525 eligible holdout reviews after the
1,500 discovery reviews are excluded. Representative and targeted membership
do not overlap. The saved
`03_aspects/02_extraction_evaluation_sample.ipynb` execution validates these
counts and identities without error.

## Purpose

This guide defines the human gold standard for evaluating extraction against
the labels and aliases in `hair_conditioners_taxonomy_v1`. Taxonomy V1 is valid
as a qualitative, human-reviewed dictionary only. Its legacy
`weighted_review_support` and `weighted_population_share` fields are invalid
and must not be used to estimate prevalence. Annotate every explicitly
supported aspect in each review and copy exact supporting phrases. Do not infer
a property from the rating, product title, or outside knowledge.

The 240-review queue contains two complementary components:

- 160 representative reviews for realistic aggregate evaluation;
- 80 targeted reviews selected by alias matching to improve coverage of rare
  aspects.

The queue does not reveal which aliases caused targeted selection. This avoids
anchoring the annotator to a heuristic label that may be wrong.

The earlier V1 queue is a frozen, unannotated historical artifact and is not a
current reproduction path. V2 samples representative rows by review-level
stratified SRS; only the targeted component uses a product cap, and targeted
rows deliberately have no population weight.

## Цель

Эта инструкция задаёт правила ручной эталонной разметки для проверки извлечения
аспектов по `hair_conditioners_taxonomy_v1`. В каждом отзыве нужно отметить все
явно выраженные аспекты и дословно скопировать подтверждающие фразы. Нельзя
угадывать свойство по оценке, названию товара или внешним знаниям.

Очередь из 240 отзывов состоит из двух дополняющих друг друга частей:

- 160 репрезентативных отзывов для оценки на реалистичном распределении;
- 80 целевых отзывов, найденных по вариантам названий редких аспектов.

В таблице не показано, по какому варианту названия отзыв попал в целевую часть.
Это не позволяет автоматической подсказке навязать человеку ошибочный ответ.

## Annotation Rules

1. Read the complete review before assigning any aspect.
2. Annotate every aspect explicitly supported by the text, including positive,
   negative, neutral, and mixed statements.
3. Copy the shortest phrase that preserves the customer's complete meaning.
   The phrase must be a case-sensitive contiguous substring of `review_text`.
4. Prefer a specific accepted aspect over `overall_effectiveness`. Use overall
   effectiveness only for a generic result with no named dimension.
5. Do not create `hair_feel` or `lather`; both were excluded during review.
6. Keep boundaries distinct:
   - knots and untangling → `detangling`;
   - glide during application → `slip`;
   - container damage or leakage → `packaging_integrity`;
   - pump or sprayer operation → `dispenser_functionality`;
   - direct price level → `price_affordability`;
   - benefit relative to money → `value_for_money`;
   - color fading → `color_protection`;
   - color change or brassiness → `color_toning_deposit`.
7. A customer's health or growth statement is an observed claim, not proof of
   medical effectiveness. Preserve the exact wording without strengthening it.
8. If no approved aspect is explicitly present, set `no_supported_aspect=true`.
9. Do not use the star rating as textual sentiment or aspect evidence.

## Правила разметки

1. Перед выбором аспектов прочитайте отзыв полностью.
2. Отметьте каждый аспект, который явно выражен в тексте: положительный,
   отрицательный, нейтральный или смешанный.
3. Копируйте самую короткую фразу, которая сохраняет полный смысл покупателя.
   Фраза должна быть непрерывной частью `review_text` с тем же регистром букв.
4. Выбирайте конкретный утверждённый аспект вместо `overall_effectiveness`.
   Общую эффективность используйте только для общего результата без названного
   свойства.
5. Не создавайте `hair_feel` и `lather`: они были исключены при проверке.
6. Не смешивайте смысловые границы:
   - узлы и распутывание → `detangling`;
   - скольжение при нанесении → `slip`;
   - повреждение или протекание упаковки → `packaging_integrity`;
   - работа помпы или распылителя → `dispenser_functionality`;
   - непосредственно уровень цены → `price_affordability`;
   - польза относительно потраченных денег → `value_for_money`;
   - вымывание цвета → `color_protection`;
   - изменение оттенка или желтизна → `color_toning_deposit`.
7. Высказывание покупателя о здоровье или росте волос является его наблюдением,
   а не доказательством медицинской эффективности. Сохраняйте точную формулировку
   и не усиливайте утверждение.
8. Если в тексте нет ни одного утверждённого аспекта, установите
   `no_supported_aspect=true`.
9. Не используйте количество звёзд как доказательство аспекта или тональности
   текста.

## CSV Representation

For a completed row, `human_aspect_ids` contains a JSON list and
`human_evidence_json` contains one evidence object per aspect.

```json
["scent", "greasy_oily_finish"]
```

```json
[
  {
    "aspect_id": "scent",
    "customer_phrases": ["Smells wonderful"]
  },
  {
    "aspect_id": "greasy_oily_finish",
    "customer_phrases": ["left my hair greasy"]
  }
]
```

Set `no_supported_aspect=false` and `annotation_status=complete`. For a review
without an accepted aspect, use empty JSON lists (`[]`), set
`no_supported_aspect=true`, and mark the row complete.

## Представление в CSV

В завершённой строке поле `human_aspect_ids` содержит JSON-список, а
`human_evidence_json` — по одному объекту с доказательствами для каждого аспекта.
Подтверждающая фраза копируется из отзыва без перевода и перефразирования.

Установите `no_supported_aspect=false` и `annotation_status=complete`. Если
утверждённого аспекта нет, запишите пустые JSON-списки (`[]`), установите
`no_supported_aspect=true` и также отметьте строку завершённой.

## Evaluation Use

Report representative and targeted results separately. The representative
component supports headline extraction metrics. The targeted component is a
diagnostic stress test for rare aspects and must not be treated as a natural
prevalence sample. No extraction approach may be tuned on the final gold labels
and then evaluated on the same rows.

## Использование при оценке

Результаты репрезентативной и целевой частей считаются отдельно.
Репрезентативная часть используется для основных показателей извлечения.
Целевая часть является диагностической проверкой редких аспектов и не отражает
их естественную распространённость. Нельзя настраивать метод на окончательных
ответах этой выборки, а затем оценивать его на тех же строках.

На 2026-08-18 все 240 строк ожидают ручной разметки. До её завершения нельзя
публиковать accuracy, precision, recall, F1 или agreement как метрики V2.
