# Сравнение моделей face parsing (май 2026)

Дата: 2026-05-21  
Контекст: выбор парсера для зон **кожа, губы, глаза, волосы** (seasonal color analysis, `face-ai`).  
Бенчмарк: **CelebAMask-HQ** (19 классов), метрика **F1** (class-wise и mean), если не указано иное.

---

## Кратко о кандидатах

| Модель | Суть | Сильные стороны | Слабые стороны |
|--------|------|-----------------|----------------|
| **SegFace** (AAAI 2025) | Transformer + class-specific tokens, Swin backbone | Лучший прирост на губах, глазах, long-tail (очки, серьги); mobile-вариант ~96 FPS | Тяжелее BiSeNet; нужен PyTorch-стек |
| **FaRL / Facer** (CVPR 2022) | ViT, visual-linguistic pretrain на LAION-Face; Facer = RetinaFace + FaRL weights | Топ leaderboard (89.56 mF1); проверен в Deep Armocromia, Colorinsight; `pyfacer` | Hair слабее SegFace/DML-CSR на части протоколов; ViT медленнее CNN |
| **DML-CSR** (CVPR 2022) | Multi-task: parsing + edges + graph reasoning | Очень сильные skin/hair; хорошие границы | Ниже SegFace/FaRL по mF1 на OpenCodePapers (86.1) |
| **FaceXFormer** (ICCV 2025) | Единый transformer на 10 задач (parsing, landmarks, pose…) | ~33 FPS на все задачи сразу; не нужны отдельные модели | Parsing — одна из задач, не узкий SOTA-only парсер |
| **BiSeNet ResNet34** (yakhyo) | Классический bilateral seg, ONNX | ~295 FPS; уже в `face-ai` (`face_parsing.py`) | mF1 ~82–84, не в топ-6 публичных leaderboard |
| **SegFormer (HF)** | Fine-tune `jonathandinu/face-parsing` на CelebAMask-HQ | Быстрый старт, Transformers.js | Community-оценка ~85–87 mF1, ниже SegFace/FaRL |

---

## Сравнительная таблица

| Модель | Mean F1 (CelebAMask-HQ) | Кожа | Волосы | Глаза (L/R) | Губы (L/U) | Скорость | Классы |
|--------|-------------------------|------|--------|-------------|------------|----------|--------|
| SegFace | **88.96** (@448); 93.03 LaPa | **97.7** | **96.2** | **92.6 / 92.7** | **90.5 / 88.8** | 39–48 FPS (Swin); **96 FPS** mobile | 19 |
| FaRL / Facer | **89.56** (FaRL-B) | ~97.2–97.7 | ~93–96 | ~91.5–92 | ~87–89 | Средняя | 19 |
| DML-CSR | 86.1 (leaderboard); 92.38 (@473) | 97.6 | 96.4 | 91.8 / 91.5 | 89.9 / 88.0 | Средняя | 19 |
| FaceXFormer | competitive* | ✓ | ✓ | ✓ | ✓ | **~33 FPS** (10 задач) | 19 + др. |
| BiSeNet R34 | ~82–84* | хорошо | хорошо | средне | средне | **~295 FPS** ONNX | 19 |
| SegFormer HF | ~85–87* | хорошо | хорошо | хорошо | хорошо | Средняя | 19 |

\* Разные протоколы resolution/train split — для продакшена валидировать на своих фото.

### Per-class F1 (фрагмент, SegFace paper @512 vs baselines)

| Класс | SegFace | DML-CSR | FaRL (scratch) |
|-------|---------|---------|----------------|
| Skin | **97.7** | 97.6 | 97.2 |
| Hair | **96.3** | 96.4 | 93.1 |
| L-Eye / R-Eye | **92.6 / 92.7** | 91.8 / 91.5 | 91.6 / 91.5 |
| L-Lip / U-Lip | **90.5 / 88.8** | 89.9 / 88.0 | 89.1 / 87.2 |

---

## Рекомендации для `face-ai`

| Задача | Модель |
|--------|--------|
| Максимальная точность масок (кожа, губы, глаза) | **SegFace** @448–512 или **FaRL/Facer** |
| Проверенный стек для color analysis (как в литературе) | **Facer** (FaRL + RetinaFace) |
| Real-time / ONNX / текущий код | **BiSeNet ResNet34** (без смены) |
| Parsing + landmarks + pose в одном прогоне | **FaceXFormer** |
| Быстрый прототип | **SegFormer** на Hugging Face |

---

## Вывод

1. **По публичным бенчмаркам (май 2026)** лидируют **FaRL-B** (89.56 mean F1) и **SegFace** (88.96–89.22); SegFace сильнее на **губах, глазах** и редких классах (очки, украшения).
2. **Для seasonal color / undertone** (нужны только кожа, волосы, глаза, губы) оптимален **FaRL/Facer** — уже используется в Deep Armocromia и Colorinsight, устойчив на поворотах.
3. **Текущий `face-ai`** на BiSeNet ONNX — правильный выбор для **скорости**; апгрейд на SegFace или Facer имеет смысл, если метрики color analysis упираются в качество масок.
4. **SegFace mobile** (mF1 87.91, ~96 FPS) — компромисс, если понадобится мобильный SOTA без полной смены архитектуры.
5. **Не рекомендуется** для этой задачи: SAM/SAM2 без fine-tune под 19 классов лица — хуже специализированных парсеров на тонких зонах (губы, радужка).

---

## Источники

- [OpenCodePapers — face-parsing-on-celebamask-hq](https://opencodepapers-b7572d.gitlab.io/benchmarks/face-parsing-on-celebamask-hq.html)
- [SegFace (AAAI 2025)](https://arxiv.org/abs/2412.08647) — class-wise F1
- [FaRL (CVPR 2022)](https://arxiv.org/abs/2112.03109)
- [FaceXFormer (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Narayan_FaceXFormer_A_Unified_Transformer_for_Facial_Analysis_ICCV_2025_paper.html)
- [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) — BiSeNet ONNX
- [Facer / pyfacer](https://github.com/FacePerceiver/facer)
