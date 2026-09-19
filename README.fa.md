# پردازش زبان طبیعی فارسی برای spaCy

کتابخانه‌ی spaCy هیچ‌گاه مدل آموزش‌دیده‌ای برای فارسی منتشر نکرده است و `spacy.blank("fa")` تنها
یک توکن‌ساز و فهرست واژه‌های اضافه در اختیار می‌گذارد. این مخزن ده بستهٔ آموزش‌دیدهٔ فارسی را
فراهم می‌کند که بر پایهٔ پیکرهٔ [UD_Persian-PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT)
ساخته شده‌اند و مانند دیگر مدل‌های spaCy با pip نصب می‌شوند. این بسته‌ها در چهار ردهٔ `sm`،
`md`، `lg` و `trf` منتشر شده‌اند.

## بسته‌ها

بسته‌های `fa_dep_news_*` دربردارندهٔ `tok2vec`، `tagger`، `morphologizer`،
`trainable_lemmatizer` و `parser` هستند؛ `fa_core_news_*` همان اجزا به‌همراه `ner` را دارد و
`fa_ent_news_*` تنها `ner` را با tok2vec اختصاصی خود می‌آورد. همهٔ رده‌های `sm`، `md` و `lg`
با پروانهٔ CC BY-SA ۴٫۰ منتشر شده‌اند.

```bash
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-3.8.0-py3-none-any.whl
# یا بدون NER
pip install https://huggingface.co/Phazel/fa_dep_news_sm/resolve/main/fa_dep_news_sm-3.8.0-py3-none-any.whl
```

```python
import spacy
nlp = spacy.load("fa_core_news_sm")
doc = nlp("محمدرضا شجریان در مشهد به دنیا آمد.")
print(doc.ents)   # (محمدرضا شجریان, مشهد)
```

بسته‌های منتشرشده روی Hugging Face:
[`fa_core_news_sm`](https://huggingface.co/Phazel/fa_core_news_sm) ·
[`fa_dep_news_sm`](https://huggingface.co/Phazel/fa_dep_news_sm) ·
[`fa_ent_news_sm`](https://huggingface.co/Phazel/fa_ent_news_sm) ·
[`fa_core_news_md`](https://huggingface.co/Phazel/fa_core_news_md) ·
[`fa_dep_news_md`](https://huggingface.co/Phazel/fa_dep_news_md) ·
[`fa_ent_news_md`](https://huggingface.co/Phazel/fa_ent_news_md) ·
[`fa_core_news_lg`](https://huggingface.co/Phazel/fa_core_news_lg) ·
[`fa_dep_news_lg`](https://huggingface.co/Phazel/fa_dep_news_lg) ·
[`fa_ent_news_lg`](https://huggingface.co/Phazel/fa_ent_news_lg) ·
[`fa_core_news_trf`](https://huggingface.co/Phazel/fa_core_news_trf).
جدول‌های بردار floret جداگانه (فقط بردار، بدون هیچ مؤلفه‌ای):

```bash
# ۵۰ هزار سطر × ۳۰۰ بعد، ۴۰۰ هزار مقالهٔ نخست ویکی‌پدیای فارسی (جدول ردهٔ md)
pip install https://huggingface.co/Phazel/fa_floret_400k/resolve/main/fa_floret_400k-0.1.0-py3-none-any.whl
# ۵۰ هزار سطر × ۳۰۰ بعد، کل دامپ ویکی‌پدیای فارسی
pip install https://huggingface.co/Phazel/fa_floret_full_wiki/resolve/main/fa_floret_full_wiki-0.1.0-py3-none-any.whl
# ۲۰۰ هزار سطر × ۳۰۰ بعد، کل دامپ ویکی‌پدیای فارسی، ۵ دوره (جدول ردهٔ lg)
pip install https://huggingface.co/Phazel/fa-floret-wiki-vectors/resolve/main/fa_floret_wiki_200k-0.1.0-py3-none-any.whl
```

## کارایی

ارزیابی با `spacy benchmark accuracy` روی بخش آزمون همان پیکره انجام شده است:

| سنجه | `sm` | `md` | `lg` | `trf` | مرجع |
| --- | --- | --- | --- | --- | --- |
| `TOKEN_ACC` / `TOKEN_F` | ۹۹٫۹۶ / ۹۹٫۱۱ | ۹۹٫۹۶ / ۹۹٫۱۱ | ۹۹٫۹۶ / ۹۹٫۱۱ | ۹۹٫۹۶ / ۹۹٫۱۱ | |
| `TAG_ACC` (XPOS) | ۹۵٫۹۶ | ۹۶٫۲۵ | ۹۶٫۵۵ | **۹۷٫۶۲** | |
| `POS_ACC` (UPOS) | ۹۶٫۲۴ | ۹۶٫۶۴ | ۹۶٫۶۸ | **۹۷٫۶۳** | |
| `MORPH_ACC` | ۹۶٫۲۹ | ۹۶٫۶۴ | ۹۶٫۷۰ | **۹۷٫۸۲** | |
| `LEMMA_ACC` | ۹۷٫۹۱ | ۹۷٫۹۶ | **۹۸٫۰۸** | ۹۷٫۳۱ | |
| `SENTS_F` | ۹۹٫۲۵ | **۹۹٫۲۸** | ۹۹٫۱۸ | ۹۷٫۳۵ | |
| `DEP_UAS` | ۸۹٫۶۹ | ۹۰٫۵۲ | ۹۰٫۹۶ | **۹۳٫۸۷** | `hazm-bert-dependency-parser`: ۹۲٫۴۶ |
| `DEP_LAS` | ۸۵٫۱۵ | ۸۶٫۳۴ | ۸۶٫۶۰ | **۹۰٫۷۹** | `hazm-bert-dependency-parser`: ۸۹٫۳۴ |
| `ENTS_P` | ۷۷٫۶۷ | ۷۶٫۵۶ | ۸۱٫۵۱ | **۸۴٫۰۶** | |
| `ENTS_R` | ۶۶٫۸۷ | ۷۲٫۹۵ | ۷۱٫۰۹ | **۸۱٫۷۶** | |
| `ENTS_F` | ۷۱٫۸۷ | ۷۴٫۷۱ | ۷۵٫۹۴ | **۸۲٫۸۹** | |
| سرعت (940MX، دستهٔ ۳۲) | ۱۰٬۲۳۵ | ۹٬۰۵۸ | ۹٬۲۱۵ | ۱٬۱۰۶ | |
| حجم بستهٔ نصب | ۱۳٫۵ مگابایت | ۶۸٫۵ مگابایت | ۲۳۵٫۲ مگابایت | ۶۰۸٫۲ مگابایت | |

ردهٔ `trf` در همه‌جا جلو است مگر در بن‌واژه‌یابی و مرزبندی جمله، و تنها ردهٔ‌ای است که از مرجع
`DEP_LAS` برابر ۸۹٫۳۴ عبور می‌کند. به کارت گرافیک نیاز دارد و مدل پایهٔ آن پروانهٔ مشخصی ندارد،
پس شرایط بازانتشار وزن‌هایش نامعلوم است و هشدار آن در `meta.json` بستهٔ منتشرشده ثبت شده است
(`docs/MODELS.md` بخش ۸).

برچسب‌های موجودیت «نقره‌ای» هستند: از لایه‌ای در خود پیکره می‌آیند که با برچسب‌زن Beheshti-NER
تولید و سپس دستی اصلاح شده است. بنابراین `ENTS_F` تا اندازه‌ای هم‌خوانی با آن برچسب‌زن را
می‌سنجد؛ دیگر سنجه‌های جدول در برابر دادهٔ طلایی سنجیده شده‌اند.

آموزش روی یک پردازندهٔ چهارهسته‌ای i5-7200U و بدون کارت گرافیک انجام شده است: ۱ ساعت و ۲۷ دقیقه
برای اجزای نحوی و ۱۷ دقیقه برای NER. این دو اجرا مستقل‌اند و می‌توانند هم‌زمان انجام شوند.

## توان عملیاتی

میانهٔ چند اجرای پیاپی `nlp.pipe` روی ۱۴۶ سند بخش آزمون PerDT (۲۳٬۸۲۵ توکن). تنها زمان خودِ
`pipe` اندازه‌گیری شده و اجرای گرم‌کردن کنار گذاشته می‌شود. برای بازتولید:
`python scripts/benchmark_throughput.py <model> --gpu-id <n>`؛ دادهٔ خام در
`metrics/throughput-*.json` است.

| رده | پردازنده i5-7200U | کارت 940MX | پردازنده Xeon @ 2.00GHz | کارت Tesla T4 |
| --- | ---: | ---: | ---: | ---: |
| `sm` | ۵٬۴۸۴ | ۱۰٬۲۳۵ | | |
| `md` | ۵٬۴۰۸ | ۹٬۰۵۸ | | |
| `lg` | ۴٬۷۱۵ | ۹٬۲۱۵ | | |
| `trf` | ۱۸۷ | ۱٬۱۰۶ | ۳۳۶ | ۸٬۳۲۰ |

ردهٔ `trf` روی یک پردازنده ۲۹ برابر کندتر از `sm` است. عددهای T4 و Xeon از یک ماشین Colab
می‌آیند، یعنی شتاب ۲۵ برابری. فاصلهٔ رده‌های پردازنده‌ای کمتر از ۱۵ درصد است، پس گلوگاه
تجزیه‌گر و بن‌واژه‌یاب است نه جست‌وجوی tok2vec. پراکندگی اجراها روی لپ‌تاپ حدود ۱۰± درصد است.
اجرای `trf` روی 940MX به نسخهٔ مشخصی از torch نیاز دارد؛ بخش ۹ از `docs/MODELS.md` را ببینید.

## هشدارها

- **برخی بن‌واژه‌ها فاصله دارند.** توکن‌های چندواژه‌ای ادغام شده‌اند، پس `کتاب‌هایش` یک توکن
  با برچسب `N_IANM_PR_JOPER` و بن‌واژهٔ `کتاب او` است. این وضعیت حدود ۱٫۵ درصد توکن‌ها را
  دربرمی‌گیرد.
- **`doc.noun_chunks` کم‌کار است.** پیاده‌سازی بالادستی در
  `spacy/lang/fa/syntax_iterators.py` برچسب‌هایی از ClearNLP را می‌جوید که در Universal
  Dependencies وجود ندارند. تحلیل کامل در
  [`docs/upstream/fa-noun-chunks.md`](docs/upstream/fa-noun-chunks.md) آمده است.

گام‌های تبدیل پیکره، آموزش، ارزیابی و بسته‌بندی در [`project.yml`](project.yml) تعریف شده‌اند.
توضیح بیشتر دربارهٔ گزینش پیکره و پروانه‌ها در [`docs/MODELS.md`](docs/MODELS.md) و شرح انگلیسی
پروژه در [`README.md`](README.md) آمده است.
