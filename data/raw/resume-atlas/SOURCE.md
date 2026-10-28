# Resume Atlas (unmodified copy)

* Source: Hugging Face dataset `ahmedheakl/resume-atlas` (file `train.csv`)
  https://huggingface.co/datasets/ahmedheakl/resume-atlas
* Licence: MIT, as stated on the dataset card (see `README.md` in this folder, which is the card as downloaded).
* Original paper and code: https://arxiv.org/abs/2406.18125 and https://github.com/noran-mohamed/Resume-Classification-Dataset
  Please cite the paper if you use the data.
* Integrity: `train.csv` is byte-for-byte identical to the upstream file (checked 2026-10-23).
  SHA-256: `c71b0d6d88e75412aefb3998d289e4af8d9c07a9b26e8a974d6a2bac485480c2` (53,573,912 bytes)
* Content: 13,389 resumes in 43 categories (columns `Category`, `Text`). The text is already lower-cased and stripped of punctuation.
  The name `train.csv` is the dataset's own file name, **not** this project's training split; the project makes its own
  cross-validation folds (see `docs/METHODOLOGY.md`).

## Personal data

A read-only scan found no email addresses. About 60% of resumes contain a `555` phone number (the reserved fictional range) and
many begin with template placeholder names, so much of the text is template content. About 16% contain a 10-digit number and
about 10% a street address, and the names cannot be proven fictional. Treat the file as possibly containing personal data:
this project removes emails, phone numbers and web links before scoring, but the file itself is unredacted.
