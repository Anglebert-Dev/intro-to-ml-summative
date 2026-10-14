# Predicting At-Risk Learners on a Digital Learning Platform

**Traditional machine learning vs deep learning for early identification of learners likely to fail or withdraw.**

Author: Anglebert Shumbusho Ishimwe · Introduction to Machine Learning, Summative · African Leadership University

| Resource | Link |
|---|---|
| GitHub repository | https://github.com/Anglebert-Dev/intro-to-ml-summative |
| Google Drive (raw dataset and saved models) | https://drive.google.com/drive/folders/1eeBOf7BjKl9XSUJ1c-37NlpVUqoZPoJK |
| Video presentation | |

## Problem
Free online learning platforms widen access to education, but many learners fail or drop out before anyone notices. Mentors are few, so they need to know early which learners need help.

This project predicts, using only data from the **first 60 days of a course**, whether a learner is **at risk** (will fail or withdraw) or **on track** (will pass or pass with distinction).

**Mission link:** the project supports my mission of using technology to make quality education accessible to underprivileged children in Rwanda and across Africa. An early-warning model lets a small mentoring team reach the learners most likely to fall behind.

## Dataset
- **Open University Learning Analytics Dataset (OULAD):** 32,593 enrolments on 22 online course presentations, with demographics, assessments and 10,655,280 daily click records.
- **Source:** UCI Machine Learning Repository, https://doi.org/10.24432/C5KK69 (licence CC BY 4.0).
- **Not stored in this repository.** The notebook downloads the original zip automatically into Google Drive; a copy is in the Drive folder above.

## Approach
1. **Cohort and leakage control:** enrolments that ended by day 60 are removed, and only data up to day 60 is used.
2. **Feature engineering:** 48 features from demographics, click behaviour (per activity type, weekly trend, inactivity) and early assessments (submission rate, scores, lateness).
3. **Split:** 70 / 15 / 15, grouped by student so no learner appears in two sets; saved with a fingerprint for reproducibility.
4. **Preprocessing:** imputation with missing-value indicators, one-hot encoding, log transform of counts, robust and standard scaling, all inside Scikit-learn pipelines fitted on training data only.
5. **Traditional ML (Scikit-learn), 9 experiments:** logistic regression, decision tree, random forest, gradient boosting, SVM, feature selection and a demographics-only ablation, tuned with grouped cross-validation.
6. **Deep learning (Keras Functional API + tf.data), 7 experiments:** learning rate, depth, dropout, L2 and batch normalisation, batch size and optimiser, each run building on the best before it.
7. **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrices, learning curves, threshold analysis, error analysis by outcome, deprivation band and module, and permutation importance.

All runs are logged to TensorBoard and to results tables in the Drive folder.

## Key results (test set)

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Tuned random forest (best traditional, chosen on validation) | 0.828 | 0.605 | 0.683 |
| Tuned gradient boosting | 0.834 | 0.639 | 0.702 |
| Network `dl_exp_06_batch_64` (best deep learning) | **0.836** | **0.633** | **0.702** |
| Random forest, demographics only (ablation) | 0.660 | 0.380 | 0.465 |

- Engineered engagement and assessment features add about 0.19 ROC-AUC over demographics alone.
- A decision threshold of about 0.33 catches 80 % of at-risk learners, at a precision of about 0.64.
- Errors are highest for the most deprived learners and for those who withdraw late, which is discussed as a limitation in the report.

## Repository contents
```
├── Summative_At_Risk_Learners.ipynb   # full pipeline with outputs and interpretations
├── requirements.txt                   # dependencies
├── .gitignore                         # keeps data, models and logs out of the repository
└── README.md
```

## How to run
**Google Colab (recommended)**
1. Open the notebook in Colab and select **Runtime > Change runtime type > GPU** (optional; CPU also works).
2. **Runtime > Run all** and allow Google Drive access.
3. The notebook creates `MyDrive/Summative_ML_Student_Success/`, downloads the dataset, and saves results, logs, figures and models there.
4. Re-running is safe: finished experiments with unchanged settings are skipped, and changed ones are retrained.

**Locally**
```bash
pip install -r requirements.txt
jupyter notebook Summative_At_Risk_Learners.ipynb
```
Change `ROOT` in section 2 to a local folder and remove the two Google Drive lines.

## Reproducibility
- Fixed random seed (42) for Python, NumPy and TensorFlow.
- Saved, grouped data split with test-set fingerprint `d3d4be18cd40`.
- Every experiment stores its settings, metrics and model; results are traceable to TensorBoard logs.

## Limitations
OULAD describes UK adult distance learners, not African children, so the model shows the method rather than a deployable tool for Rwanda. Results come from one data split, and the model is least accurate for learners from the most deprived areas.

## Citation
J. Kuzilek, M. Hlosta and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, 170171, 2017.
