# Employee_Attitude_classification_using-machine_learning
predict the positive negative  output for employee attitude. currently have 9 attitudes

# Workplace Attitude Scenario Classifier

A Python and `scikit-learn` notebook exploring text classification on workplace scenarios (`Short_Term_Outcome` and `Long_Term_Outcome`) to predict broad attitude categories (`Positive` vs. `Negative`) and return a recommendation.

## Overview
- **Libraries:** `pandas`, `scikit-learn` (`TfidfVectorizer`, `SVC`, `train_test_split`, `classification_report`, `accuracy_score`), `matplotlib`, `seaborn`
- **Class Mapping:** Maps 9 specific attitude labels (`Growth Mindset`, `Proactive`, `Resilient`, `Collaborative`, `Empathetic`, `Accountable`, `Transparent` -> `Positive`; `Defensive`, `Complacent` -> `Negative`) into `Broad_Attitude`.
- **Current Status:** Baseline prototype / work in progress.

## Results & Observations
- **Dataset Distribution:** 180 total samples (`140` Positive / `77.78%`, `40` Negative / `22.22%`).
- **Test Split (30% stratified):** 54 samples (`42` Positive, `12` Negative).
- **Performance:** Overall accuracy is **77.78%**, with `Positive` F1-score at `0.88` and `Negative` precision/recall/F1 at `0.00` using `SVC(kernel='linear', C=1.0, class_weight='balanced')` with `(1, 2)` n-grams.

## Next Improvements
- Address minority-class recall (more `Negative` examples, hyperparameter tuning, or alternative classifiers/thresholds).
- Align training text features with inference inputs (`Short_Term_Outcome` + `Long_Term_Outcome`) and expand recommendation selection beyond `.iloc[0]`.
- Clean up empty notebook cells before committing.
