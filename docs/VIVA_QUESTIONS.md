# Viva Preparation — 20 Important Questions & Answers

Simple, confident answers you can give in a college project viva.

---

**1. What is credit-card fraud detection?**
It's the process of identifying transactions that were made without the
genuine cardholder's authorization. Here, we treat it as a supervised
Machine Learning problem: given past labeled transactions, train a model
to predict whether a new transaction is genuine or fraudulent.

**2. Why is this a classification problem?**
Because the output we want is one of two discrete categories — Genuine
(`Class = 0`) or Fraudulent (`Class = 1`) — not a continuous number. That
makes it a **binary classification** problem.

**3. What is class imbalance?**
It's when one class vastly outnumbers the other. In our dataset, genuine
transactions are about 578 times more common than fraudulent ones
(~99.83% vs ~0.17%). This makes learning to recognize the minority class
(fraud) much harder.

**4. Why is accuracy not enough?**
Because a model that predicts "genuine" for every single transaction
would already score around 99.8% accuracy here — while catching zero
fraud. Accuracy hides how the model performs on the rare, important
class, so we use precision, recall, F1-score, and ROC-AUC instead.

**5. What is SMOTE?**
SMOTE (Synthetic Minority Over-sampling Technique) creates new, synthetic
examples of the minority class (fraud) by interpolating between existing
fraud cases and their nearest fraud neighbors in feature space — rather
than just duplicating existing rows. This gives the model more varied
fraud examples to learn from.

**6. Why should SMOTE only be applied to training data?**
If SMOTE is applied before splitting into train/test, synthetic points
derived from a fraud case can end up in both the training and test sets.
The model would then be partly "tested" on data very similar to what it
trained on, giving falsely optimistic results — a form of **data
leakage**. Applying SMOTE only after splitting, and only to the training
set, keeps the test set a fair, untouched measure of real-world
performance.

**7. What is Logistic Regression?**
A linear classification algorithm that models the probability of a class
using a weighted sum of the input features passed through a sigmoid
function. It's simple, fast, and a good interpretable baseline.

**8. What is Random Forest?**
An ensemble learning method that builds many decision trees on random
subsets of data and features, then combines their predictions (by
majority vote / averaged probability). It generally handles non-linear
patterns better than a single linear model.

**9. Why did you use Random Forest?**
Because fraud patterns in anonymised PCA features (`V1`–`V28`) are
unlikely to be purely linear. Random Forest can capture non-linear
interactions between features and, in our evaluation, achieved a higher
F1-score and ROC-AUC on the fraud class than Logistic Regression, so it
was automatically selected as the best model.

**10. What is precision?**
Precision = TP / (TP + FP). Of all the transactions the model *flagged*
as fraud, what fraction were actually fraud? High precision means few
false alarms.

**11. What is recall?**
Recall = TP / (TP + FN). Of all the transactions that were *actually*
fraud, what fraction did the model catch? High recall means few missed
fraud cases.

**12. What is F1-score?**
The harmonic mean of precision and recall: `F1 = 2 * (P * R) / (P + R)`.
It balances the two — useful when you care about both false alarms and
missed fraud, which is exactly the fraud-detection situation.

**13. What is ROC-AUC?**
ROC-AUC is the area under the Receiver Operating Characteristic curve,
which plots True Positive Rate vs False Positive Rate at every possible
decision threshold. An AUC of 1.0 is a perfect classifier; 0.5 is no
better than random guessing. It measures how well the model ranks fraud
transactions above genuine ones, independent of a specific threshold.

**14. What is a confusion matrix?**
A table comparing actual vs predicted classes, broken into four counts:
True Negatives, False Positives, False Negatives, and True Positives. It
shows exactly what kind of mistakes the model is making, not just an
overall score.

**15. What are TP, TN, FP, and FN?**
- **TP (True Positive):** fraud correctly predicted as fraud.
- **TN (True Negative):** genuine correctly predicted as genuine.
- **FP (False Positive):** genuine incorrectly predicted as fraud.
- **FN (False Negative):** fraud incorrectly predicted as genuine (missed fraud).

**16. Why is fraud recall important?**
Because a False Negative (FN) means real fraud slips through undetected —
usually the most costly kind of mistake for a bank or cardholder. High
recall on the fraud class means the model is catching most real fraud,
even if it means flagging a few extra genuine transactions for review.

**17. What is feature scaling?**
Transforming numeric features so they're on a comparable scale (here,
using `StandardScaler`, which centers each feature to mean 0 and standard
deviation 1). Some algorithms (like Logistic Regression) are sensitive to
feature scale, so scaling helps them train properly and converge faster.

**18. What is data leakage?**
When information from outside the training data — often from the test
set, or from the future — accidentally influences model training,
producing evaluation results that look better than the model would
actually achieve in the real world. In this project we avoid it by:
fitting the scaler only on training data, and applying SMOTE only after
the train/test split.

**19. How is the trained model integrated with Streamlit?**
`train_model.py` is run once from the command line, producing a saved
model (`fraud_detection_model.pkl`), a saved scaler (`scaler.pkl`), and
metadata about feature order. The Streamlit app (`streamlit_app.py`)
loads these saved files at startup (cached, so it doesn't reload on every
interaction) and uses them to preprocess new input and generate
predictions — it never retrains the model itself.

**20. What are the limitations of this project?**
It's trained on one historical dataset and may not reflect current fraud
patterns without retraining. The `V1`–`V28` features are anonymised via
PCA, so we can't explain predictions in plain business terms. The demo UI
requires manually entering already-anonymised feature values, which a
real production pipeline would compute automatically. And it's an
educational project, not a certified, production-grade banking fraud
system.
