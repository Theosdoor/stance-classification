# Classifier Evaluation Walkthrough

## Summary

Evaluated the stance classifier using:
1. **W&B Run Statistics** - Analyzed 35 hyperparameter sweep runs
2. **Saved Checkpoint Evaluation** - Loaded best BERTweet model and ran inference on dev set

---

## W&B Sweep Results

### Best Runs by Validation F1

| Run Name | Best Val F1 | Model | LoRA Targets | LR | Batch | LoRA r | LoRA α |
|----------|-------------|-------|--------------|-----|-------|--------|--------|
| skilled-sweep-13 | **0.5795** | bertweet | attention | 5e-5 | 8 | 8 | 64 |
| jolly-sweep-12 | 0.5616 | bertweet | all_linear | 5e-5 | 8 | 8 | 64 |
| curious-sweep-25 | 0.5608 | bertweet | all_linear | 3e-5 | 8 | 8 | 64 |
| laced-sweep-10 | 0.5594 | bertweet | all_linear | 5e-5 | 8 | 8 | 64 |

### Key Findings from Hyperparameter Search

```
Model Performance Comparison:
┌────────────┬────────┬────────┬───────┐
│ Model      │ Mean   │ Std    │ Count │
├────────────┼────────┼────────┼───────┤
│ BERTweet   │ 0.5389 │ 0.0236 │ 19    │
│ DeBERTa    │ 0.4379 │ 0.0720 │ 6     │
└────────────┴────────┴────────┴───────┘

LoRA Target Comparison:
┌────────────┬────────┬────────┬───────┐
│ Targets    │ Mean   │ Std    │ Count │
├────────────┼────────┼────────┼───────┤
│ all_linear │ 0.5299 │ 0.0304 │ 20    │
│ attention  │ 0.5068 │ 0.0967 │ 3     │
│ minimal    │ 0.3733 │ 0.0355 │ 2     │
└────────────┴────────┴────────┴───────┘
```

> [!IMPORTANT]
> **BERTweet significantly outperforms DeBERTa** (+10 F1 points) on this Twitter data. This makes sense as BERTweet was pre-trained on Twitter data.

---

## Saved Model Evaluation

The saved checkpoint is a **BERTweet model with LoRA** (all_linear targets, r=8, α=64).

### Classification Report (Dev Set, n=256)

```
              precision    recall  f1-score   support

     support       0.43      0.48      0.45        46
        deny       0.21      0.40      0.28        10
       query       0.64      0.82      0.72        28
      comment       0.85      0.74      0.80       172

    accuracy                           0.69       256
   macro avg       0.53      0.61      0.56       256
```

### Confusion Matrix

```
              Pred:support  Pred:deny  Pred:query  Pred:comment
True:support            22          5           3            16
True:deny                3          4           1             2
True:query               1          0          23             4
True:comment            25         10           9           128
```

---

## Per-Class Analysis

### Support Class (n=46)
- **Accuracy: 47.8%** ❌
- Many support examples get confused with comment (16 errors)
- The model struggles to distinguish supportive agreement from neutral commentary

### Deny Class (n=10)  
- **Accuracy: 40.0%** ❌
- Severe data imbalance (only 10 samples!)
- Often confused with support (3) - the exact opposite stance

### Query Class (n=28)
- **Accuracy: 82.1%** ✅
- Best performing minority class
- Question marks likely provide strong signal

### Comment Class (n=172)
- **Accuracy: 74.4%** ⚠️
- Dominant class but 25 examples incorrectly labeled as support
- Model appears to have an "over-support" bias

---

## Example Predictions

````carousel
### ✅ Correct: Query Detection
**Source:** BREAKING: 148 feared dead in crashed #Germanwings flight...  
**Reply:** @foxandfriends As long as plane stayed in air... no mention of possible terrorists on board?

**Ground Truth:** QUERY  
**Prediction:** QUERY (92.8% confidence)

The model correctly identifies the rhetorical question.
<!-- slide -->
### ✅ Correct: Support Detection  
**Source:** BREAKING: 148 feared dead in crashed #Germanwings flight...  
**Reply:** "@foxandfriends: BREAKING: 148 feared dead..." Very tragic!! #Airbus320 #GermanWingsCrash

**Ground Truth:** SUPPORT  
**Prediction:** SUPPORT (82.4% confidence)

Retweeting with affirmation correctly identified.
<!-- slide -->
### ❌ Error: Deny → Support
**Source:** Latest on #Germanwings crash: Pilots signaled 911 before dropping out of midair...  
**Reply:** "@mashable: Latest on #Germanwings crash..." dropping out of...no...irresponsible reporting

**Ground Truth:** DENY  
**Prediction:** SUPPORT (78.9% confidence)

The word "no" indicating denial was not caught. Quote-tweet format may confuse the model.
<!-- slide -->
### ❌ Error: Comment → Deny
**Source:** RT @khjelmgaard: German media reporting #AndreasLubitz had a 'serious depressive episode'...  
**Reply:** @tinkalee_12 My 2nd source is confirm 4 me French media is reporting...

**Ground Truth:** COMMENT  
**Prediction:** DENY (81.7% confidence)

The informal language and source verification is misread as denial.
````

---

## 🎯 Improvement Suggestions

### High Priority

| Area | Suggestion | Expected Impact |
|------|------------|-----------------|
| **Data Augmentation** | Use back-translation or PHEME dataset for more Deny samples | +5-10% Deny F1 |
| **Focus on Support/Comment** | Add explicit markers or use contrastive learning | +3-5% macro F1 |
| **Quote-Tweet Handling** | Detect and handle "@user: QUOTE" tweet format specially | Reduce Deny→Support errors |

### Medium Priority

| Area | Suggestion |
|------|------------|
| **Focal Loss** | Replace weighted CE with focal loss (γ=2) for minority classes |
| **Label Smoothing** | Use 0.1 smoothing to reduce overconfidence |
| **Input Markers** | Add `[SOURCE]` and `[REPLY]` special tokens |
| **Higher LoRA Rank** | Try r=32 or r=64 for more expressivity |

### Lower Priority

| Area | Suggestion |
|------|------------|
| **Ensemble** | Combine BERTweet + DeBERTa predictions |
| **Thread Context** | Include parent reply context |
| **Emoji Processing** | Install `emoji` package for better emoji handling |

---

## Files Modified

- Created [evaluate_clf.py](file:///Users/Subspace_Explorer/Documents/Programming/NLP_cswk/evaluate_clf.py) - Comprehensive evaluation script

---

## Commands to Reproduce

```bash
cd /Users/Subspace_Explorer/Documents/Programming/NLP_cswk
source .venv/bin/activate
python evaluate_clf.py
```