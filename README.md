## Project Overview

* **Dataset:** Brain MRI Images for Brain Tumor Detection
* **Source:** [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Model Architecture:** ResNet50
* **LLM Integration:** OpenRouter

---

## Dataset Statistics

### Data Splits
| Stage / Split | Number of Images | Percentage |
| :--- | :--- | :--- |
| **Total Images** | 7,200 | 100% |
| **Training Folder (Before Split)** | 5,600 | - |
| **Training Split** | 4,760 | 85% |
| **Validation Split** | 840 | 15% |
| **Test Folder** | 1,600 | - |

### Label Distribution
| Class | Number of Images |
| :--- | :--- |
| glioma | 1,400 |
| meningioma | 1,400 |
| notumor | 1,400 |
| pituitary | 1,400 |

<img width="991" height="608" alt="Label Distribution Chart" src="https://github.com/user-attachments/assets/ba9496d8-a5b7-48da-997b-f2e1ea571b54" />

---

## Sample Images

<p>
  <img width="347" height="431" alt="MRI Sample 1" src="https://github.com/user-attachments/assets/319e5bd9-717d-4862-85e0-ffbe73a3c1eb" />
  <img width="500" height="497" alt="MRI Sample 2" src="https://github.com/user-attachments/assets/92197f03-3c25-411b-ae48-75e83085db2f" />
  <img width="217" height="206" alt="MRI Sample 3" src="https://github.com/user-attachments/assets/e336a949-7394-4f6a-9ff0-97a4d7c18589" />
  <img width="207" height="212" alt="MRI Sample 4" src="https://github.com/user-attachments/assets/0c7267f2-81ba-4a1a-a78d-4902a5af4046" />
  <img width="495" height="500" alt="MRI Sample 5" src="https://github.com/user-attachments/assets/dc739ba3-58a6-40d0-a8c8-c4929c5090d7" />
</p>

---

## Model Training & Evaluation

### Training History
**Accuracy Graph:**
<img width="717" height="530" alt="Training and Validation Accuracy" src="https://github.com/user-attachments/assets/e189cfa3-4f94-4454-9b3c-3bd9f9a21bf0" />

**Loss Graph:**
<img width="702" height="531" alt="Training and Validation Loss" src="https://github.com/user-attachments/assets/a35c4ce6-a249-4d00-a038-d2b96fa113ed" />

### Evaluation Metrics

```text
              precision    recall  f1-score   support

      glioma       0.97      0.65      0.78       400
  meningioma       0.74      0.90      0.81       400
     notumor       0.91      0.99      0.95       400
   pituitary       0.95      0.96      0.96       400

    accuracy                           0.88      1600
   macro avg       0.89      0.88      0.87      1600
weighted avg       0.89      0.88      0.87      1600
