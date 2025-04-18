## KDID-Net
Code and resources for **KDID-Net**, a motion deblurring model for plant disease classification using knowledge distillation.

### 📁 Datasets

#### [[PlantVillage](https://drive.google.com/file/d/1JtOzI9LVij1rkU71AncbnR4uORQJEyJB/view?usp=sharing)]  [[FieldPlant](https://drive.google.com/file/d/1XP1ECzXdsK9ntAt5IRPjSxpl6eiKrw0d/view?usp=sharing)]


### 📌 Teacher weights

#### [PlantVillage]
##### 1st fold  [[link](https://drive.google.com/file/d/10Ao2gQiKzyhzEEQhM4_HsnhiccpsvJtW/view?usp=sharing)]  2nd fold  [[link](https://drive.google.com/file/d/18EeFr2MQI8Q2vYXTrxU-s_OKHW6zL8Cl/view?usp=sharing)]

#### [FieldPlant]
##### 1st fold  [[link](https://drive.google.com/file/d/1cVYdmdWUDC1yRIn_VUSCc4do5EEFeb9I/view?usp=sharing)]  2nd fold  [[link](https://drive.google.com/file/d/1Txi31udu3UszKmRMJW6KPp5FNiwm4Hq6/view?usp=sharing)]

### Training
Teacher
```python
python train.py --config [config path]
```

Without distill
```python
python train_small.py --config [config path]
```

With distill
```python
python KD_train.py --config [config path]
```

### Experimental results
comparison of deblurring results on PlantVillage datasets
| Methods           | PSNR    | SSIM   | Accuracy | Precision | Recall  | F1 Score |
|-------------------|---------|--------|----------|-----------|---------|----------|
| No restoration    | 16.9437 | 0.4101 | 0.8593   | 0.7106    | 0.6608  | 0.6799   |
| DeblurGANv2       | 19.6649 | 0.3448 | 0.8948   | 0.7462    | 0.7064  | 0.7226   |
| MIMO-UNet         | 22.2699 | 0.4461 | 0.7997   | 0.6689    | 0.6063  | 0.6274   |
| MIMO-UNet+        | 22.5755 | 0.4529 | 0.8199   | 0.6812    | 0.6234  | 0.6435   |
| Uformer           | 22.4970 | 0.4423 | 0.7841   | 0.6549    | 0.5893  | 0.6110   |
| NAFNet            | 21.9176 | 0.4251 | 0.7618   | 0.6368    | 0.5655  | 0.5881   |
| WRA-Net           | 20.8756 | 0.4514 | 0.8713   | 0.7332    | 0.6871  | 0.7045   |
| DeepRFT           | 22.9125 | 0.5017 | 0.9178   | 0.8030    | 0.7713  | 0.7841   |
| DeepRFT+          | 22.8032 | 0.5020 | 0.8998   | 0.7775    | 0.7405  | 0.7549   |
| **KDID-Net (proposed)** | 19.8182 | 0.3774 | 0.9432 | 0.8411 | 0.8174 | 0.8275 |


comparison of knowledge distillation results on PlantVillage datasets
| Methods         | PSNR   | SSIM   | Accuracy | Precision | Recall | F1 Score |
|----------------|--------|--------|----------|-----------|--------|----------|
| Teacher        | 19.3997 | 0.3466 | 0.9485   | 0.8509    | 0.8290 | 0.8385   |
| Student        | 20.0957 | 0.3839 | 0.8574   | 0.7097    | 0.6605 | 0.6791   |
| Logits         | 19.7652 | 0.3700 | 0.9333   | 0.8321    | 0.8071 | 0.8176   |
| FitNet         | 19.7689 | 0.3814 | 0.9328   | 0.8221    | 0.7948 | 0.8063   |
| SP             | 20.1954 | 0.3931 | 0.9190   | 0.7935    | 0.7618 | 0.7749   |
| AT             | 20.3084 | 0.4011 | 0.9259   | 0.8061    | 0.7765 | 0.7888   |
| CWD            | 19.8584 | 0.3814 | 0.9223   | 0.8049    | 0.7744 | 0.7869   |
| WKD            | 19.5283 | 0.3759 | 0.9345   | 0.8231    | 0.7965 | 0.8077   |
| SRD            | 20.0554 | 0.3921 | 0.8892   | 0.7442    | 0.7037 | 0.7198   |
| MLP            | 19.9193 | 0.3928 | 0.8861   | 0.7317    | 0.6896 | 0.7065   |
| AttnFD         | 20.1079 | 0.3867 | 0.9259   | 0.8073    | 0.7778 | 0.7901   |
| **KDID-Net (proposed)** | 19.8182 | 0.3774 | 0.9432 | 0.8411 | 0.8174 | 0.8275 |


