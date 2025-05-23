## KDID-Net
Code and resources for **KDID-Net**, a motion deblurring model for plant disease classification using knowledge distillation.

### 📁 Datasets

#### [[PlantVillage](https://drive.google.com/file/d/1JtOzI9LVij1rkU71AncbnR4uORQJEyJB/view?usp=sharing)]  [[FieldPlant](https://drive.google.com/file/d/1XP1ECzXdsK9ntAt5IRPjSxpl6eiKrw0d/view?usp=sharing)]


### 📌 Teacher weights

#### [PlantVillage]
##### 1st fold  [[link](https://drive.google.com/file/d/10Ao2gQiKzyhzEEQhM4_HsnhiccpsvJtW/view?usp=sharing)]  2nd fold  [[link](https://drive.google.com/file/d/18EeFr2MQI8Q2vYXTrxU-s_OKHW6zL8Cl/view?usp=sharing)]

#### [FieldPlant]
##### 1st fold  [[link](https://drive.google.com/file/d/1cVYdmdWUDC1yRIn_VUSCc4do5EEFeb9I/view?usp=sharing)]  2nd fold  [[link](https://drive.google.com/file/d/1Txi31udu3UszKmRMJW6KPp5FNiwm4Hq6/view?usp=sharing)]

### 📌 Training
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


