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

comparison of kd results on PlantVillage datasets
Methods	PSNR	SSIM	Accuracy	Precision	Recall	F1 score
Teacher	19.3997	0.3466	0.9485	0.8509	0.8290	0.8385
Student	20.0957	0.3839	0.8574	0.7097	0.6605	0.6791
Logits [50]	19.7652	0.3700	0.9333	0.8321	0.8071	0.8176
FitNet [51]	19.7689	0.3814	0.9328	0.8221	0.7948	0.8063
SP [52]	20.1954	0.3931	0.9190	0.7935	0.7618	0.7749
AT [53]	20.3084	0.4011	0.9259	0.8061	0.7765	0.7888
CWD [54]	19.8584	0.3814	0.9223	0.8049	0.7744	0.7869
WKD [33]	19.5283	0.3759	0.9345	0.8231	0.7965	0.8077
SRD [55]	20.0554	0.3921	0.8892	0.7442	0.7037	0.7198
MLP [56]	19.9193	0.3928	0.8861	0.7317	0.6896	0.7065
AttnFD [57]	20.1079	0.3867	0.9259	0.8073	0.7778	0.7901
KDID-Net (proposed)	19.8182	0.3774	0.9432	0.8411	0.8174	0.8275
