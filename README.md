# NTUC LearningHub SkillsFuture Career Transition Programmes (SCTP) Associate Data Analyst: Capstone Project

## Heart Disease Prediction

Last Updated On 25th November 2024.

## Objective

To develop a robust binary classification machine learning model with multiple numerical and categorical features that can accurately predict the likelihood of adverse heart disease outcomes, based on a comprehensive set of patient health indicators. The model will be used to predict whether a patient is at risk of heart disease depending on multiple attributes, as early and accurate prediction of heart disease outcomes can significantly improve patient care and survival rates.

## Dataset

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:
- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations
- **Total: 1190 observations**
- *Duplicated: 272 observations*
- `Final dataset: 918 observations`

Every dataset used can be found from UCI Machine Learning Repository on the following link:
- https://archive.ics.uci.edu/dataset/45/heart+disease
- https://archive.ics.uci.edu/dataset/145/statlog+heart

### Acknowledgments

[Creators](https://archive.ics.uci.edu/dataset/45/heart+disease):
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Donor:
- David W. Aha (aha@ics.uci.edu) (714) 856-8779
- Date: July, 1988

## Conclusion To Exploratory Data Analysis Of The Whole Dataset

| Feature | Data Type | Description | Mean | Standard Deviation | Minimum & Maximum | Frequency (Proportion) | Order Or Value Range For Positive Heart Disease | Feature Encoding |
| :-: | :-: | :- | :-: | :-: | :-: | :-: | :-: | :-: |
| Age | Numerical | Age Of The Patients In Years | 53.5 | 9.4 | 28 & 77 | - | >55 | - |
| Sex | Categorical | *M : Male* <br> *F : Female* | - | - | - | M : 725 (79.0%) <br> F : 193 (21.0%) | M (90.2%) > F (9.8%) | F=0 ; M=1 |
| ChestPainType | Categorical | Chest Pain Type <br> *TA : Typical Angina* <br> *ATA : Atypical Angina* <br> *NAP : Non-Anginal Pain* <br> *ASY : Asymptomatic* | - | - | - | TA : 46 (5.0%) <br> ATA : 173 (18.9%) <br> NAP : 203 (22.1%) <br> ASY : 496 (54.0%) | ASY (77.2%) > NAP (14.2%) > ATA (4.7%) > TA (3.9%) | ASY=0 ; ATA=1 ; NAP=0 ; TA=3 |
| RestingBP | Numerical | Resting Blood Pressure On Admission To The Hospital In mmHg | 132.4 | 18.5 | 0 & 200 | - | >75 | - |
| Cholesterol | Numerical | Serum Cholesterol In mg/dl | 198.8 | 109.4 | 0 & 603 | - | 0 & >200 | - |
| FastingBS | Categorical | Fasting Blood Sugar <br> *1 : If FastingBS > 120 mg/dl* <br> *0 : Otherwise* | - | - | - | 1 : 214 (23.3%) <br> 0 : 704 (76.7%) | (FBS < 120 mg/dl) (66.5%) > (FBS > 120 mg/dl) (33.5%) | - |
| RestingECG | Categorical | Resting Electrocardiogram Results <br> *Normal : Normal* <br> *ST : Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)* <br> *LVH : Showing Probable Or Definite Left Ventricular Hypertrophy By Estes' Criteria* | - | - | - | Normal : 552 (60.1%) <br> ST : 188 (19.4%) <br> LVH : 178 (20.5%) | Normal (56.1%) > ST (23%) > LVH (20.9%) | LVH=0 ; Normal=1 ; ST=2 |
| MaxHR | Numerical | Maximum Heart Rate Achieved | 136.8 | 25.5 | 60 & 202 | - | 70 - 150 | - |
| ExerciseAngina | Catergorical | Exercise Induced Angina <br> *Y : Yes* <br> *N : No* | - | - | - | Y : 371 (40.4%) <br> N : 547 (59.6%) | Angina (62.2%) > No Angina (37.8%) | N=0 ; Y=1 |
| Oldpeak | Numerical | ST Depression Induced By Exercise Relative To Rest (Where 1 mm = 0.1 mV) | 0.9 | 1.1 | -2.6 & 6.2 | - | >0.5 | - |
| ST_Slope | Categorical | The Slope Of The Peak Exercise ST Segment <br> *Up : Upsloping* <br> *Flat : Flat* <br> *Down : Downsloping* | - | - | - | Up : 395 (43.0%) <br> Flat : 460 (50.1%) <br> Down : 63 (6.9%) | Flat (75%) > Up (15.4%) > Down (9.6%) | Down=0 ; Flat=1 ; Up=2 |
| HeartDisease | Categorical | Output Class <br> *1 : Heart Disease* <br> *0 : Normal* | - | - | - | 1 : 508 (55.3%) <br> 0 : 410 (44.7%) | - | - |

## Conclusion To Classification Modeling

| Supervised<br>Machine Learning<br>Classification Algorithm | Accuracy | Recall<br>True Positive Rate | Cross Validation Score | ROC AUC Score |
| :-: | :-: | :-: | :-: | :-: |
| Logistic Regression | 84.24% | 0.81 | 91.43% | 84.81% |
| Support Vector | 82.07% | 0.79 | 91.17% | 82.76% |
| Decision Tree | 86.96% | 0.88 | 88.89% | 86.78% |
| Random Forest | 86.96% | 0.87 | 92.63% | 86.96% |
| k-Nearest Neighbors | 84.24% | 0.81 | 89.03% | 84.81% |

## Decision Tree Classification
Hyperparameters Used For Decision Tree Classification:
- **max_depth=4** : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- **min_samples_leaf=1** : The minimum number of samples required to be at a leaf node, with default=1. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
- **random_state=7** : Controls the randomness of the estimator.

![model-accuracy](https://i.imgur.com/qBHmkCM.png)

![model-confusion-matrix](https://i.imgur.com/QFckuHb.png)

![model-tree](https://i.imgur.com/msNFJMC.png)

The [Importance](sklearn.inspection.permutation_importance.importances_mean) Of Each Feature:
| Sorted By Accuracy | Sorted By F1 Score |
| :-: | :-: |
| ST_Slope | ST_Slope |
| Oldpeak | MaxHR |
| MaxHR | ChestPainType |
| ChestPainType | Oldpeak |
| ExerciseAngina | ExerciseAngina |
| Sex | Sex |
| Cholesterol | Cholesterol |
| FastingBS | FastingBS |
| Age | Age |

## Conclusion

- 5 different classification models are trained with this dataset and all have accuracies over 80%.
- Using these models (namely Decision Tree Classification with accuracy of 87%), doctors or nurses at the hospital will be better able to assess the risk of heart disease using non-invasive techniques. Therefore, they will be able to provide a patient with the appropriate next course of actions if a risk of heart disease is predicted.
- It is clear that some features (like "ST_Slope", "Oldpeak", "ChestPainType") are more important to predict the risk of heart disease than others (like "RestingECG", "RestingBP", "Age", "FastingBS").

## Future Investigations

- The issue of "Cholesterol" feature being bimodal distribution is not fully address and we can use [binning transformation](https://www.linkedin.com/advice/0/how-do-you-transform-skewed-bimodal-data-set-normal) by grouping it into categories based on a certain criterion, instead of removal of the feature. However, this needs to be investigated further than just applying binning transformation as the machine learning models might be trained to predict that "Cholesterol=0" means have Heart Disease.
    - [When AI flags the ruler, not the tumor — and other arguments for abolishing the black box](https://venturebeat.com/business/when-ai-flags-the-ruler-not-the-tumor-and-other-arguments-for-abolishing-the-black-box-vb-live/)
    - [History in medicine: the story of cholesterol, lipids and cardiology](https://www.escardio.org/Journals/E-Journal-of-Cardiology-Practice/Volume-19/history-in-medicine-the-story-of-cholesterol-lipids-and-cardiology)
- The current dataset does not have data from Asian individuals and it would be [critical](https://my.clevelandclinic.org/health/articles/23051-ethnicity-and-heart-disease) to carry out similar modeling technique for such a dataset.
- The current dataset does not have some features that we know risk factors for CVDs like lifestyle factors (smoking, unhealthy diet, low physical activity, alcohol use), weight, family history, diabetes, and other environmental factors. The [UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) does have some of the features but there are too many missing values within those features.
- Hyperparameter tuning was not done for modeling and it usually can improve the performance of the algorithms. Nevertheless, the current performance of the models above are good.
- Outliers were not removed even though, based on the boxplots, there are some outliers but this will require more investigation on how the data is collected and better understand of the features.



<!--

# SCTP-DataAnalyst-CapstoneProject
This is my final capstone project for NTUC LearningHub SkillsFuture Career Transition Programme For Associate Data Analyst.

# BELOW IS PLACEHOLDER FOR NOW (4 November 2024)

# My first capstone

### An analysis of snake-eating squirrels in Peru in 1988

![Screenshot of dashboard](https://i.imgur.com/UujCjhB.png)

[Link to dataset](https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv)

Description of dataset

(Para about your findings and techniques you used) "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

<details>
<summary><b>Foldable hidden section</b></summary>

Any folded content here. It requires an empty line just above it!

</details>

<details>
  <summary>Click me</summary>
  
  ### Heading
  1. Foo
  2. Bar
     * Baz
     * Qux

</details>

What you learned

What you'd change

Link to your LinkedIn

-->


<!--

Bishmer — Today at 10:40
For editing the above:
https://dillinger.io/
Online Markdown Editor - Dillinger, the Last Markdown Editor ever.
Dillinger is an online cloud based HTML5 filled
Markdown Editor. Sync with Dropbox, Github, Google Drive or OneDrive.
Convert HTML to Markdown. 100% Open Source!

-->
