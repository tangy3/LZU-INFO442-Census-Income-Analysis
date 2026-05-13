
# Proposal

## 1. Domain and Motivation

This project focuses on income prediction and workforce analytics using data science methods. Income level is closely related to education, occupation, working hours, and demographic characteristics, making it an important topic in both economics and human resource management.

Understanding the factors that influence income can help organizations improve recruitment strategies, employee training plans, and compensation policies. In addition, analyzing income inequality and potential gender disparities can support fairer and more inclusive decision-making in the workplace.

As companies increasingly rely on data-driven management, predictive analysis can provide practical insights for talent acquisition, salary planning, and workforce development. Therefore, this project is meaningful from both business and social perspectives.

## 2. Dataset Description

This project will use the UCI Machine Learning Repository Adult Income Dataset.

**Dataset Information**

**Source:** UCI Machine Learning Repository

**Dataset Name:** Adult Income Dataset

**Size:** Approximately 48,000 records

**Format:** CSV / tabular structured data

**Main Features:** age, education, occupation, marital status, gender, workclass, hours-per-week, native-country, etc.

**Target Variable:** Whether annual income exceeds $50K (<=50K or >50K)

**Access Method:**

The dataset can be downloaded directly from the official UCI repository (https://archive.ics.uci.edu/dataset/2/adult) and imported into Python using Pandas.

**Ethical Considerations:**

Although the dataset is anonymized, it contains demographic information such as gender and race. The project will only use the data for academic analysis and predictive modeling purposes. We will avoid biased interpretations and carefully discuss fairness issues when analyzing income disparities.

## 3. Scientific Questions

This project aims to explore the factors that influence annual income using the UCI Adult Income Dataset and investigate whether income level can be effectively predicted through data analysis and machine learning techniques.

At the current stage, we have preliminarily proposed the following five research questions based on the dataset characteristics and project objectives. As the project progresses, we may further refine the scope and select several key questions for deeper analysis and modeling according to the findings from data preprocessing and exploratory data analysis.

**3.1 Are there significant income distribution differences across demographic groups?**

The project will investigate whether income levels vary across groups such as gender, education background, and occupation categories, and whether these differences are statistically meaningful.

**3.2 Which demographic and employment-related factors have the greatest influence on annual income?**

We will examine how variables such as education level, occupation, age, marital status, and weekly working hours relate to whether an individual earns more than $50K annually.

**3.3 Can annual income level be accurately predicted using demographic and employment-related information, and which model will provides the best predictive performance?**

Machine learning classification models will be used to determine whether individual income categories can be predicted effectively based on the available features in the dataset, as measured by appropriate metrics such as F1-score and AUC-ROC.

**3.4 What limitations and potential biases exist in income prediction models?**

The project will evaluate possible issues such as class imbalance, demographic bias, overfitting, and limited generalizability when applying predictive models to income-related problems.

## 4. Preliminary Hypotheses

Based on previous economic research, exploratory understanding of the dataset, and the scientific questions proposed in this project, we propose the following initial hypotheses:

**4.1 Education level, occupation, and working hours will be among the most influential factors affecting annual income.**

We hypothesize that individuals with higher education levels, professional or technical occupations, and longer working hours are more likely to earn more than $50K annually. This expectation is based on the common relationship between human capital, job qualifications, and earning potential observed in labor market studies.

**4.2 Significant income distribution differences may exist across demographic groups.**

We expect that the proportion of individuals earning >$50K will differ significantly across these groups. Specifically, we anticipate a higher proportion of high-income earners among males compared to females, among individuals with advanced degrees compared to those with only high school education, and among those in managerial or professional roles compared to those in service or manual occupations.

**4.3 Machine learning models can predict income levels with accuracy higher than a random baseline.**

This project assumes that demographic and employment-related features can be used to reasonably predict whether an individual’s annual income exceeds $50K. It is expected that classification models such as Random Forest, Decision Tree, and Logistic Regression will achieve good predictive performance, with ensemble learning models likely to perform better in terms of evaluation metrics.

**4.4 Income prediction models may contain limitations and potential biases.**

We hypothesize that issues such as class imbalance, demographic bias, and overfitting may affect model performance and fairness. In addition, since the dataset represents a specific population and time period, the model’s generalizability to other contexts may be limited.

## 5. Team Roles and Responsibilities

**Jianing Han**
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Model evaluation
- Report writing support

**Chenyu Li**
- Data visualization
- Presentation preparation
- Report writing
- EDA support

**Jiayi Liu**
- Data collection and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Machine learning model support

**Zifei Wang**
- Data preprocessing
- Machine learning model training
- Visualization and evaluation
- Statistical analysis support

The team members will collaborate closely throughout all stages of the project, including data preprocessing, exploratory analysis, model development, visualization, and report preparation. Several tasks will be completed jointly to ensure balanced workload distribution, consistent analysis results, and effective integration of project findings.
