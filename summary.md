# Data Quality Summary

The original dataset contains 32,561 records and 15 fields; after missing value imputation, outlier winsorization, duplicate removal, and field format standardization, the final cleaned dataset contains 32,534 records and 13 modeling fields.

## 1. Null Rates

Missing values in the original data appear only in 3 categorical fields, and there are no missing values in numeric fields.

| Field          | Missing Count | Missing Rate |
| -------------- | ------------: | -----------: |
| occupation     |         1,843 |        5.66% |
| workclass      |         1,836 |        5.64% |
| native_country |           583 |        1.79% |

The overall missing rate is relatively low, but the missing rates of `occupation` and `workclass` both exceed 5%. In addition, the missingness mechanism check for `occupation` in the Notebook shows that the income distribution and education distribution differ between the occupation-missing group and the non-missing group. For example, the proportion of `>50K` in the non-missing `occupation` group is 24.90%, while the proportion of `>50K` in the missing group is 10.36%.

For treatment, we used mode imputation for `workclass`, `occupation`, and `native_country`:

| Field          | Imputed Value  |
| -------------- | -------------- |
| workclass      | Private        |
| occupation     | Prof-specialty |
| native_country | United-States  |

After imputation, the number of missing values in all fields is 0. Compared with deletion strategies, mode imputation retains all samples and avoids a reduction in sample size caused by directly deleting rows with missing values. In the Notebook, `dropna()` reduces the data from 32,561 rows to 30,162 rows, retaining only 92.6%; deleting only rows with missing `occupation` retains 30,718 rows, or 94.3%.

## 2. Class Balance

### Target Variable income

The distribution of the original target variable `income` is as follows:

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| <=50K | 24,720 |     75.92% |
| >50K  |  7,841 |     24.08% |

The distribution of the target variable after cleaning and deduplication is as follows:

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| <=50k | 24,696 |     75.91% |
| >50k  |  7,838 |     24.09% |

It can be seen that the target variable has clear class imbalance: the low-income class accounts for about three quarters, while the high-income class accounts for about one quarter. This imbalance is not extreme, but during modeling, accuracy alone should not be relied on. Precision, recall, F1-score, ROC-AUC, or PR-AUC should also be observed, especially the recognition performance for the minority class `>50k`.

### Distribution of Main Categorical Fields

In the `sex` field, males account for 66.92% and females account for 33.08%, showing a certain degree of gender imbalance.

| sex    |  Count | Percentage |
| ------ | -----: | ---------: |
| male   | 21,773 |     66.92% |
| female | 10,761 |     33.08% |

The `race` field is highly concentrated in the `white` category, accounting for 85.42%; other racial groups account for relatively small proportions.

| race               |  Count | Percentage |
| ------------------ | -----: | ---------: |
| white              | 27,792 |     85.42% |
| black              |  3,122 |      9.60% |
| asian-pac-islander |  1,038 |      3.19% |
| amer-indian-eskimo |    311 |      0.96% |
| other              |    271 |      0.83% |

The `workclass` field is also clearly concentrated in `private`, which accounts for 75.32%.

| workclass        |  Count | Percentage |
| ---------------- | -----: | ---------: |
| private          | 24,506 |     75.32% |
| self-emp-not-inc |  2,540 |      7.81% |
| local-gov        |  2,093 |      6.43% |
| state-gov        |  1,298 |      3.99% |
| self-emp-inc     |  1,116 |      3.43% |
| federal-gov      |    960 |      2.95% |
| without-pay      |     14 |      0.04% |
| never-worked     |      7 |      0.02% |

Overall, multiple categorical fields in the dataset show long-tail distributions or category concentration.

## 3. Outlier Treatment Method

We performed multiple outlier detection methods on numeric fields, including boxplots, IQR, standard Z-score, modified Z-score, and Isolation Forest.

The number of outliers detected by the IQR method is as follows:

| Field          | Outlier Count | Percentage |
| -------------- | ------------: | ---------: |
| age            |           143 |      0.44% |
| fnlwgt         |           992 |      3.05% |
| education_num  |         1,198 |      3.68% |
| capital_gain   |         2,712 |      8.33% |
| capital_loss   |         1,519 |      4.67% |
| hours_per_week |         9,008 |     27.66% |

Isolation Forest flagged 1,628 outlier records at the multivariate level, accounting for 5.00%. The anomalous samples mainly show extremely high `capital_gain`, high `hours_per_week`, or special combinations of sample weights.

For outlier treatment, we did not directly delete a large number of samples. Instead, we used Winsorizing to retain records while limiting the influence of extreme values. The changes in maximum values after treatment are as follows:

| Field          | Original Maximum | Maximum After Winsorizing |
| -------------- | ---------------: | ------------------------: |
| age            |               90 |                        74 |
| fnlwgt         |        1,484,705 |                   510,072 |
| education_num  |               16 |                        16 |
| capital_gain   |           99,999 |                    15,024 |
| capital_loss   |            4,356 |                     1,980 |
| hours_per_week |               99 |                        80 |

In addition, we used rule-based checks to verify the reasonable ranges of key numeric fields, and no obvious rule errors were found: the number of rule-error records such as age below 17 or above 90, and weekly working hours below 1 or above 99, was 0. Therefore, this project treats these extreme values as real but center-deviating data points, and mainly reduces their impact on model training through winsorization rather than deleting them as erroneous records.

Regarding duplicate records, 27 exact duplicate records were detected after cleaning, accounting for 0.08%. After deduplication, the data changed from 32,561 rows to 32,534 rows.

## 4. Data Schema and Field Structure

The original data contains 15 fields, including 6 numeric fields and 9 categorical fields:

| Field          | Type        | Meaning                       | Unique Count |
| -------------- | ----------- | ----------------------------- | -----------: |
| age            | numeric     | Age                           |           58 |
| workclass      | categorical | Work type                     |            8 |
| fnlwgt         | numeric     | Sample weight                 |       21,134 |
| education      | categorical | Education level               |           16 |
| education_num  | numeric     | Education years code          |           14 |
| marital_status | categorical | Marital status                |            7 |
| occupation     | categorical | Occupation                    |           14 |
| relationship   | categorical | Family relationship           |            6 |
| race           | categorical | Race                          |            5 |
| sex            | categorical | Gender                        |            2 |
| capital_gain   | numeric     | Capital gain                  |          109 |
| capital_loss   | numeric     | Capital loss                  |           52 |
| hours_per_week | numeric     | Weekly working hours          |           70 |
| native_country | categorical | Native country                |           41 |
| income         | categorical | Income class, target variable |            2 |

The final cleaned table `df_clean` used for modeling retains 13 fields:

| Field             | Cleaned Type   | Unique Count |
| ----------------- | -------------- | -----------: |
| age               | integer        |           58 |
| workclass         | category       |            8 |
| education         | category       |           16 |
| marital_status    | category       |            7 |
| occupation        | category       |           14 |
| relationship      | category       |            6 |
| race              | category       |            5 |
| sex               | category       |            2 |
| hours_per_week    | integer        |           70 |
| native_country    | category       |           41 |
| capital_gain_flag | integer/binary |            2 |
| capital_loss_flag | integer/binary |            2 |
| income            | category       |            2 |

Field format processing includes: removing leading and trailing spaces from strings, converting strings to lowercase, converting categorical fields to the `category` type, keeping `income` as the binary target variable, and converting `capital_gain` and `capital_loss` into binary indicators of whether capital gain or loss exists. This can reduce the impact of extreme monetary values while retaining capital-related information.

## 5. Overall Evaluation

Overall, the dataset has good quality: the field structure is clear, numeric fields have no missing values, categorical fields have relatively low missing rates, and rule-based checks did not find obvious invalid numeric values. The main data quality issues include: small amounts of missing values in `occupation`, `workclass`, and `native_country`; a class imbalance of about 76:24 in the target variable `income`; concentrated distributions in some categorical fields; and obvious extreme values in `capital_gain`, `capital_loss`, `fnlwgt`, and `hours_per_week`.

After mode imputation, Winsorizing, duplicate removal, and string standardization, the data is suitable for subsequent encoding and modeling workflows.
