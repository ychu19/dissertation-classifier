# Classifying Countries of Origins among Naturalized Citizens
----
Table of Contents
- [Classifying Countries of Origins among Naturalized Citizens](#classifying-countries-of-origins-among-naturalized-citizens)
  * [Purpose](#purpose)
  * [Data Source](#data-source)
  * [Data Cleaning](#data-cleaning)
  * [Data Pre-processing](#data-pre-processing)
  * [Build Models](#build-models)
    + [Cross-validation with `sklearn.model_selection`](#cross-validation-with--sklearnmodel-selection-)
    + [Use Both Training and Validation Sets to Train the Model](#use-both-training-and-validation-sets-to-train-the-model)
  * [Run the Model on Test Data to Predict Citizenship](#run-the-model-on-test-data-to-predict-citizenship)
  * [Conclusion](#conclusion)
  * [Future Steps](#future-steps)

--
## Purpose

My dissertation project seeks to explain why permanent immigrants in Japan refused to acquire Japanese citizenship even when they were born and raised there <sup>1</sup>. I hypothesize that their home country attachment through diasporic organizations affects their propensity to naturalize.

I pulled the naturalization records from the Japanese Government Gazette ([官報](https://search.npb.go.jp/kanpou/)), with information about each and every naturalized individuals in Japan since the 1950s. I focus on the time between the 1971 and 1980, with a total of 72,416 individuals who have naturalized.

This document presents a smaller project within my dissertation - **classifying the country of origin for each naturalized individual**. The information about naturalized citizens from the Gazette includes (a) their names, (b) their names before naturalization <sup>2</sup>, (c) their residential addresses, (d) their dates of birth, and (e) dates of approval. While the Gazette provides a rare and valuable opportunity to look into the individual-level features of naturalized citizens, it does not include information about their countries of origin. Fortunately, the Gazette did include original citizenship for those who naturalized in the 50s. **This project uses the data from 1954 to 1955 as prior to predict the countries of origin for those who naturalized in the 70s.**

[1]: Japan is not governed by *jus soli*, meaning that there is no birthright citizenship in Japan. [See Japan MOJ](http://www.moj.go.jp/ENGLISH/information/tnl-01.html)

[2]: Prior to 1983, most of the applicants to naturalization were implicitly asked to change their names to a Japanese-sounding name. [See Wikipedia](https://en.wikipedia.org/wiki/Japanese_nationality_law#Naturalization).

## Data Source

Japanese Government Gazette ([官報](https://search.npb.go.jp/kanpou/)) in 1954 and 1955, with a total of 5,034 individuals who naturailzed. (Personally identifiable items have been anonymized here)

|    | full_name     | citizenship   | address_anonym   | birthdate                |   household |   date_approval | betsume.1   |   betsume.2 |   betsume.3 |
|---:|:---------:|:--------------:|:-----------------:|:-------------------------:|------------:|----------------:|:------------:|:------------:|:------------:|
|  0 | ＊光＊    | 無国籍        | 東京都           | 大正十四年十二月十一日生 |           1 |        19540105 | nan         |         nan |         nan |
|  1 | ＊鎮＊    | 朝鮮          | 同県山           | 昭和十四年三月十六日生   |           2 |        19540105 | ＊城正＊      |         nan |         nan |
|  2 | ＊本万＊  | 朝鮮          | 高知県           | 明治四十二年七月七日生   |           3 |        19540105 | ＊万＊      |         nan |         nan |
|  3 | ＊本又＊  | 朝鮮          | 高知県           | 大正六年三月二十四日生   |           4 |        19540105 | ＊又＊        |         nan |         nan |
|  4 | ＊本玉＊  | 朝鮮          | 同県同           | 昭和十三年九月二十九日生 |           5 |        19540105 | ＊玉＊        |         nan |         nan |

## Data Cleaning

1. Created y: Coded Koreans (朝鮮) as 1, Chinese (無国籍 or 中華民国) as 2, and others as 0
```python
def nationality(string):
  """Get a string from the archival data, 
     Return whether an individual is of (1) Korean, (2) Chinese, 
     or (0) Other national origin"""
  nationality = 0 # default set as others
  if string == '朝鮮':
    nationality = 1
  elif string in ['無国籍','中華民国']:
    nationality = 2
  return nationality
```
2. Created X
    1. Created a column with numbers of "betsumes" each individual has (numbers of non-NAs in column `betsume.1`, `betsume.2`, `betsume.3`)
    ```python
    def betsume_numbers(betsume1, betsume2, betsume3):
       """calculate the numbers of betsume regardless of the column sequence"""
       b1 = pd.notna(betsume1)
       b2 = pd.notna(betsume2)
       b3 = pd.notna(betsume3) # python calculates Bloolean as integers 
       return (b1+b2+b3)
    ```
    2. Created one-hot encoding columns `kr_last_name` and `ch_last_name` if the individual's last name matches top 100 common last names in Korea and China
    ```python
    data['kr_last_name'] = data['last.name'].isin(kr_last_names).astype(int)
    data['ch_last_name'] = data['last.name'].isin(ch_last_names).astype(int)
    ```
    3. Crossed the features `kr_last_name` and `ch_last_name` for the overlapping last names (like Lee(李) or Kim(金))
    ```python
    data['last_name_cross'] = data['kr_last_name'] * data['ch_last_name']
    ```
    4. Created one-hot encoding columns `kr_first_name` and `ch_first_name` if the indvidiaul's first name (either characters) matches the most common first names in Korea and China in the history
    ```python
    name_list = ch_first_names.str.cat()
    def name_in_list(name):
        """input: name
           output: whether name is a member of our list
           function also deal with NA values + ""
        """
        if pd.notna(name) and len(name) > 0:
          return name in name_list
        else:
          return False
    ```
    ```python
    name_list = kr_first_names.str.cat()
    def name_in_list(name):
        """input: name
           output: whether name is a member of our list
           function also deal with NA values + ""
        """
        if pd.notna(name) and len(name) > 0:
          return name in name_list
        else:
          return False
    ```
    5. Crossed the features `kr_first_name` and `ch_first_name` for the overlapping words in first names (like 蘭 and 英)
    ```python
    data['first_name_cross'] = data['kr_first_name'] * data['ch_first_name']
    ```

## Data Pre-processing

1. Combined the data from 1954 and 1955
```python
frames = [data_1954, data_1955]
data = pd.concat(frames).reset_index(drop=True)
```
2. Shuffled the rows and set aside 25% of the samples as test set
```python
data = data.reindex(np.random.permutation(data.index)) # shuffle the rows
train_set = data.sample(frac=0.75, random_state=0)
test_set = data.drop(train_set.index)
```
3. Since the data is moderately imbalanced (with around 5,000 Koreans and 250 Chinese), downsampled the Koreans in the training set to 65%
```python
target_chinese_sample_size = len(train_set[train_set['citizenship']==2]) 
target_chinese_sample_ratio = 0.35
adjusted_sample_total = int(target_chinese_sample_size/target_chinese_sample_ratio)

target_korean_sample_size = adjusted_sample_total - target_chinese_sample_size

adjusted_sample_koreans = train_set[train_set['citizenship']==1].sample(
    n=target_korean_sample_size, 
    random_state=0)
adjusted_sample_chinese = train_set[train_set['citizenship']==2]

frames = [adjusted_sample_koreans, adjusted_sample_chinese]
adjusted_sample = pd.concat(frames)
adjusted_sample = adjusted_sample.reindex(np.random.permutation(adjusted_sample.index)) # shuffle the rows

y = adjusted_sample.citizenship
X = adjusted_sample[features]
```
4. Split the training set into training and validation sets 
```python
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) # train data and validation data
```
## Build Models

Usking `sklearn`, I built the following models:
1. DecisionTreeClassifier
2. RandomForestClassifier

### Cross-validation with `sklearn.model_selection`
```python
cv = cross_validate(rf_model_on_full_data, X, y, cv=10)
```
With a mean precision score of 0.6614

### Use Both Training and Validation Sets to Train the Model
```python
rf_model_on_train_data = RandomForestClassifier(random_state = 1,class_weight={1:5,2:1})
rf_model_on_train_data.fit(X, y)
```

## Run the Model on Test Data to Predict Citizenship

```python
test_preds = rf_model_on_full_data.predict_proba(test_X)

test_y_pred = [test_preds[i][0]>=threshold for i in range(len(test_preds))]
test_y_pred = pd.Series(test_y_pred).astype(int)
test_accuracy = accuracy_score(test_y, test_y_pred)
```
Accuracy score in test data is 0.7957.

## Conclusion

With features of first names, last names, and names used before naturalization (betsume), I was able to predict countries of origin among naturalized individuals in Japan. 

## Future Steps

1. Build and test more models (such as `BaggingClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, etc.
2. Find optimal parameters across all models and compare across models 
3. Use the best model to predict countries of origin for naturalized individuals between 1971 and 1980
