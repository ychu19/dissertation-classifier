# Classifying Countries of Origins among Naturalized Citizens
----

{:toc}
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
2. Created X
    1. Created a column with numbers of "betsumes" each individual has (numbers of non-NAs in column `betsume.1`, `betsume.2`, `betsume.3`)
    2. Created one-hot encoding columns `kr_last_name` and `ch_last_name` if the individual's last name matches top 100 common last names in Korea and China
    3. Crossed the features `kr_last_name` and `ch_last_name` for the overlapping last names (like Lee(李) or Kim(金))
    4. Created one-hot encoding columns `kr_first_name` and `ch_first_name` if the indvidiaul's first name (either characters) matches the most common first names in Korea and China in the history
    5. Crossed the features `kr_first_name` and `ch_first_name` for the overlapping words in first names (like 蘭 and 英)

## Data Pre-processing

1. Combined the data from 1954 and 1955
2. Shuffled the rows and set aside 25% of the samples as test set
3. Since the data is moderately imbalanced (with around 5,000 Koreans and 250 Chinese), downsampled the Koreans in the training set to 65%
4. Split the training set into training and validation sets 

## Build Models


