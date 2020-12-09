#%%
### IMPORTING LIBRARIES TO USE, READING IN DATA ### 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from scipy import stats
from collections import Counter
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


whiffData = pd.read_csv('PitcherXData.csv')

#%%
### INITIAL DATA EXPLORATION AND CLEANING ### 

## Identify categorical variables and numeric variables
numericCols = whiffData.select_dtypes(include=np.number).columns
categoricalCols = list(set(whiffData.columns) - set(numericCols))

print(numericCols)
print(categoricalCols)

## Detect Null values
whiffNulls = whiffData[whiffData.isnull().any(axis=1)]

## Confirm that Null values do not have significant skew before dropping null rows
## We want to avoid a situation where the values in the null rows significantly change the overall data
for col in numericCols: 
    originalMean = whiffData[col].mean()
    nullMean = whiffNulls[col].mean()

    print(col, ': ', originalMean, ' ', nullMean)

for col in categoricalCols: 
    originalSet = set(whiffData[col])
    nullSet = set(whiffNulls[col])

    print(col)
    print(originalSet)
    print(nullSet)

## Worth looking into InducedVertBreak and HorzBreak from first glance
## Induced vertical break has a standard deviation of around 6.5 - the null rows' aggregate InducedVertBreak is within 1 std of the main set 
whiffData['InducedVertBreak'].std()

## HorzBreak is over one standard deviation, but not by much. It is close, but acceptable to drop - no skew!
whiffData['HorzBreak'].std()

## All the numeric columns look like they display no skew when comparing the null rows and the non null rows (main set)
whiffData = whiffData.dropna()

#%% 
### OUTLIER DETECTION AND HANDLING ### 

## For numeric variables, do outlier detection
whiffNumeric = whiffData[numericCols]

for col in numericCols:
    outliers = whiffNumeric[~(np.abs(whiffNumeric[col] - whiffNumeric[col].mean()) < (3 *whiffNumeric[col].std()))]
    
    if len(outliers) > 0:
        print(col, len(outliers))

## After a cursory analysis of the columns with outliers (looking at the range and standard deviation), only SpinRate is of note 
## Problem: several zeros in data (very unlikely), around ~50 rows 
## Options: impute data OR get rid of 50 more rows. To decide, we conduct another skew analysis 

whiffZeros = whiffData[whiffData.SpinRate == 0]
for col in numericCols: 
    originalMean = whiffData[col].mean()
    zeroMean = whiffZeros[col].mean()

    print(col, ': ', originalMean, ' ', zeroMean)

for col in categoricalCols: 
    originalSet = set(whiffData[col])
    zeroSet = set(whiffZeros[col])

    print(col)
    print(originalSet)
    print(zeroSet)

## The dataset where the Spin Rate is zero is almost identical on average to the main data set, so no skew detected 
## As a result, we simply remove the 50 rows where the Spin Rate is 0 (likely a data quality issue) 
whiffData = whiffData[whiffData.SpinRate != 0]

# %%
### CORRELATION ANALYSIS ###

## Here we build a correlation matrix
corrMatrix = pd.DataFrame(whiffData.corr()).abs()
corrMatrix.loc['average'] = corrMatrix.mean()

high_corrs = []
for idx,row in corrMatrix.iterrows(): 
    for col in corrMatrix.columns: 
        if (row[col] < 1) & (row[col] > 0.75): 
            high_corrs.append(col)

print(Counter(high_corrs))

## Pitch of Plate Appearance is correlated with Balls and Strikes, this we need to solve 
## Release Speed is correlated (as expected) with Induced Vertical Break (these are considered distinct, so we keep both, despite the correlation)

## Pitch of Plate Appearance, Balls, and Strikes actually combine to say the same thing: Count
## So, instead of having 3 variables here, we construct a simple 'Count' categorical value 
whiffData['Count'] = (whiffData.Balls).astype('str') + (whiffData.Strikes).astype('str')
categoricalCols.append('Count')

## Making a full copy before slimming down whiffData (there was only one value in Pitcher, so that column gets dropped)
## We also drop PitcherThrows given that only 3 records total have the pitcher throwing right handed
whiffDataFull = whiffData 
whiffData = whiffData.drop(['Balls', 'Strikes', 'PitchofPA', 'Pitcher', 'PitcherThrows'], axis=1)


# %%
### FEATURE ENGINEERING ### 

## For this analysis, we don't use Year or Date, given that we expect some change in the pitcher's behavior, it does not make sense to extrapolate trends over time (changes in behavior can change the nature of these trends)
whiffRegressionData = whiffData.drop(['Date', 'Year'], axis = 1)

## Here we want to create dummy variables for our categorical variables 
## We choose this over other techniques, such as target encoding because we don't have too many categorical variables
whiffRegressionData = pd.get_dummies(whiffRegressionData)



#%% 
### TARGET DISTRIBUTION ANALYSIS ### 

whiffRegressionData['whiff_prob'].hist(bins = 10)

val = 0
for i in range(0, 10): 
    print(val)
    print(len(whiffRegressionData[(whiffRegressionData.whiff_prob < val + .05) & (whiffRegressionData.whiff_prob > val)]))
    val += .05
    print(val)
    print('---------------')

whiffRegressionData['whiff_prob_category'] = 0
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.05) & (whiffRegressionData['whiff_prob'] > 0), 1, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.1) & (whiffRegressionData['whiff_prob'] > 0.05), 2, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.15) & (whiffRegressionData['whiff_prob'] > 0.1), 3, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.2) & (whiffRegressionData['whiff_prob'] > 0.15), 4, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.25) & (whiffRegressionData['whiff_prob'] > 0.2), 5, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.3) & (whiffRegressionData['whiff_prob'] > 0.25), 6, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.35) & (whiffRegressionData['whiff_prob'] > 0.3), 7, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] > 0.35), 8, whiffRegressionData['whiff_prob_category'])


groupedMeans = pd.DataFrame(whiffRegressionData.groupby(['whiff_prob_category']).mean())
groupedMeans['category'] = groupedMeans.index

# %%
groupedMeans[['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER']]

sigVars = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob']

## PLAN
## use work from target distrubiton analysis to figure out which variables to improve 

## train an actually good model, predict whiff prob using status quo data and then show that with improved variabled, the prediction yields a higher whiff prob 

## create a what if scenario: if spin rate improves by X --> whiff prob improves by y%
## OR if spin rate improves by X and horz break improves by Y --> whif prob improves by z%

# %%


whiffRegressionData['PlateHeight'].std()

for threshhold in [2.5, 2.4, 2.3, 2.2, 2.1, 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, .9, .8, .7, .6, .5]:
    avg = whiffRegressionData[abs(whiffRegressionData.PlateHeight - threshhold) < .2]['whiff_prob'].mean()
    print(threshhold, avg)

# %%

whiffRegressionData['PlateSide'].std()







whiffRegressionData['PlateHeight'].hist(bins = 20)
# %%
