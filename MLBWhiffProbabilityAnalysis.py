#%%
###################################################
### IMPORTING LIBRARIES TO USE, READING IN DATA ###
###################################################
 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from scipy import stats
from collections import Counter
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import math 
import random 
from tqdm import tqdm

whiffData = pd.read_csv('PitcherXData.csv')

## Set noisy = True for all print statements
noisy = True
## Set max output to True to produce every graph 
maxOutput = False

#%%
#############################################
### INITIAL DATA EXPLORATION AND CLEANING ### 
#############################################

## Identify categorical variables and numeric variables
numericCols = whiffData.select_dtypes(include=np.number).columns
categoricalCols = list(set(whiffData.columns) - set(numericCols))

if noisy:
    print(numericCols)
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')
    print(categoricalCols)
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')

## Detect Null values
whiffNulls = whiffData[whiffData.isnull().any(axis=1)]

## Confirm that Null values do not have significant skew before dropping null rows
## We want to avoid a situation where the values in the null rows significantly change the overall data
for col in numericCols: 
    originalMean = whiffData[col].mean()
    nullMean = whiffNulls[col].mean()

    if noisy:
        print(col, ': ', originalMean, ' ', nullMean)

for col in categoricalCols: 
    originalSet = set(whiffData[col])
    nullSet = set(whiffNulls[col])

    if noisy:
        print('>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<')
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
######################################
### OUTLIER DETECTION AND HANDLING ###
######################################  

## For numeric variables, do outlier detection
whiffNumeric = whiffData[numericCols]

for col in numericCols:
    outliers = whiffNumeric[~(np.abs(whiffNumeric[col] - whiffNumeric[col].mean()) < (3 *whiffNumeric[col].std()))]
    
    if len(outliers) > 0 and noisy:
        print(col, len(outliers))
        print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')


## After a cursory analysis of the columns with outliers (looking at the range and standard deviation), only SpinRate is of note 
## Problem: several zeros in data (very unlikely), around ~50 rows 
## Options: impute data OR get rid of 50 more rows. To decide, we conduct another skew analysis 

whiffZeros = whiffData[whiffData.SpinRate == 0]
for col in numericCols: 
    originalMean = whiffData[col].mean()
    zeroMean = whiffZeros[col].mean()

    if noisy: 
        print(col, ': ', originalMean, ' ', zeroMean)

if noisy:
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')


for col in categoricalCols: 
    originalSet = set(whiffData[col])
    zeroSet = set(whiffZeros[col])

    if noisy:
        print(col)
        print(originalSet)
        print(zeroSet)

## The dataset where the Spin Rate is zero is almost identical on average to the main data set, so no skew detected 
## As a result, we simply remove the 50 rows where the Spin Rate is 0 (likely a data quality issue) 
whiffData = whiffData[whiffData.SpinRate != 0]






# %%
############################
### CORRELATION ANALYSIS ###
############################

## Here we build a correlation matrix
corrMatrix = pd.DataFrame(whiffData.corr()).abs()
corrMatrix.loc['average'] = corrMatrix.mean()

high_corrs = []
for idx,row in corrMatrix.iterrows(): 
    for col in corrMatrix.columns: 
        if (row[col] < 1) & (row[col] > 0.75): 
            high_corrs.append(col)

if noisy:
    print(Counter(high_corrs))

## Pitch of Plate Appearance is correlated with Balls and Strikes, this we need to solve 
## Release Speed is correlated (as expected) with Induced Vertical Break (these are considered distinct, so we keep both, despite the correlation)

## Pitch of Plate Appearance, Balls, and Strikes actually combine to say the same thing: Count
## So, instead of having 3 variables here, we construct a simple 'Count' categorical value 
whiffData['Count'] = (whiffData.Balls).astype('str') + (whiffData.Strikes).astype('str')
categoricalCols.append('Count')

## Making a full copy before slimming down whiffData (there was only one value in Pitcher, so that column gets dropped)
## We also drop PitcherThrows given that only 3 records total have the pitcher throwing right handed
## Induced Vertical break and Release Speed are correlated, but distinct enough to stay
whiffDataFull = whiffData 
whiffData = whiffData.drop(['Balls', 'Strikes', 'PitchofPA', 'Pitcher', 'PitcherThrows'], axis=1)



# %%
###########################
### FEATURE ENGINEERING ###
########################### 

## For this analysis, we don't use Year or Date, given that we expect some change in the pitcher's behavior, it does not make sense to extrapolate trends over time (changes in behavior can change the nature of these trends)
whiffRegressionData = whiffData.drop(['Date', 'Year'], axis = 1)

## Here we want to create dummy variables for our categorical variables 
## We choose this over other techniques, such as target encoding because we don't have too many categorical variables
whiffRegressionData = pd.get_dummies(whiffRegressionData)






#%% 
####################################
### TARGET DISTRIBUTION ANALYSIS ###
#################################### 

## Visualize the target variable (no normal distribution based assumptions hold)
whiffRegressionData['whiff_prob'].hist(bins = 20)

## Define skew 
val = 0
for i in range(0, 10): 
    if noisy:
        print('Min :', val)
        print(len(whiffRegressionData[(whiffRegressionData.whiff_prob < val + .05) & (whiffRegressionData.whiff_prob > val)]))
        print('Max :', round(val,1) + .05)
        print('---------------')

    val += .05


whiffRegressionData['whiff_prob_category'] = 0
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.05) & (whiffRegressionData['whiff_prob'] > 0), 1, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.1) & (whiffRegressionData['whiff_prob'] > 0.05), 2, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.15) & (whiffRegressionData['whiff_prob'] > 0.1), 3, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.2) & (whiffRegressionData['whiff_prob'] > 0.15), 4, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.25) & (whiffRegressionData['whiff_prob'] > 0.2), 5, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.3) & (whiffRegressionData['whiff_prob'] > 0.25), 6, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] < 0.35) & (whiffRegressionData['whiff_prob'] > 0.3), 7, whiffRegressionData['whiff_prob_category'])
whiffRegressionData['whiff_prob_category'] = np.where((whiffRegressionData['whiff_prob'] > 0.35), 8, whiffRegressionData['whiff_prob_category'])

## Aggregate each variable 
groupedMeans = pd.DataFrame(whiffRegressionData.groupby(['whiff_prob_category']).mean())
groupedMeans['category'] = groupedMeans.index

groupedMeans[['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER']]

## These are the variables that in this analysis, appear to be significant
sigVars = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob']








# %%
###############################################
### CHECKING VARIABLE CORRELATION FOR MODEL ###
############################################### 

## Checking Correlation (takes a long time to run)
pairPlot = sns.pairplot(data=whiffRegressionData,
                  y_vars=['whiff_prob'],
                  x_vars=['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32'])


## Appears to be a non-linear, perhaps radial relationship 




#%%
#########################################
### MODEL IMPLEMENTATION, SVR - WHIFF ###
#########################################

## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob']


## Baseline Model (with all variables included)
## Generate Model
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')

## Model with one or two variables removed (loop can be edited)
rsqValues = []
for variable in tqdm(varsToInclude):
    ## The second loop can be used to understand the impacts of interaction terms (or omitted)
     for variable2 in varsToInclude:
        varList = list(set(varsToInclude) - set([variable, variable2]))
        
        ## Generate Model
        X = whiffRegressionData[varList]
        y = whiffRegressionData['whiff_prob'].values.reshape(-1,1)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)
        
        ## Score Model
        score = regressor.score(x_test, y_test)
        rsqValues.append(score)

        if score < 0.6 and noisy:
            print(variable, variable2, score)

## Plate Height and Swing Prob combine for 63%, essentially even contribution (slightly more towards plate height)

### Significant variables: 
sigVarsWhiff = ['PlateHeight', 'swing_prob']





#%%
###############################
### PLATE HEIGHT - ANALYSIS ###
############################### 

## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob']


## Baseline Model (with all variables included)
## Generate Model
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')


### Simulation - Whiff Rate Baseline
deltas = []
avgs = []
for std in tqdm([1,2,3,4,5]):
    for seed in list(range(1,101)):
        seed = random.randint(1,1000)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = seed)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)
        #print('MODEL SCORE ', regressor.score(x_test,y_test))


        x_testDF = pd.DataFrame(x_test)
        x_testDF.columns = varsToInclude

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        baselineAvg = sum(scaledVals) / len(scaledVals)

        ### Simulation - Increase
        oneStDev = abs(x_testDF['PlateHeight'].std())
        x_testDF['PlateHeight'] = random.uniform(0.9,1.1) # x_testDF['PlateHeight'] - (std * oneStDev) 

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        increasedAvg = sum(scaledVals) / len(scaledVals)

        delta = (increasedAvg - baselineAvg) / baselineAvg 
        deltas.append(delta)
        #print('DELTA ', delta)
        #print('----------------------------')

    deltaAvg = sum(deltas) / len(deltas) * 100
    avgs.append(deltaAvg)
    if noisy:
        print('Change Avg: ', deltaAvg)

## Sanity check that it is not skewed based on batter side 
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

## Note: these are pretty low correlations, but we aren't concerned 
## Primarily because the model is likely reliant on iteraction terms instead of singular variables 
## That is why we do the simulation test, to remove a variable and have it impact all interactions
print(whiffDataRighty['PlateHeight'].corr(whiffDataRighty['swing_prob']))
print(whiffDataLefty['PlateHeight'].corr(whiffDataLefty['swing_prob']))






#%%
#############################
### SWING PROB - ANALYSIS ###
#############################

## Tried backwise selection, but model performed better with all variables 
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']


## Baseline Model 
## Generate Model
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['swing_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model
score = regressor.score(x_test, y_test)
print(score)


## Model with one or two variables removed (loop can be edited)
rsqValues = []
for variable in tqdm(varsToInclude):
    ## The second loop can be used to understand the impacts of interaction terms (or omitted)
     for variable2 in varsToInclude:
        varList = list(set(varsToInclude) - set([variable, variable2]))
        
        ## Generate Model
        X = whiffRegressionData[varList]
        y = whiffRegressionData['swing_prob'].values.reshape(-1,1)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)
        
        ## Score Model
        score = regressor.score(x_test, y_test)
        rsqValues.append(score)

        if score < 0.6 and noisy:
            print(variable, variable2, score)

### RESULT: Plate Side and Count have the biggest impacts (also Plate Height, but we have already covered that) 






#%%
###################################
### SWING PROB ANALYSIS - COUNT ###
################################### 



## Let's split based off batter side 
whiffDataRighty = whiffRegressionData[whiffRegressionData.BatterSide_Right == 1]
whiffDataLefty = whiffRegressionData[whiffRegressionData.BatterSide_Left == 1]

# print('swingProb', whiffRegressionData['swing_prob'].corr(whiffDataRighty['whiff_prob']))

## Count 
print('COUNT')
for DF in [whiffDataRighty, whiffDataLefty]: 
    print('00', DF['Count_00'].corr(DF['swing_prob']))
    print('01', DF['Count_01'].corr(DF['swing_prob']))
    print('02', DF['Count_02'].corr(DF['swing_prob']))
    print('10', DF['Count_10'].corr(DF['swing_prob']))
    print('11', DF['Count_11'].corr(DF['swing_prob']))
    print('12', DF['Count_12'].corr(DF['swing_prob']))
    print('20', DF['Count_20'].corr(DF['swing_prob']))
    print('21', DF['Count_21'].corr(DF['swing_prob']))
    print('22', DF['Count_22'].corr(DF['swing_prob']))
    print('30', DF['Count_30'].corr(DF['swing_prob']))
    print('31', DF['Count_31'].corr(DF['swing_prob']))
    print('32', DF['Count_32'].corr(DF['swing_prob']))
    print('>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<')


## Hard to "control" count, but this analysis will be woven into conclusions





#%%
####################################
### PLATE SIDE BASELINE ANALYSIS ###
####################################

for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataRighty[abs(whiffDataRighty.PlateSide - threshhold) < .2]['swing_prob'].mean()
    if noisy: 
        print(threshhold, avg)

## Right: Range of -0.4 to 0.4 for swing prob


for threshhold in list(np.linspace(-2.5,3.5,100)):
    avg = whiffDataLefty[abs(whiffDataLefty.PlateSide - threshhold) < .2]['swing_prob'].mean()
    if noisy: 
        print(threshhold, avg)

## Right: Range of -0.5 to 0.2 for swing prob


for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataRighty[abs(whiffDataRighty.PlateSide - threshhold) < .2]['whiff_prob'].mean()
    if noisy: 
        print(threshhold, avg)

## Right: Range of -0.5 to 1.1 for whiff prob


for threshhold in list(np.linspace(-2.5,3.5,100)):
    avg = whiffDataLefty[abs(whiffDataLefty.PlateSide - threshhold) < .2]['whiff_prob'].mean()
    if noisy: 
        print(threshhold, avg)

## Right: Range of -0.25 to 1.25 for whiff prob




#%%
##########################################
### SWING PROB ANALYSIS - PLATE SIDE R ###
##########################################

## Plate Side Correlations
if noisy:
    print(whiffRegressionData['PlateSide'].corr(whiffRegressionData['swing_prob']))
    print('Right', whiffDataRighty['PlateSide'].corr(whiffDataRighty['swing_prob']))
    print('Left', whiffDataLefty['PlateSide'].corr(whiffDataLefty['swing_prob']))

### Splitting By Right vs Left Side Batters
### Plate Side, Righty ### 

whiffDataRighty = whiffRegressionData[whiffRegressionData.BatterSide_Right == 1]

### Generate Baseline Model
## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob', 'PlateSide']


## Baseline Model (with all variables included)
## Generate Model
X = whiffDataRighty[varsToInclude]
y = whiffDataRighty['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')


### Simulation - Plate Side, Righty
avgs = []
for iter in tqdm([1,2,3,4,5]):
    deltas = []
    for seed in list(range(1,101)):
        seed = random.randint(1,1000)
        
        ## Generate Model
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = seed)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)

        x_testDF = pd.DataFrame(x_test)
        x_testDF.columns = varsToInclude

        ## Generate Predictions
        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1, this methods retains the distribution
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        baselineAvg = sum(scaledVals) / len(scaledVals)

        ### Simulation - Change Values in Prediction Set
        x_testDF['PlateSide'] = random.uniform(-0.5, 1.1)  

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        changedAvg = sum(scaledVals) / len(scaledVals)

        ## We use percent change here because the actual prediction values have lost meaning
        ## One, because the prediction range is wider than the actual possible range of values (infinite range vs [0,1])
        ## Two, because we have normalized the values 
        delta = (changedAvg - baselineAvg) / baselineAvg 
        deltas.append(delta)


    deltaAvg = sum(deltas) / len(deltas) * 100
    avgs.append(deltaAvg)
    print(iter, ' Change Avg: ', deltaAvg)

print(sum(avgs) / len(avgs))

## Average improve of x16% in Whiff Rate 



#%%
##########################################
### SWING PROB ANALYSIS - PLATE SIDE L ###
##########################################


### Splitting By Right vs Left Side Batters
### Plate Side, Lefty ### 

whiffDataLefty = whiffRegressionData[whiffRegressionData.BatterSide_Left == 1]

### Generate Baseline Model
## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob', 'PlateSide']


## Baseline Model (with all variables included)
## Generate Model
X = whiffDataLefty[varsToInclude]
y = whiffDataLefty['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')


### Simulation - Plate Side, Righty
avgs = []
for iter in tqdm([1,2,3,4,5]):
    deltas = []
    for seed in list(range(1,101)):
        seed = random.randint(1,1000)
        
        ## Generate Model
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = seed)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)

        x_testDF = pd.DataFrame(x_test)
        x_testDF.columns = varsToInclude

        ## Generate Predictions
        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1, this methods retains the distribution
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        baselineAvg = sum(scaledVals) / len(scaledVals)

        ### Simulation - Change Values in Prediction Set
        x_testDF['PlateSide'] = random.uniform(0.25,1.25)  

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        changedAvg = sum(scaledVals) / len(scaledVals)

        ## We use percent change here because the actual prediction values have lost meaning
        ## One, because the prediction range is wider than the actual possible range of values (infinite range vs [0,1])
        ## Two, because we have normalized the values 
        delta = (changedAvg - baselineAvg) / baselineAvg 
        deltas.append(delta)


    deltaAvg = sum(deltas) / len(deltas) * 100
    avgs.append(deltaAvg)
    print(iter, ' Change Avg: ', deltaAvg)

print(sum(avgs) / len(avgs))

## Average improvement of x6%

##################################################################
### Weighted Average Improvement for Lefty and Righty Datasets ###
##################################################################

wAvg = ((16 * len(whiffDataRighty)) + (6 * len(whiffDataLefty))) / (len(whiffDataRighty) + len(whiffDataLefty))

## OVERALL: x13.4% improvement

#%%
#########################################
### PLATE SIDE ANALYSIS - ALTERNATIVE ###
#########################################

## Secondary analysis to corroborate trends and quantifications 
## Reasoning behind this is because previous analysis of Plate Side was WRT Swing Prob, not Whiff Prob

## Plate Side Right
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']

## Understand the distribution (previously confirmed to be normal)
if noisy:
    print(whiffDataRighty['PlateSide'].mean())
    print(whiffDataRighty['PlateSide'].std())

avgs = [] 
avgswo = []
for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataRighty[abs(whiffDataRighty.PlateSide - threshhold) < .2]['whiff_prob'].mean()

    if math.isnan(avg):
        continue

    if threshhold > -0.5 and threshhold < 1.1:
        avgs.append(avg)
    else:
        avgswo.append(avg)

    #print(threshhold, '--->', avg)

print('Without Optimal Range :', sum(avgswo) / len(avgswo))
print('With Optimal Range :', sum(avgs) / len(avgs))

## Findings: Righty Optimal Range: -0.5 to 1.1 feet


## Plate Side Left
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

## Understand the distribution (previously confirmed to be normal)
if noisy:
    print(whiffDataLefty['PlateSide'].mean())
    print(whiffDataLefty['PlateSide'].std())

avgs = [] 
avgswo = []
for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataLefty[abs(whiffDataLefty.PlateSide - threshhold) < .2]['whiff_prob'].mean()

    if math.isnan(avg):
        continue

    if threshhold > 0.25 and threshhold < 1.25:
        avgs.append(avg)
    else:
        avgswo.append(avg)

    #print(threshhold, '--->', avg)

print('Without Optimal Range :', sum(avgswo) / len(avgswo))
print('With Optimal Range :', sum(avgs) / len(avgs))

## Findings: Lefty Optimal Range: 0.25 to 1.25 feet





#%%
###############################################################
### SIMULATION WITH OPTIMAL PLATE SIDE AND PLATE HEIGHT - R ###
###############################################################


### Splitting By Right vs Left Side Batters
### Plate Side, Righty ### 

whiffDataRighty = whiffRegressionData[whiffRegressionData.BatterSide_Right == 1]

### Generate Baseline Model
## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob', 'PlateSide']


## Baseline Model (with all variables included)
## Generate Model
X = whiffDataRighty[varsToInclude]
y = whiffDataRighty['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')


### Simulation - Plate Side, Righty
avgs = []
for iter in tqdm([1,2,3,4,5]):
    deltas = []
    for seed in list(range(1,101)):
        seed = random.randint(1,1000)
        
        ## Generate Model
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = seed)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)

        x_testDF = pd.DataFrame(x_test)
        x_testDF.columns = varsToInclude

        ## Generate Predictions
        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1, this methods retains the distribution
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        baselineAvg = sum(scaledVals) / len(scaledVals)

        ### Simulation - Change Values in Prediction Set
        x_testDF['PlateSide'] = random.uniform(-0.5, 1.1)  
        x_testDF['PlateHeight'] = random.uniform(0.9,1.1) 

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        changedAvg = sum(scaledVals) / len(scaledVals)

        ## We use percent change here because the actual prediction values have lost meaning
        ## One, because the prediction range is wider than the actual possible range of values (infinite range vs [0,1])
        ## Two, because we have normalized the values 
        delta = (changedAvg - baselineAvg) / baselineAvg 
        deltas.append(delta)


    deltaAvg = sum(deltas) / len(deltas) * 100
    avgs.append(deltaAvg)
    print(iter, ' Change Avg: ', deltaAvg)

print(sum(avgs) / len(avgs))










#%%
###############################################################
### SIMULATION WITH OPTIMAL PLATE SIDE AND PLATE HEIGHT - L ###
###############################################################

### Splitting By Right vs Left Side Batters
### Plate Side, Lefty ### 

whiffDataLefty = whiffRegressionData[whiffRegressionData.BatterSide_Left == 1]

### Generate Baseline Model
## Started off with all variables
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

## These were determined using backwards selection 
varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob', 'PlateSide']


## Baseline Model (with all variables included)
## Generate Model
X = whiffDataLefty[varsToInclude]
y = whiffDataLefty['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

## Score Model 
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')


### Simulation - Plate Side, Righty
avgs = []
for iter in tqdm([1,2,3,4,5]):
    deltas = []
    for seed in list(range(1,101)):
        seed = random.randint(1,1000)
        
        ## Generate Model
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = seed)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)

        x_testDF = pd.DataFrame(x_test)
        x_testDF.columns = varsToInclude

        ## Generate Predictions
        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1, this methods retains the distribution
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        baselineAvg = sum(scaledVals) / len(scaledVals)

        ### Simulation - Change Values in Prediction Set
        x_testDF['PlateSide'] = random.uniform(0.25,1.25)  
        x_testDF['PlateHeight'] = random.uniform(0.9,1.1) 

        preds = []
        for idx, row in x_testDF.iterrows(): 
            preds.append(regressor.predict(row.values.reshape(1,-1)))

        ## Normalize Data to within 0 and 1 
        minVal = min(preds)[0]
        maxVal = max(preds)[0]
        rangeVal = maxVal - minVal
        scaledVals = []
        for val in preds:
            val = val[0]
            scaledVal = (val - minVal) / (rangeVal)
            scaledVals.append(scaledVal)

        changedAvg = sum(scaledVals) / len(scaledVals)

        ## We use percent change here because the actual prediction values have lost meaning
        ## One, because the prediction range is wider than the actual possible range of values (infinite range vs [0,1])
        ## Two, because we have normalized the values 
        delta = (changedAvg - baselineAvg) / baselineAvg 
        deltas.append(delta)


    deltaAvg = sum(deltas) / len(deltas) * 100
    avgs.append(deltaAvg)
    print(iter, ' Change Avg: ', deltaAvg)

print(sum(avgs) / len(avgs))

## Average Improvement 28%

########################
### Weighted Average ###
######################## 
wAvg = ((10 * len(whiffDataRighty)) + (28 * len(whiffDataLefty))) / (len(whiffDataRighty) + len(whiffDataLefty))




#%%
###################################
### DISTRIBUTION VISUALIZATIONS ###
###################################

## Plate Side
whiffDataLefty['PlateSide'].hist(bins = 30)
whiffDataRighty['PlateSide'].hist(bins = 30)

## Plate Height
whiffData['PlateHeight'].hist(bins = 30)