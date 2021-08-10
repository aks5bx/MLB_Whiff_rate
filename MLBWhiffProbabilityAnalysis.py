#%%
###################################################
### IMPORTING LIBRARIES TO USE, READING IN DATA ###
###################################################
 
## General Tools 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import math 
import random 

## Sklearn Models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

## Specialized Tools
from scipy import stats
from collections import Counter
import statsmodels.api as sm
from tqdm import tqdm
from boruta import BorutaPy

## Reading in the Data Set
whiffData = pd.read_csv('PitcherXData.csv')

## When "noisy" is set to True, the notebook prints out additional output
noisy = True
## When maxOutput is set to True, all graphs and plots are produced
maxOutput = False

#%%
#############################################
### INITIAL DATA EXPLORATION AND CLEANING ### 
#############################################

## Identify categorical variable columns and numeric variable columns
numericCols = whiffData.select_dtypes(include=np.number).columns
categoricalCols = list(set(whiffData.columns) - set(numericCols))

## Print out the columns for viewing
if noisy:
    print('Numeric Columns : ', numericCols)
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')
    print('Categorical Columns : ', categoricalCols)
    print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')

## Detect Null values, this data set that contains only rows where one or more values are Null is the "Null Dataset"
whiffNulls = whiffData[whiffData.isnull().any(axis=1)]

## Confirm that Null values do not have significant skew before dropping null rows
## We want to avoid a situation where the values in the null rows significantly change the overall data

## For each numeric column, compare the mean value of the full dataset to the mean value of the Null Dataset  
for col in numericCols: 
    originalMean = whiffData[col].mean()
    nullMean = whiffNulls[col].mean()

    if noisy:
        print(col, ': ', originalMean, ' ', nullMean)

## For each categorical column, compare the values that show up in the entire data set versus the null dataset 
for col in categoricalCols: 
    originalSet = set(whiffData[col])
    nullSet = set(whiffNulls[col])

    if noisy:
        print('>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<')
        print(col)
        print(originalSet)
        print(nullSet)

## Worth looking into InducedVertBreak and HorzBreak from first glance
## Induced vertical break has a standard deviation of around 6.5 - the null rows' aggregate InducedVertBreak value is within 1 std of the main set 
whiffData['InducedVertBreak'].std()

## The null row's aggregate Horizontal Break value is over one standard deviation, but not by much. It is close, but acceptable to drop - very little skew
whiffData['HorzBreak'].std()

## All the numeric columns look like they display no skew when comparing the null rows and the non null rows (main set), so we drop the nulls
whiffData = whiffData.dropna()

## Spin Axis should be from 0 - 360 - there are ~100 incorrectly entered rows 
## We cannot guess what the correct value is so we drop these 
whiffData = whiffData[whiffData.SpinAxis >= 0]

#%% 
######################################
### OUTLIER DETECTION AND HANDLING ###
######################################  

## For numeric variables, do outlier detection
whiffNumeric = whiffData[numericCols]

## For each column, detect outliers (more than 3 standard deviations from the mean) 
for col in numericCols:
    outliers = whiffNumeric[~(np.abs(whiffNumeric[col] - whiffNumeric[col].mean()) < (3 *whiffNumeric[col].std()))]
    
    if len(outliers) > 0 and noisy:
        print(col, len(outliers))
        print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<')


## After a cursory analysis of the columns with outliers (looking at the range and standard deviation), only SpinRate is of note 
## Problem: several zeros in data (very unlikely), around ~50 rows 
## Options: impute data OR get rid of 50 more rows. To decide, we conduct another skew analysis 

## Subset the dataset to the cases where Spin Rate is 0, we'll call this the Spin 0 Dataset
spinZero = whiffData[whiffData.SpinRate == 0]

## For each column, compare mean values from the Full Dataset to the Spin0 Dataset
for col in numericCols: 
    originalMean = whiffData[col].mean()
    zeroMean = spinZero[col].mean()

    if noisy: 
        print(col, ': ', originalMean, ' ', zeroMean)

## For each categorical column, compare the values that show up in the Full Dataset versus the Spin0 Dataset
for col in categoricalCols: 
    originalSet = set(whiffData[col])
    zeroSet = set(spinZero[col])

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

## Identify high correlation pairs
high_corrs = []
for idx,row in corrMatrix.iterrows(): 
    for col in corrMatrix.columns: 
        if (row[col] < 1) & (row[col] > 0.75): 
            high_corrs.append(col)

if noisy:
    print(Counter(high_corrs))

## Pitch of Plate Appearance is correlated with Balls and Strikes, this we need to solve 
## Release Speed is correlated with Induced Vertical Break (these are considered distinct, so we keep both, despite the correlation)

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
whiffModelData = whiffData.drop(['Date', 'Year'], axis = 1)

## Here we want to create dummy variables for our categorical variables 
## We choose this over other techniques, such as target encoding because we don't have too many categorical variables
whiffModelData = pd.get_dummies(whiffModelData)

## WhiffModelData (full set) gets split up into WhiffModelData X and WhiffModelData Y
whiffModelDataY = whiffModelData[['whiff_prob']]

## A pitcher cannot directly influence swing probability OR whiff proability given swing, so we don't include them in the model
whiffModelDataX = whiffModelData.loc[:, whiffModelData.columns != 'whiff_prob']
whiffModelDataX = whiffModelDataX.loc[:, whiffModelDataX.columns != 'swing_prob']
whiffModelDataX = whiffModelDataX.loc[:, whiffModelDataX.columns != 'whiff_prob_gs']

## Changing the nature of SpinAxis
## Spin axis is not a true continous variable (a higher number does not mean "more spin" it instead means an entirely different kind of spin)
whiffModelDataX['TopSpin'] = 0
whiffModelDataX['LeftSpin'] = 0
whiffModelDataX['BackSpin'] = 0
whiffModelDataX['RightSpin'] = 0

## Here we identify the type of spin and then, if necessary, input the magnitude of the spin (ranging from 0 - 90)
def topSpinMagnitude(spinAxis): 
    if spinAxis >= 0 and spinAxis < 90: 
        topSpinDiff = abs(spinAxis - 0)
        topSpinMagnitude = abs(90 - topSpinDiff)
        return topSpinMagnitude
    else:
        return 0 

def leftSpinMagnitude(spinAxis): 
    if spinAxis > 0 and spinAxis < 180: 
        leftSpinDiff = abs(spinAxis - 90)
        leftSpinMagnitude = abs(90 - leftSpinDiff)
        return leftSpinMagnitude
    else:
        return 0 

def backSpinMagnitude(spinAxis): 
    if spinAxis > 90 and spinAxis < 270: 
        backSpinDiff = abs(spinAxis - 180)
        backSpinMagnitude = abs(90 - backSpinDiff)
        return backSpinMagnitude
    else:
        return 0 

def rightSpinMagnitude(spinAxis): 
    if spinAxis > 180 and spinAxis < 360: 
        rightSpinDiff = abs(spinAxis - 270)
        rightSpinMagnitude = abs(90 - rightSpinDiff)
        return rightSpinMagnitude
    else:
        return 0 


whiffModelDataX['TopSpin'] = whiffModelDataX.apply(lambda row : topSpinMagnitude(row['SpinAxis']), axis = 1)
whiffModelDataX['LeftSpin'] = whiffModelDataX.apply(lambda row : leftSpinMagnitude(row['SpinAxis']), axis = 1)
whiffModelDataX['BackSpin'] = whiffModelDataX.apply(lambda row : backSpinMagnitude(row['SpinAxis']), axis = 1)
whiffModelDataX['RightSpin'] = whiffModelDataX.apply(lambda row : rightSpinMagnitude(row['SpinAxis']), axis = 1)


#%% 
####################################
### TARGET DISTRIBUTION ANALYSIS ###
#################################### 

## Visualize the target variable (no normal distribution based assumptions hold)
whiffModelData['whiff_prob'].hist(bins = 20)

## Define skew 
val = 0
for i in range(0, 10): 
    if noisy:
        print('Min :', val)
        print(len(whiffModelData[(whiffModelData.whiff_prob < val + .05) & (whiffModelData.whiff_prob > val)]))
        print('Max :', round(val,1) + .05)
        print('---------------')

    val += .05

## This is purely for Exploratory Data Analysis to categorize and plot the target variable
whiffModelData['whiff_prob_category'] = 0
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.05) & (whiffModelData['whiff_prob'] > 0), 1, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.1) & (whiffModelData['whiff_prob'] > 0.05), 2, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.15) & (whiffModelData['whiff_prob'] > 0.1), 3, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.2) & (whiffModelData['whiff_prob'] > 0.15), 4, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.25) & (whiffModelData['whiff_prob'] > 0.2), 5, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.3) & (whiffModelData['whiff_prob'] > 0.25), 6, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] < 0.35) & (whiffModelData['whiff_prob'] > 0.3), 7, whiffModelData['whiff_prob_category'])
whiffModelData['whiff_prob_category'] = np.where((whiffModelData['whiff_prob'] > 0.35), 8, whiffModelData['whiff_prob_category'])

## Aggregate each variable 
groupedMeans = pd.DataFrame(whiffModelData.groupby(['whiff_prob_category']).mean())
groupedMeans['category'] = groupedMeans.index

groupedMeans[['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER']]

#%%
#####################################
### FEATURE SELECTION WITH BORUTA ###
#####################################

### Initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 7, 
   verbose = 2
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 500, # number of trials to perform, 
   verbose = 2
)
### modify datatype for Boruta (it accepts np.array, not pd.DataFrame)
whiffModelDataX_Arr = whiffModelDataX.to_numpy()
whiffModelDataY_Arr = whiffModelDataY.to_numpy()

## Boruta has already been run so for future runs of this notebook we avoid a re-run
## To re-run Boruta, switch runBortua to True
runBoruta = False


if runBoruta:
    boruta.fit(whiffModelDataX_Arr, whiffModelDataY_Arr)

    ## Green Area variables have been cleared as significant, blue area variables are still uncertain
    green_area = whiffModelDataX.columns[boruta.support_].to_list()
    blue_area = whiffModelDataX.columns[boruta.support_weak_].to_list()
    print('features in the green area:', green_area)
    print('features in the blue area:', blue_area)

## Here are the columns to keep based on the boruta analysis 
## Note: BatterSide was not shown as significant but was retained because of its impact on various other variables such as PlateSide, HorzBreak, etc
columnsToKeep = [   'ReleaseSpeed',
                    'InducedVertBreak',
                    'HorzBreak',
                    'Extension',
                    'PlateHeight',
                    'PlateSide',
                    'SpinRate',
                    'SpinAxis',
                    'BatterSide_Right',
                    'BatterSide_Left',
                    'PitchType_CHANGEUP',
                    'PitchType_FASTBALL',
                    'PitchType_SLIDER',
                    'Count_00', 
                    'Count_01', 
                    'Count_02', 
                    'Count_10', 
                    'Count_11', 
                    'Count_12',
                    'Count_20', 
                    'Count_21', 
                    'Count_22', 
                    'Count_30', 
                    'Count_31', 
                    'Count_32',
                    'LeftSpin', 
                    'RightSpin',
                    'BackSpin',
                    'TopSpin',
                    'BatterSide_Left', 
                    'BatterSide_Right']

# %%
###############################################
### CHECKING VARIABLE CORRELATION FOR MODEL ###
############################################### 

runPairPlot = False

if runPairPlot:
    ## Checking Correlation (takes a long time to run)
    pairPlot = sns.pairplot(data=whiffModelData,
                    y_vars=['whiff_prob'],
                    x_vars=columnsToKeep)


#%%
#######################################
### MODEL IMPLEMENTATION Parameters ###
#######################################

## Baseline Model (with all variables included)
## Generate Model
X = whiffModelDataX[columnsToKeep]
y = whiffModelDataY['whiff_prob'].values.reshape(-1,1)

## Choosing to not scale in order to preserve the target variable range 
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 20)

#%%
#################################
### MODEL IMPLEMENTATION, SVR ###
#################################

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)
print('BASELINE MODEL SCORE (R2) :', regressor.score(x_test,y_test))


#%%
#################################
### MODEL IMPLEMENTATION, MLR ###
#################################

reg = LinearRegression().fit(x_train, y_train)
reg.score(x_test, y_test)


#%%
###########################################
### MODEL IMPLEMENTATION, RANDOM FOREST ###
###########################################

## Random Grid search has already been run, to re-run, turn runRandomGrid to True
runRandomGrid = False 

if runRandomGrid:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 25, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(x_train, y_train)
    rf_random.score(x_test, y_test)

regr = RandomForestRegressor(max_depth=15, n_estimators = 2000, random_state= 1)
regr.fit(x_train, y_train)
regr.score(x_test,y_test)

for i,v in enumerate(regr.feature_importances_):
	print('Feature: %0d, Score: %.5f' % (i,v))

featureImportanceDict = {'Features':columnsToKeep,'Importances':list(regr.feature_importances_)}
featureImportanceDictDF = pd.DataFrame(featureImportanceDict, columns=['Features','Importances'])

sns.barplot(y='Features', x='Importances', data=featureImportanceDictDF)

#%%
##########################
### AVERAGE PREDICTION ###
##########################

## Average whiff_prob prediction generated by the model, before any modification
beforeMod = sum(regr.predict(x_test)) / len(regr.predict(x_test))
beforeMod

#%%
####################################
### SIGNIFICANT FEATURE ANALYSIS ###
####################################

### The Feature importance chart underscored Pitch Type (Fastball), Plate Side, Plate Height as significant features 
### Here, we analyze how these variables can be leveraged to improve whiff_probability

##################
### Plate Side ###
##################

## Artificially Shift PlateSide value lower (towards left batter box) and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left : ', afterMod)

## Artificially Shift PlateSide value higher (towards right batter box) and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] + (x_test_copy1['PlateSide'] / 3)
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Right : ', afterMod)

## Shifting plate side by 25-35% towards the left handed box improves whiff prob by almost a full 1% 

####################
### Plate Height ###
#################### 

## Artificially Shift PlateHeight up and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] + (x_test_copy1['PlateHeight'] / 3)
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Up : ', afterMod)

## Artificially Shift PlateHeight down and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Down : ', afterMod)

## Shifting plate height by 25-35% towards the left handed box improves whiff prob by almost a full 1% 

##################
### Pitch Type ###
##################

## Replace Fastballs with Sliders and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_SLIDER'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Slider : ', afterMod)

## Shift Fastballs with Changeups and see how the average whiff_prob prediction changes
x_test_copy1 = x_test.copy()
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_CHANGEUP'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Changeup : ', afterMod)

## Replacing Fastballs with Sliders/Changeups can increase whiff probability by almost 1.5% on a given pitch

###########################
### Combining Variables ###
###########################

## Shift PlateSide value lower (left batter box) and Plate Height lower
x_test_copy1 = x_test.copy()
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left + Low : ', afterMod)

## About +1.5% 


### Shift Plate Height and Side Low and to the Left, make the pitch a Slider
x_test_copy1 = x_test.copy()
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_SLIDER'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left + Low, Slider : ', afterMod)

## 18.6% - +7% 

### Shift Plate Height and Side Low and to the Left, make the pitch a Changeup
x_test_copy1 = x_test.copy()
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_CHANGEUP'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left + Low, Changeup : ', afterMod)

## 18.4% - +7% 

########################
## Check: Batter Side ##
########################

### Low and to the Left, Slider versus Lefties
x_test_copy1 = x_test.copy()
x_test_copy1['BatterSide_Left'] = 1
x_test_copy1['BatterSide_Right'] = 0
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_CHANGEUP'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left + Low, Changeup verses Lefties : ', afterMod)

### Low and to the Left, Slider versus Righties
x_test_copy1 = x_test.copy()
x_test_copy1['BatterSide_Left'] = 0
x_test_copy1['BatterSide_Right'] = 1
x_test_copy1['PlateSide'] = x_test_copy1['PlateSide'] - (x_test_copy1['PlateSide'] / 3)
x_test_copy1['PlateHeight'] = x_test_copy1['PlateHeight'] - (x_test_copy1['PlateHeight'] / 3)
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_CHANGEUP'] = 1
x_test_copy1.loc[x_test_copy1['PitchType_FASTBALL'] > 0, 'PitchType_FASTBALL'] = 0
afterMod = sum(regr.predict(x_test_copy1)) / len(x_test_copy1)
print('Left + Low, Changeup verses Righties : ', afterMod)

## Generally the same

# %%
################
### FINDINGS ###
################

## We can produce a model with a R2 score of around 0.85 in order to predict whiff probability given a particular pitch 
## Out model predicts that changes to Plate Height, Plate Side, and Pitch Type could yield a 60% increase in whiff probability (all else equal) 