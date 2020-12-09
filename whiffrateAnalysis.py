#%%
### IMPORTING LIBRARIES TO USE, READING IN DATA ### 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from scipy import stats
from collections import Counter
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split



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

#%%
### OVERALL TREND ANALYSIS ### 

## Using whiffDataFull here 


# %%
### FEATURE ENGINEERING ### 

## For this analysis, we don't use Year or Date, given that we expect some change in the pitcher's behavior, it does not make sense to extrapolate trends over time (changes in behavior can change the nature of these trends)
whiffRegressionData = whiffData.drop(['Date', 'Year'], axis = 1)

## Here we want to create dummy variables for our categorical variables 
## We choose this over other techniques, such as target encoding because we don't have too many categorical variables
whiffRegressionData = pd.get_dummies(whiffRegressionData)


# %%
### CHECKING ASSUMPTIONS FOR REGRESSION ### 

## Vaguely linear relationship 
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

## Multicollinearity has already been checked 

## Heteroskedasticity will be checked after model is generated 


#%%
### MODEL IMPLEMENTATION, SVR - WHIFF ###

varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

varsToInclude = ['ReleaseSpeed', 'InducedVertBreak', 'HorzBreak', 'PlateHeight', 'SpinRate', 'SpinAxis', 'swing_prob']

rsqValues = []
for variable in varsToInclude:
     #for variable2 in varsToInclude:
        varList = list(set(varsToInclude) - set([variable]))
        ## Here we remove the whiff probability (gs and non-gs) values
        X = whiffRegressionData[varList]
        y = whiffRegressionData['whiff_prob'].values.reshape(-1,1)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)
        score = regressor.score(x_test, y_test)
        rsqValues.append(score)

        if score < 0.6:
            print(variable, score)

## Plate Height and Swing Prob combine for 63%, essentially even contribution (slightly more towards plate height)

### Significant variables: 
sigVarsWhiff = ['PlateHeight', 'swing_prob']

### Actual model
## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 25)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)
print(regressor.score(x_test,y_test))




#%%
### PLATE HEIGHT - ANALYSIS ### 

### Simulation - Whiff Rate Baseline
deltas = []
avgs = []
for std in tqdm([1,2,3,4,5]):
    for seed in list(range(1,51)):

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
    print('Change Avg: ', deltaAvg)

## Sanity check
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

print(whiffDataRighty['PlateHeight'].corr(whiffDataRighty['swing_prob']))
print(whiffDataLefty['PlateHeight'].corr(whiffDataLefty['swing_prob']))








#%%

whiffRegressionData.columns


#%%
### SWING PROB ###

## SVR

varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']

# varsToIncludeSwing = []
# for var in varsToInclude:
#     corr = whiffRegressionData[var].corr(whiffRegressionData['swing_prob'])
#     if corr > 0.075 or corr < -0.075:
#         print(var)
#         varsToIncludeSwing.append(var)


rsqValues = []
for variable in varsToInclude:
     for variable2 in varsToInclude:
        varList = list(set(varsToInclude) - set([variable, variable2]))
        ## Here we remove the whiff probability (gs and non-gs) values
        X = whiffRegressionData[varList]
        y = whiffRegressionData['swing_prob'].values.reshape(-1,1)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

        regressor = SVR(kernel='rbf')
        regressor.fit(x_train,y_train)
        score = regressor.score(x_test, y_test)
        rsqValues.append(score)

        print(score)

        if score < 0.6:
            print(variable, variable2, score)

#%%
## Actual Model 
## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['swing_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)
score = regressor.score(x_test, y_test)
rsqValues.append(score)

print(score)

## Count is +6%, Plate Side is 33% 



#%%

### SWING PROB ANALYSIS PART II ### 

### RESULT: Plate Side and Count have the biggest impacts 

## Plate Side
print(whiffRegressionData['PlateSide'].corr(whiffRegressionData['swing_prob']))

## Let's split based off batter side 
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

print('Right', whiffDataRighty['PlateSide'].corr(whiffDataRighty['swing_prob']))
print('Left', whiffDataLefty['PlateSide'].corr(whiffDataLefty['swing_prob']))

print('swingProb', whiffRegressionData['swing_prob'].corr(whiffDataRighty['whiff_prob']))


## Count 
print('COUNT')
print(whiffRegressionData['Count_00'].corr(whiffDataRighty['swing_prob']))
print(whiffRegressionData['Count_01'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_02'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_10'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_11'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_12'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_20'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_21'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_22'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_30'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_31'].corr(whiffRegressionData['swing_prob']))
print(whiffRegressionData['Count_32'].corr(whiffRegressionData['swing_prob']))



#%%

### Simulation - Swing Rate Baseline
deltas = []
avgs = []
for std in tqdm([1,2,3,4,5]):
    for seed in list(range(1,51)):

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
        x_testDF['PlateHeight'] = x_testDF['PlateHeight'] - (std * oneStDev) 

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
    print('Change Avg: ', deltaAvg)
    print('------------------------------')

## Sanity check
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

print(whiffDataRighty['PlateHeight'].corr(whiffDataRighty['swing_prob']))
print(whiffDataLefty['PlateHeight'].corr(whiffDataLefty['swing_prob']))








#%%


### Plate Side, Righty

varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']


whiffDataRighty = whiffRegressionData[whiffRegressionData.BatterSide_Right == 1]

X = whiffDataRighty[varsToInclude]
y = whiffDataRighty['swing_prob'].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)
score = regressor.score(x_test, y_test)
rsqValues.append(score)

print(score)


### Simulation - Plate Side, Righty
deltas = []
avgs = []
for std in tqdm([1,2,3,4,5]):
    for seed in list(range(1,101)):

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
        x_testDF['PlateHeight'] = random.uniform(-0.5,1.1) #x_testDF['PlateHeight'] - (std * oneStDev) 

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
    print('Change Avg: ', deltaAvg)
    print('------------------------------')








#%%
### Plate Height Correlation 

print(whiffDataRighty['PlateHeight'].corr(whiffDataRighty['whiff_prob']))
print(whiffDataLefty['PlateHeight'].corr(whiffDataLefty['whiff_prob']))






#%%

## Plate Side 
whiffDataRighty = whiffData[whiffData.BatterSide == 'Right']
whiffDataLefty = whiffData[whiffData.BatterSide == 'Left']

print(whiffDataRighty['PlateSide'].mean())
print(whiffDataRighty['PlateSide'].std())

for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataRighty[abs(whiffDataRighty.PlateSide - threshhold) < .2]['whiff_prob'].mean()

    print(threshhold, '              ', avg)



## Righty Optimal Range: -0.5 to 1.1



print(whiffDataLefty['PlateSide'].mean())
print(whiffDataLefty['PlateSide'].std())


for threshhold in list(np.linspace(-3,3,100)):
    avg = whiffDataLefty[abs(whiffDataLefty.PlateSide - threshhold) < .2]['whiff_prob'].mean()

    print(threshhold, '              ', avg)

## Lefty Optimal Range: 0.25 to 1.25 

#%%
#whiffDataLefty['PlateSide'].hist(bins = 30)
whiffDataRighty['PlateSide'].hist(bins = 30)





# %%
### MODEL IMPLEMENTATION, REGRESSION - WHIFF ### 
## First Run
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis', 'swing_prob',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']
## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'] 

## Adding a constant term 
X = sm.add_constant(X) 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())



## Given that there are a few variables that have a weak correlation with whiff_prob, we remove these variables and re-run the algorithm with the idea of producing a clearly prediction with only variables that are significantly correlated 

## Iteration Two 
varsToInclude = list(set(varsToInclude) - set(['Inning', 'PAofInning', 'ReleaseSide', 'SpinAxis', 'HorzBreak']))

## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'] 

## Adding a constant term 
X = sm.add_constant(X) 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())


## Iteration Three 
varsToInclude = list(set(varsToInclude) - set(['Extension', 'ReleaseHeight', 'Count_00', 'Count_01', 'Count_02', 'Count_10', 'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30', 'Count_31', 'Count_32']))

## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'] 

## Adding a constant term 
X = sm.add_constant(X) 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

sigVars_Whiff = ['swing_prob', 'PitchType_CHANGEUP', 'SpinRate', 'BatterSide_Right', 'InducedVertBreak', 'ReleaseSpeed', 'PlateSide', 'PitchType_SLIDER', 'PlateHeight', 'PitchType_FASTBALL', 'BatterSide_Left']

## Swing Probability really helps with Whiff (understandably, so in additional to how to increase swing_prob), we want to know how to increase swing probability itself 


# %%
### MODEL IMPLEMENTATION, REGRESSION - SWING PROB ### 
## First Run
varsToInclude = ['Inning', 'PAofInning', 'ReleaseSpeed', 'InducedVertBreak', 'HorzBreak',
                            'ReleaseHeight', 'ReleaseSide', 'Extension', 'PlateHeight', 'PlateSide',
                            'SpinRate', 'SpinAxis',
                            'BatterSide_Left',
                            'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL',
                            'PitchType_SLIDER', 'Count_00', 'Count_01', 'Count_02', 'Count_10',
                            'Count_11', 'Count_12', 'Count_20', 'Count_21', 'Count_22', 'Count_30',
                            'Count_31', 'Count_32']
## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['swing_prob'] 

## Adding a constant term 
X = sm.add_constant(X) 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

## Given that there are a few variables that have a weak correlation with whiff_prob, we remove these variables and re-run the algorithm with the idea of producing a clearly prediction with only variables that are significantly correlated 

## Iteration Two 
varsToInclude = list(set(varsToInclude) - set(['Inning', 'PAofInning', 'ReleaseSpeed', 'ReleaseSide', 'Extension', 'SpinAxis','BatterSide_Left', 'BatterSide_Right', 'PitchType_CHANGEUP', 'PitchType_FASTBALL', 'PitchType_SLIDER']))

## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'] 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

## Iteration Three 
varsToInclude = list(set(varsToInclude) - set(['SpinRate']))

## Here we remove the whiff probability (gs and non-gs) values
X = whiffRegressionData[varsToInclude]
y = whiffRegressionData['whiff_prob'] 

## Run model 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
print(model.summary())


sigVars_Swing = ['HorzBreak', 'InducedVertBreak', 'PlateSide', 'ReleaseHeight', 'PlateHeight']


# %%
### SIGNIFICANT VARIABLE ANALYSIS - WHIFF ###
# sigVars_Whiff

sns.scatterplot(data = whiffRegressionData, x = 'PitchType_FASTBALL', y = 'whiff_prob')
# %%
