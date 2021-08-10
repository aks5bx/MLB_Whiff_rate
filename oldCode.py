
#%%

## Score Model 
print('>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<')

## Model with one or two variables removed (loop can be edited)
rsqValues = []
for variable in tqdm(varsToInclude):
    ## The second loop can be used to understand the impacts of interaction terms (or omitted)
     for variable2 in varsToInclude:
        varList = list(set(varsToInclude) - set([variable, variable2]))
        
        ## Generate Model
        X = whiffModelData[varList]
        y = whiffModelData['whiff_prob'].values.reshape(-1,1)

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
X = whiffModelData[varsToInclude]
y = whiffModelData['whiff_prob'].values.reshape(-1,1)

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
X = whiffModelData[varsToInclude]
y = whiffModelData['swing_prob'].values.reshape(-1,1)

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
        X = whiffModelData[varList]
        y = whiffModelData['swing_prob'].values.reshape(-1,1)

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
whiffDataRighty = whiffModelData[whiffModelData.BatterSide_Right == 1]
whiffDataLefty = whiffModelData[whiffModelData.BatterSide_Left == 1]

# print('swingProb', whiffModelData['swing_prob'].corr(whiffDataRighty['whiff_prob']))

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
    print(whiffModelData['PlateSide'].corr(whiffModelData['swing_prob']))
    print('Right', whiffDataRighty['PlateSide'].corr(whiffDataRighty['swing_prob']))
    print('Left', whiffDataLefty['PlateSide'].corr(whiffDataLefty['swing_prob']))

### Splitting By Right vs Left Side Batters
### Plate Side, Righty ### 

whiffDataRighty = whiffModelData[whiffModelData.BatterSide_Right == 1]

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

whiffDataLefty = whiffModelData[whiffModelData.BatterSide_Left == 1]

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

whiffDataRighty = whiffModelData[whiffModelData.BatterSide_Right == 1]

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

whiffDataLefty = whiffModelData[whiffModelData.BatterSide_Left == 1]

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
        x_testDF['PlateSide'] = random.uniform(-0.5,1.1)  
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
wAvg = ((25 * len(whiffDataRighty)) + (28 * len(whiffDataLefty))) / (len(whiffDataRighty) + len(whiffDataLefty))

## The data confirms that when combined, these changes increase whiff prob by x26%



#%%
###################################
### DISTRIBUTION VISUALIZATIONS ###
###################################

## Plate Side
whiffDataLefty['PlateSide'].hist(bins = 30)
whiffDataRighty['PlateSide'].hist(bins = 30)

## Plate Height
whiffData['PlateHeight'].hist(bins = 30)