# Improving a Pitcher's Whiff Rate

## Overview 

In this project, I use pitch data for an anonymous pitcher and make recommendations on how the pitcher should improve their Whiff Rate. A "Whiff" is when a batter swings at a pitch and misses. Whiff Rate is the number of swings and misses divided by the total number of swings at balls thrown by a given pitcher. 

Improving Whiff Rate is important for MLB pitchers as it increases their strikeout rate and also prevents walks. A 2017 article by Bleacher Report reported some of the best Whiff Rates in the MLB around 25-30%. 

## Data Sourcing

The data was sourced from an unnamed Major League Baseball team and is considered representative of real MLB pitch data. The pitcher was not specifically identified by the source. 

The data was provided in a single csv file with additional metadata provided in a follow-up PDF. 

## Data Overview

The data contains many columns and so a preview of the data is not included. However, the original data source is included in the repo in the file PitcherXData.csv. 

The columns in the dataset are as follows: 

Variable | Explanation | 
| --- | --- |
Date| Date pitch was thrown on|
Year| Season that pitch was from|
Pitcher| Pitcher identity|
PitcherThrows| Pitcher handedness|
Inning| nth inning of the game|
PAofInning| nth batter of the inning|
BatterSide| Batter handedness|
PitchofPA| nth pitch thrown to this batter in this plate appearance|
Balls| Balls in count before pitch is thrown|
Strikes| Strikes in count before pitch is thrown|
PitchType| The classification of the pitch thrown|
ReleaseSpeed| Velocity of pitch|
InducedVertBreak| Vertical movement of pitch, in inches, where zero is a "gravity-only" trajectory|
HorzBreak| Horizontal movement of pitch, in inches|
ReleaseHeight| Distance above ground pitch was released from, in feet|
ReleaseSide| Distance from center of mound pitch was release from, in feet, positive numbers indicate closer to 3B side of diamond|
Extension| Distance in front of the mound (towards the plate) that pitch was released, in feet|
PlateHeight| Distance above ground that pitch crosses the plate, in feet, zero is at ground level|
PlateSide| Distance from centerline of plate that pitch crosses, in feet, positive numbers indicate closer to right-handed batter's box|
SpinRate| How fast the pitch was spinning, in rotations per minute|
SpinAxis| The angle or "tilt" that the pitch was spinning at; 0 is true topsin; 180 pure backspin; 90 sidespin towards left-handed batter's box; 270 sidespin towards right-handed batter's box|
swing_prob| The probability that given the metrics above, this pitch will result in a swing|
whiff_prob_gs| The probability that given a swing, the hitter will miss (whiff probability conditional onswing)|
whiff_prob The| probability that given the metrics above, this pitch|

The target variable is the final variable listed: whiff_prob (whiff probability)

## Data Processing 

Detailed data processing steps are outlined as commented code in the python file mlbwhiffprobabilityanalysis.py. However, important components are underscored in this section. 

### Data Cleaning 

Some steps for data cleaning included: 
- Detecting null values
- Checking to see if null values were skewed in one direction - null values did not seem to skew in any particular direction, so null values were dropped
- Outlier detection and handlings 
- Checking to see if variable values made sense given the expected range of the variables 
- Verifying datatypes 

### Feature Engineering 

#### Count 
The first step I took towards feature engineering was producing a pairplot between all variables and checking for variable correlation. Pitch of Plate Appearance was correlated with Balls and Strikes (understandably). In order to fix this problem, I considered what these variables were truly conveying. I realized that the combination of all three of these variables form a well-known variable in the baseball world: Count. So, I took these three variables and created a new categorical variable that displayed the count of the plate appearance (more information about Count here: https://www.rookieroad.com/baseball/101/count/#:~:text=The%20count%20in%20baseball%20is,or%20more%20than%20two%20strikes). 

A couple of other variables were loosly correlated. However, after digging deeper into what the variables represented in a baseball context, I was able to determine that the variables are not significantly related. 

#### Other Steps 
Afterwards, I took a few additional steps to further feature engineering: 
- Removed unneeded variables (eg: Year and Date were removed because these are not variables that the pitcher can modify to improve performance) 
- Made categorical variables into dummy variables 
- Removed swing_prob and whiff_prob_gs; these variables are almost completely uncorrelated to whiff_prob and are not variables the pitcher can control; these variables are determined by the pitch attributes (and how they are determined is completely unknown), which makes them difficult to use and interpret 

#### Spin Axis
Next, I took a look at the variable spin_axis. spin_axis was given to me as a continuous variable from 0 to 360. However, I realized that increasing spin_axis did not mean increasing spin. Instead, increasing spin axis creates a completely different type of spin. From the metadata, we know: 0 is true topsin; 180 pure backspin; 90 sidespin towards left-handed batter's box; 270 sidespin towards right-handed batter's box. So, I took the variable spin_axis and broke it up into four variables; TopSpin, RightSpin, LeftSpin, and BackSpin. I then converted the spin_axis variable from a 0-360 range to a 0-90 range. For instance, if BackSpin is 0 that means there is no backspin on the ball. If BackSpin is 90, that means that there is true, pure backspin on the ball. This principle extends to all types of spin. By definition there cannot be backspin and topspin and there also cannot be rightspin and leftspin. 

### Feature Selection

In order to conduct feature selection, I utilized the Boruta method. Instead of manually pulling out variables that appeared unhelpful, the Boruta method of feature
extraction was put to use. Boruta functions essentially by taking each variable in a data set, creating a copy of that variable, randomizing the variable copy, and attaching it back to the dataset. So, if a dataset had columns A and B, Boruta takes that dataset and produces a new dataset with Columns A, A_Copy, B, and B_Copy. A_Copy and B_Copy however have been completely shuffled and therefore the columns do not match up to A and B (consider the copy columns essentially a random number generator sampling from the range of their original columns). 

What Boruta does next is it takes all of the copy columns and selects the copy column that performed the best in terms of predicting the dependent variable. Then, Boruta says that if any of the original columns was a worse indicator of the dependent variable than the best copy column, that original column gets tossed. This entire process is repeated many times to ensure statistical significance. The basic idea here is that no column in the data set is worse than simply having a random column of data – each column must have a non-random approach towards predicting the depending variable.

Once I had the features that Bortua selected as significant, I was ready to feed the data into a model. 

## Model Building

### Iteration 1: SVR

My first attempt was a Support Vector Machine Regression. I chose this because I sensed a potential radial relationship with some of the independent variables and the dependent variable. The R2 metric after many tunings of the model was weak, never crossing more than 0.3. So, I turned to something else. 

### Iteration 2: Logistic Regression

I turned to a comfortable and trusted method, logistic regression with an ordinary least squares loss function. This performed marginally better than the previous model, but still was not good enough. 

### Iteration 3: Random Forest Regressor

I realized that the SVR and Logistic Regression did not have any variable interaction. The data and context, however, did seem to lend itself to interaction. For example, a low ball pitched to a lefty could be different from a low ball pitched to a righty. A Decision Tree allows for conditional consideration. So, I decided to try a Random Forest Regressor. 

After utilizing a Random Grid Search to optimize parameters, I was able to find success with the Random Forest Regressor, producing an R2 metric of around 0.85. While this metric could be better, understanding that I had no information about the batters and had a small to medium sized dataset, I found this metric quite strong. So, I decided to claim victory with the Random Forest model. 

After running the model, I was also able to extract the most important features to the model, which are shown below: 


## Investigating Important Features
