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
- Date: Date pitch was thrown on
- Year: Season that pitch was from
- Pitcher: Pitcher identity
- PitcherThrows: Pitcher handedness
- Inning: nth inning of the game
- PAofInning: nth batter of the inning
- BatterSide: Batter handedness
- PitchofPA: nth pitch thrown to this batter in this plate appearance
- Balls: Balls in count before pitch is thrown
- Strikes: Strikes in count before pitch is thrown
- PitchType: The classification of the pitch thrown
- ReleaseSpeed: Velocity of pitch
- InducedVertBreak: Vertical movement of pitch, in inches, where zero is a "gravity-only" trajectory
- HorzBreak: Horizontal movement of pitch, in inches
- ReleaseHeight: Distance above ground pitch was released from, in feet
- ReleaseSide: Distance from center of mound pitch was release from, in feet, positive numbers indicate closer to 3B side of diamond
- Extension: Distance in front of the mound (towards the plate) that pitch was released, in feet
- PlateHeight: Distance above ground that pitch crosses the plate, in feet, zero is at ground level
- PlateSide: Distance from centerline of plate that pitch crosses, in feet, positive numbers indicate closer to right-handed batter's box
- SpinRate: How fast the pitch was spinning, in rotations per minute
- SpinAxis: The angle or "tilt" that the pitch was spinning at; 0 is true topsin; 180 pure backspin; 90 sidespin towards left-handed batter's box; 270 sidespin towards right-handed batter's box
- swing_prob: The probability that given the metrics above, this pitch will result in a swing
- whiff_prob_gs: The probability that given a swing, the hitter will miss (whiff probability conditional onswing)
- whiff_prob The: probability that given the metrics above, this pitch


