---
title: "Practical Machine Learning - Exercise Prediction"
author: "Nicolas Barea"
date: "26/12/2020"
output: 
  html_document: 
    keep_md: yes
editor_options: 
  chunk_output_type: inline
---
# Predicting weight lifting mistakes

## Introduction

Based on the data collected from six male participants with little weightlifting experience. They were asked to perform ten repetitions with a 1.25kg weight, in this five different ways:

-Class A: Exactly according to the specification.
-Class B: Throwing the elbows to the front.
-Class C: Lifting the dumbbell only halfway.
-Class D: Lowering the dumbbell only halfway.
-Class E: Throwing the hips to the front.

The goal of this analysis is to create a model than can predict the "Class" of the exercise based on the metrics collected. 

I downloaded two datasets, one called `train` with 19622 observations and 159 variables. I split the dataset into a `trainTrain` one with 75% of the observations which I will use to train the model, and another one called `trainTest`to perform cross validation and assess accuracy.

The `classe` that is the target of the prediction has to be converted to a factor for the classification algorithms to work:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.0.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 4.0.3
```

```r
library(ggplot2)

#Download data
train<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                        sep=",",head=T,row.names=1)
test<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                       sep=",",head=T,row.names=1)

set.seed(123)

#Break up the train data into a train and test sets for cross validation
inTrain = createDataPartition(train$classe, p = 3/4,list = FALSE)
trainTrain = train[inTrain,]
trainTest = train[-inTrain,]

#Make Classe a factor
trainTrain$classe<-as.factor(trainTrain$classe)
trainTest$classe<-as.factor(trainTest$classe)
```

## Data exploration - Dropping variables.

We have 158 potential predictors in our data. The `nearzeroVar` function in the Caret package in R diagnoses predictors that have one unique value (i.e. are zero variance predictors) or predictors that are have both of the following characteristics: they have very few unique values relative to the number of samples and the ratio of the frequency of the most common value to the frequency of the second most common value is large. I ran `nearzeroVar` in the `trainTrain` dataset and eliminated the variables that would add no information to the prediction model.

I eliminate the same variables from the `trainTest` dataset.

Many variables have a significant proportion of `NA` values, I eliminate from the datasets the one with more than 50% of them.



The resulting dataset includes 58 variables.

## Model fit using Gradient Boosting Machine

Our training set still has 57 predicting variables, so it will be a complex model. Boosting helps reduce variance and bias. The algorithm helps in the conversion of weak learners into strong learners by combining multiple number of learners.

I will use the Gradient Boosting Machine ("`gbm`" in the Caret package) algorithm to fit the model. 


```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1355
##      2        1.5204             nan     0.1000    0.0905
##      3        1.4613             nan     0.1000    0.0684
##      4        1.4165             nan     0.1000    0.0574
##      5        1.3798             nan     0.1000    0.0521
##      6        1.3466             nan     0.1000    0.0461
##      7        1.3176             nan     0.1000    0.0426
##      8        1.2909             nan     0.1000    0.0375
##      9        1.2654             nan     0.1000    0.0384
##     10        1.2413             nan     0.1000    0.0396
##     20        1.0549             nan     0.1000    0.0196
##     40        0.8349             nan     0.1000    0.0127
##     60        0.6967             nan     0.1000    0.0082
##     80        0.5945             nan     0.1000    0.0059
##    100        0.5192             nan     0.1000    0.0056
##    120        0.4557             nan     0.1000    0.0048
##    140        0.4035             nan     0.1000    0.0028
##    150        0.3825             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1953
##      2        1.4830             nan     0.1000    0.1349
##      3        1.3934             nan     0.1000    0.1238
##      4        1.3149             nan     0.1000    0.0988
##      5        1.2512             nan     0.1000    0.0745
##      6        1.2023             nan     0.1000    0.0887
##      7        1.1478             nan     0.1000    0.0737
##      8        1.1024             nan     0.1000    0.0626
##      9        1.0632             nan     0.1000    0.0622
##     10        1.0245             nan     0.1000    0.0577
##     20        0.7598             nan     0.1000    0.0317
##     40        0.4676             nan     0.1000    0.0147
##     60        0.3119             nan     0.1000    0.0081
##     80        0.2176             nan     0.1000    0.0056
##    100        0.1550             nan     0.1000    0.0041
##    120        0.1143             nan     0.1000    0.0027
##    140        0.0857             nan     0.1000    0.0014
##    150        0.0750             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2542
##      2        1.4475             nan     0.1000    0.1838
##      3        1.3310             nan     0.1000    0.1356
##      4        1.2424             nan     0.1000    0.1272
##      5        1.1624             nan     0.1000    0.1109
##      6        1.0925             nan     0.1000    0.1131
##      7        1.0220             nan     0.1000    0.0812
##      8        0.9703             nan     0.1000    0.0767
##      9        0.9227             nan     0.1000    0.0725
##     10        0.8788             nan     0.1000    0.0672
##     20        0.5769             nan     0.1000    0.0400
##     40        0.2866             nan     0.1000    0.0138
##     60        0.1580             nan     0.1000    0.0058
##     80        0.0971             nan     0.1000    0.0031
##    100        0.0653             nan     0.1000    0.0022
##    120        0.0451             nan     0.1000    0.0011
##    140        0.0332             nan     0.1000    0.0005
##    150        0.0288             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1298
##      2        1.5229             nan     0.1000    0.0886
##      3        1.4644             nan     0.1000    0.0667
##      4        1.4203             nan     0.1000    0.0511
##      5        1.3848             nan     0.1000    0.0533
##      6        1.3504             nan     0.1000    0.0448
##      7        1.3217             nan     0.1000    0.0433
##      8        1.2931             nan     0.1000    0.0385
##      9        1.2686             nan     0.1000    0.0362
##     10        1.2437             nan     0.1000    0.0399
##     20        1.0543             nan     0.1000    0.0199
##     40        0.8388             nan     0.1000    0.0122
##     60        0.7005             nan     0.1000    0.0090
##     80        0.5932             nan     0.1000    0.0061
##    100        0.5167             nan     0.1000    0.0049
##    120        0.4529             nan     0.1000    0.0037
##    140        0.4027             nan     0.1000    0.0039
##    150        0.3811             nan     0.1000    0.0039
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1944
##      2        1.4826             nan     0.1000    0.1336
##      3        1.3951             nan     0.1000    0.1172
##      4        1.3200             nan     0.1000    0.1029
##      5        1.2539             nan     0.1000    0.0929
##      6        1.1958             nan     0.1000    0.0836
##      7        1.1437             nan     0.1000    0.0693
##      8        1.1007             nan     0.1000    0.0626
##      9        1.0618             nan     0.1000    0.0545
##     10        1.0270             nan     0.1000    0.0599
##     20        0.7612             nan     0.1000    0.0276
##     40        0.4656             nan     0.1000    0.0210
##     60        0.3082             nan     0.1000    0.0102
##     80        0.2144             nan     0.1000    0.0044
##    100        0.1524             nan     0.1000    0.0045
##    120        0.1128             nan     0.1000    0.0028
##    140        0.0843             nan     0.1000    0.0012
##    150        0.0740             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2560
##      2        1.4474             nan     0.1000    0.1850
##      3        1.3303             nan     0.1000    0.1393
##      4        1.2402             nan     0.1000    0.1350
##      5        1.1546             nan     0.1000    0.1093
##      6        1.0859             nan     0.1000    0.0927
##      7        1.0273             nan     0.1000    0.0821
##      8        0.9759             nan     0.1000    0.0676
##      9        0.9321             nan     0.1000    0.0706
##     10        0.8893             nan     0.1000    0.0599
##     20        0.5879             nan     0.1000    0.0479
##     40        0.2956             nan     0.1000    0.0113
##     60        0.1666             nan     0.1000    0.0085
##     80        0.1023             nan     0.1000    0.0022
##    100        0.0681             nan     0.1000    0.0025
##    120        0.0470             nan     0.1000    0.0007
##    140        0.0346             nan     0.1000    0.0002
##    150        0.0299             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1291
##      2        1.5224             nan     0.1000    0.0910
##      3        1.4635             nan     0.1000    0.0693
##      4        1.4184             nan     0.1000    0.0555
##      5        1.3823             nan     0.1000    0.0515
##      6        1.3488             nan     0.1000    0.0450
##      7        1.3200             nan     0.1000    0.0451
##      8        1.2921             nan     0.1000    0.0346
##      9        1.2705             nan     0.1000    0.0385
##     10        1.2434             nan     0.1000    0.0413
##     20        1.0565             nan     0.1000    0.0224
##     40        0.8387             nan     0.1000    0.0137
##     60        0.7008             nan     0.1000    0.0084
##     80        0.5996             nan     0.1000    0.0068
##    100        0.5195             nan     0.1000    0.0032
##    120        0.4584             nan     0.1000    0.0031
##    140        0.4090             nan     0.1000    0.0034
##    150        0.3889             nan     0.1000    0.0038
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2001
##      2        1.4824             nan     0.1000    0.1458
##      3        1.3888             nan     0.1000    0.1131
##      4        1.3156             nan     0.1000    0.1016
##      5        1.2508             nan     0.1000    0.0839
##      6        1.1971             nan     0.1000    0.0840
##      7        1.1447             nan     0.1000    0.0787
##      8        1.0960             nan     0.1000    0.0675
##      9        1.0541             nan     0.1000    0.0597
##     10        1.0175             nan     0.1000    0.0522
##     20        0.7644             nan     0.1000    0.0338
##     40        0.4740             nan     0.1000    0.0126
##     60        0.3241             nan     0.1000    0.0100
##     80        0.2307             nan     0.1000    0.0068
##    100        0.1658             nan     0.1000    0.0037
##    120        0.1215             nan     0.1000    0.0024
##    140        0.0916             nan     0.1000    0.0019
##    150        0.0804             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2539
##      2        1.4463             nan     0.1000    0.1811
##      3        1.3301             nan     0.1000    0.1531
##      4        1.2339             nan     0.1000    0.1230
##      5        1.1543             nan     0.1000    0.1039
##      6        1.0875             nan     0.1000    0.1025
##      7        1.0229             nan     0.1000    0.0830
##      8        0.9709             nan     0.1000    0.0770
##      9        0.9229             nan     0.1000    0.0738
##     10        0.8769             nan     0.1000    0.0683
##     20        0.5731             nan     0.1000    0.0328
##     40        0.2965             nan     0.1000    0.0184
##     60        0.1662             nan     0.1000    0.0072
##     80        0.1029             nan     0.1000    0.0051
##    100        0.0672             nan     0.1000    0.0023
##    120        0.0463             nan     0.1000    0.0010
##    140        0.0341             nan     0.1000    0.0011
##    150        0.0294             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1341
##      2        1.5224             nan     0.1000    0.0869
##      3        1.4636             nan     0.1000    0.0669
##      4        1.4194             nan     0.1000    0.0560
##      5        1.3829             nan     0.1000    0.0488
##      6        1.3504             nan     0.1000    0.0493
##      7        1.3196             nan     0.1000    0.0379
##      8        1.2953             nan     0.1000    0.0453
##      9        1.2662             nan     0.1000    0.0365
##     10        1.2429             nan     0.1000    0.0381
##     20        1.0539             nan     0.1000    0.0208
##     40        0.8366             nan     0.1000    0.0113
##     60        0.6984             nan     0.1000    0.0079
##     80        0.5997             nan     0.1000    0.0061
##    100        0.5237             nan     0.1000    0.0068
##    120        0.4586             nan     0.1000    0.0052
##    140        0.4081             nan     0.1000    0.0042
##    150        0.3844             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1917
##      2        1.4834             nan     0.1000    0.1476
##      3        1.3905             nan     0.1000    0.1103
##      4        1.3204             nan     0.1000    0.0909
##      5        1.2618             nan     0.1000    0.0881
##      6        1.2058             nan     0.1000    0.0904
##      7        1.1505             nan     0.1000    0.0823
##      8        1.1006             nan     0.1000    0.0664
##      9        1.0597             nan     0.1000    0.0486
##     10        1.0284             nan     0.1000    0.0614
##     20        0.7600             nan     0.1000    0.0288
##     40        0.4702             nan     0.1000    0.0166
##     60        0.3136             nan     0.1000    0.0077
##     80        0.2179             nan     0.1000    0.0050
##    100        0.1567             nan     0.1000    0.0041
##    120        0.1163             nan     0.1000    0.0036
##    140        0.0876             nan     0.1000    0.0017
##    150        0.0774             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2616
##      2        1.4442             nan     0.1000    0.1773
##      3        1.3293             nan     0.1000    0.1374
##      4        1.2407             nan     0.1000    0.1320
##      5        1.1576             nan     0.1000    0.1140
##      6        1.0855             nan     0.1000    0.0940
##      7        1.0267             nan     0.1000    0.0845
##      8        0.9747             nan     0.1000    0.0693
##      9        0.9310             nan     0.1000    0.0781
##     10        0.8835             nan     0.1000    0.0684
##     20        0.5738             nan     0.1000    0.0373
##     40        0.2961             nan     0.1000    0.0189
##     60        0.1618             nan     0.1000    0.0059
##     80        0.0995             nan     0.1000    0.0033
##    100        0.0656             nan     0.1000    0.0019
##    120        0.0454             nan     0.1000    0.0018
##    140        0.0325             nan     0.1000    0.0008
##    150        0.0280             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1293
##      2        1.5233             nan     0.1000    0.0861
##      3        1.4652             nan     0.1000    0.0719
##      4        1.4192             nan     0.1000    0.0537
##      5        1.3839             nan     0.1000    0.0475
##      6        1.3538             nan     0.1000    0.0477
##      7        1.3230             nan     0.1000    0.0411
##      8        1.2968             nan     0.1000    0.0440
##      9        1.2678             nan     0.1000    0.0379
##     10        1.2433             nan     0.1000    0.0384
##     20        1.0582             nan     0.1000    0.0232
##     40        0.8421             nan     0.1000    0.0135
##     60        0.7009             nan     0.1000    0.0076
##     80        0.5997             nan     0.1000    0.0058
##    100        0.5207             nan     0.1000    0.0049
##    120        0.4614             nan     0.1000    0.0036
##    140        0.4114             nan     0.1000    0.0032
##    150        0.3878             nan     0.1000    0.0030
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1928
##      2        1.4846             nan     0.1000    0.1410
##      3        1.3930             nan     0.1000    0.1126
##      4        1.3212             nan     0.1000    0.0971
##      5        1.2595             nan     0.1000    0.0895
##      6        1.2023             nan     0.1000    0.0793
##      7        1.1524             nan     0.1000    0.0730
##      8        1.1063             nan     0.1000    0.0774
##      9        1.0596             nan     0.1000    0.0657
##     10        1.0200             nan     0.1000    0.0514
##     20        0.7702             nan     0.1000    0.0380
##     40        0.4810             nan     0.1000    0.0161
##     60        0.3219             nan     0.1000    0.0079
##     80        0.2250             nan     0.1000    0.0059
##    100        0.1646             nan     0.1000    0.0050
##    120        0.1204             nan     0.1000    0.0026
##    140        0.0917             nan     0.1000    0.0011
##    150        0.0813             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2546
##      2        1.4468             nan     0.1000    0.1746
##      3        1.3349             nan     0.1000    0.1493
##      4        1.2406             nan     0.1000    0.1270
##      5        1.1605             nan     0.1000    0.1112
##      6        1.0911             nan     0.1000    0.1001
##      7        1.0286             nan     0.1000    0.0832
##      8        0.9781             nan     0.1000    0.0881
##      9        0.9241             nan     0.1000    0.0623
##     10        0.8853             nan     0.1000    0.0665
##     20        0.5824             nan     0.1000    0.0341
##     40        0.3008             nan     0.1000    0.0153
##     60        0.1686             nan     0.1000    0.0059
##     80        0.1023             nan     0.1000    0.0047
##    100        0.0679             nan     0.1000    0.0022
##    120        0.0480             nan     0.1000    0.0012
##    140        0.0358             nan     0.1000    0.0006
##    150        0.0311             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1312
##      2        1.5212             nan     0.1000    0.0911
##      3        1.4608             nan     0.1000    0.0713
##      4        1.4133             nan     0.1000    0.0536
##      5        1.3771             nan     0.1000    0.0573
##      6        1.3416             nan     0.1000    0.0437
##      7        1.3136             nan     0.1000    0.0443
##      8        1.2856             nan     0.1000    0.0414
##      9        1.2581             nan     0.1000    0.0340
##     10        1.2357             nan     0.1000    0.0361
##     20        1.0517             nan     0.1000    0.0226
##     40        0.8356             nan     0.1000    0.0130
##     60        0.6983             nan     0.1000    0.0086
##     80        0.5972             nan     0.1000    0.0080
##    100        0.5186             nan     0.1000    0.0045
##    120        0.4569             nan     0.1000    0.0040
##    140        0.4047             nan     0.1000    0.0027
##    150        0.3825             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2040
##      2        1.4792             nan     0.1000    0.1475
##      3        1.3842             nan     0.1000    0.1096
##      4        1.3139             nan     0.1000    0.1008
##      5        1.2512             nan     0.1000    0.0979
##      6        1.1914             nan     0.1000    0.0822
##      7        1.1391             nan     0.1000    0.0724
##      8        1.0933             nan     0.1000    0.0644
##      9        1.0514             nan     0.1000    0.0667
##     10        1.0115             nan     0.1000    0.0494
##     20        0.7609             nan     0.1000    0.0280
##     40        0.4647             nan     0.1000    0.0153
##     60        0.3152             nan     0.1000    0.0093
##     80        0.2139             nan     0.1000    0.0041
##    100        0.1525             nan     0.1000    0.0040
##    120        0.1101             nan     0.1000    0.0021
##    140        0.0828             nan     0.1000    0.0018
##    150        0.0726             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2563
##      2        1.4444             nan     0.1000    0.1844
##      3        1.3278             nan     0.1000    0.1529
##      4        1.2320             nan     0.1000    0.1197
##      5        1.1562             nan     0.1000    0.1090
##      6        1.0875             nan     0.1000    0.0973
##      7        1.0288             nan     0.1000    0.0938
##      8        0.9716             nan     0.1000    0.0662
##      9        0.9288             nan     0.1000    0.0677
##     10        0.8855             nan     0.1000    0.0764
##     20        0.5790             nan     0.1000    0.0334
##     40        0.2922             nan     0.1000    0.0141
##     60        0.1634             nan     0.1000    0.0078
##     80        0.0980             nan     0.1000    0.0040
##    100        0.0646             nan     0.1000    0.0016
##    120        0.0446             nan     0.1000    0.0015
##    140        0.0317             nan     0.1000    0.0006
##    150        0.0274             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1358
##      2        1.5213             nan     0.1000    0.0932
##      3        1.4607             nan     0.1000    0.0689
##      4        1.4152             nan     0.1000    0.0591
##      5        1.3775             nan     0.1000    0.0459
##      6        1.3469             nan     0.1000    0.0452
##      7        1.3175             nan     0.1000    0.0443
##      8        1.2901             nan     0.1000    0.0426
##      9        1.2615             nan     0.1000    0.0362
##     10        1.2379             nan     0.1000    0.0381
##     20        1.0555             nan     0.1000    0.0266
##     40        0.8338             nan     0.1000    0.0142
##     60        0.6964             nan     0.1000    0.0058
##     80        0.5960             nan     0.1000    0.0058
##    100        0.5170             nan     0.1000    0.0049
##    120        0.4536             nan     0.1000    0.0045
##    140        0.4020             nan     0.1000    0.0036
##    150        0.3794             nan     0.1000    0.0038
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2041
##      2        1.4766             nan     0.1000    0.1475
##      3        1.3819             nan     0.1000    0.1162
##      4        1.3078             nan     0.1000    0.0945
##      5        1.2477             nan     0.1000    0.0930
##      6        1.1880             nan     0.1000    0.0737
##      7        1.1403             nan     0.1000    0.0786
##      8        1.0929             nan     0.1000    0.0674
##      9        1.0514             nan     0.1000    0.0577
##     10        1.0155             nan     0.1000    0.0560
##     20        0.7522             nan     0.1000    0.0275
##     40        0.4642             nan     0.1000    0.0131
##     60        0.3076             nan     0.1000    0.0068
##     80        0.2147             nan     0.1000    0.0058
##    100        0.1533             nan     0.1000    0.0039
##    120        0.1129             nan     0.1000    0.0031
##    140        0.0864             nan     0.1000    0.0021
##    150        0.0763             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2562
##      2        1.4467             nan     0.1000    0.1842
##      3        1.3300             nan     0.1000    0.1520
##      4        1.2348             nan     0.1000    0.1337
##      5        1.1511             nan     0.1000    0.0997
##      6        1.0874             nan     0.1000    0.0881
##      7        1.0312             nan     0.1000    0.0784
##      8        0.9822             nan     0.1000    0.0858
##      9        0.9294             nan     0.1000    0.0737
##     10        0.8844             nan     0.1000    0.0765
##     20        0.5773             nan     0.1000    0.0336
##     40        0.2853             nan     0.1000    0.0147
##     60        0.1576             nan     0.1000    0.0068
##     80        0.0967             nan     0.1000    0.0028
##    100        0.0645             nan     0.1000    0.0021
##    120        0.0445             nan     0.1000    0.0013
##    140        0.0319             nan     0.1000    0.0011
##    150        0.0271             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1303
##      2        1.5216             nan     0.1000    0.0931
##      3        1.4610             nan     0.1000    0.0694
##      4        1.4151             nan     0.1000    0.0547
##      5        1.3787             nan     0.1000    0.0520
##      6        1.3454             nan     0.1000    0.0479
##      7        1.3148             nan     0.1000    0.0440
##      8        1.2879             nan     0.1000    0.0371
##      9        1.2644             nan     0.1000    0.0409
##     10        1.2375             nan     0.1000    0.0362
##     20        1.0555             nan     0.1000    0.0211
##     40        0.8386             nan     0.1000    0.0128
##     60        0.6986             nan     0.1000    0.0087
##     80        0.5971             nan     0.1000    0.0051
##    100        0.5186             nan     0.1000    0.0036
##    120        0.4581             nan     0.1000    0.0052
##    140        0.4050             nan     0.1000    0.0038
##    150        0.3831             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1966
##      2        1.4791             nan     0.1000    0.1397
##      3        1.3867             nan     0.1000    0.1157
##      4        1.3114             nan     0.1000    0.0991
##      5        1.2487             nan     0.1000    0.0786
##      6        1.1985             nan     0.1000    0.0841
##      7        1.1456             nan     0.1000    0.0815
##      8        1.0953             nan     0.1000    0.0585
##      9        1.0592             nan     0.1000    0.0639
##     10        1.0197             nan     0.1000    0.0546
##     20        0.7506             nan     0.1000    0.0290
##     40        0.4607             nan     0.1000    0.0175
##     60        0.3097             nan     0.1000    0.0082
##     80        0.2138             nan     0.1000    0.0059
##    100        0.1564             nan     0.1000    0.0055
##    120        0.1147             nan     0.1000    0.0026
##    140        0.0871             nan     0.1000    0.0012
##    150        0.0766             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2554
##      2        1.4461             nan     0.1000    0.1901
##      3        1.3240             nan     0.1000    0.1526
##      4        1.2264             nan     0.1000    0.1289
##      5        1.1465             nan     0.1000    0.1086
##      6        1.0790             nan     0.1000    0.0927
##      7        1.0206             nan     0.1000    0.0790
##      8        0.9710             nan     0.1000    0.0881
##      9        0.9165             nan     0.1000    0.0740
##     10        0.8725             nan     0.1000    0.0574
##     20        0.5677             nan     0.1000    0.0298
##     40        0.2883             nan     0.1000    0.0136
##     60        0.1603             nan     0.1000    0.0065
##     80        0.0995             nan     0.1000    0.0030
##    100        0.0673             nan     0.1000    0.0014
##    120        0.0477             nan     0.1000    0.0013
##    140        0.0345             nan     0.1000    0.0006
##    150        0.0294             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1315
##      2        1.5215             nan     0.1000    0.0901
##      3        1.4611             nan     0.1000    0.0649
##      4        1.4173             nan     0.1000    0.0566
##      5        1.3798             nan     0.1000    0.0513
##      6        1.3463             nan     0.1000    0.0493
##      7        1.3159             nan     0.1000    0.0383
##      8        1.2902             nan     0.1000    0.0368
##      9        1.2663             nan     0.1000    0.0409
##     10        1.2390             nan     0.1000    0.0326
##     20        1.0534             nan     0.1000    0.0215
##     40        0.8387             nan     0.1000    0.0106
##     60        0.7028             nan     0.1000    0.0105
##     80        0.6019             nan     0.1000    0.0076
##    100        0.5219             nan     0.1000    0.0065
##    120        0.4593             nan     0.1000    0.0043
##    140        0.4071             nan     0.1000    0.0039
##    150        0.3858             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1966
##      2        1.4819             nan     0.1000    0.1443
##      3        1.3893             nan     0.1000    0.1106
##      4        1.3173             nan     0.1000    0.0940
##      5        1.2568             nan     0.1000    0.0905
##      6        1.1991             nan     0.1000    0.0743
##      7        1.1525             nan     0.1000    0.0738
##      8        1.1053             nan     0.1000    0.0715
##      9        1.0608             nan     0.1000    0.0479
##     10        1.0291             nan     0.1000    0.0580
##     20        0.7635             nan     0.1000    0.0342
##     40        0.4668             nan     0.1000    0.0144
##     60        0.3082             nan     0.1000    0.0125
##     80        0.2140             nan     0.1000    0.0062
##    100        0.1565             nan     0.1000    0.0046
##    120        0.1148             nan     0.1000    0.0023
##    140        0.0861             nan     0.1000    0.0013
##    150        0.0761             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2517
##      2        1.4486             nan     0.1000    0.1771
##      3        1.3352             nan     0.1000    0.1520
##      4        1.2366             nan     0.1000    0.1180
##      5        1.1597             nan     0.1000    0.1068
##      6        1.0927             nan     0.1000    0.0931
##      7        1.0333             nan     0.1000    0.0886
##      8        0.9793             nan     0.1000    0.0939
##      9        0.9222             nan     0.1000    0.0749
##     10        0.8762             nan     0.1000    0.0598
##     20        0.5680             nan     0.1000    0.0334
##     40        0.2975             nan     0.1000    0.0114
##     60        0.1621             nan     0.1000    0.0086
##     80        0.0996             nan     0.1000    0.0032
##    100        0.0661             nan     0.1000    0.0025
##    120        0.0470             nan     0.1000    0.0008
##    140        0.0340             nan     0.1000    0.0006
##    150        0.0292             nan     0.1000    0.0002
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1258
##      2        1.5230             nan     0.1000    0.0918
##      3        1.4631             nan     0.1000    0.0680
##      4        1.4174             nan     0.1000    0.0549
##      5        1.3811             nan     0.1000    0.0503
##      6        1.3483             nan     0.1000    0.0389
##      7        1.3225             nan     0.1000    0.0410
##      8        1.2966             nan     0.1000    0.0406
##      9        1.2717             nan     0.1000    0.0419
##     10        1.2445             nan     0.1000    0.0366
##     20        1.0611             nan     0.1000    0.0232
##     40        0.8458             nan     0.1000    0.0120
##     60        0.7105             nan     0.1000    0.0083
##     80        0.6105             nan     0.1000    0.0073
##    100        0.5303             nan     0.1000    0.0067
##    120        0.4653             nan     0.1000    0.0038
##    140        0.4124             nan     0.1000    0.0035
##    150        0.3887             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1949
##      2        1.4834             nan     0.1000    0.1441
##      3        1.3898             nan     0.1000    0.1183
##      4        1.3133             nan     0.1000    0.1021
##      5        1.2479             nan     0.1000    0.0804
##      6        1.1957             nan     0.1000    0.0875
##      7        1.1414             nan     0.1000    0.0795
##      8        1.0924             nan     0.1000    0.0620
##      9        1.0536             nan     0.1000    0.0624
##     10        1.0150             nan     0.1000    0.0502
##     20        0.7577             nan     0.1000    0.0331
##     40        0.4688             nan     0.1000    0.0169
##     60        0.3098             nan     0.1000    0.0072
##     80        0.2166             nan     0.1000    0.0061
##    100        0.1548             nan     0.1000    0.0039
##    120        0.1143             nan     0.1000    0.0030
##    140        0.0862             nan     0.1000    0.0021
##    150        0.0755             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2480
##      2        1.4489             nan     0.1000    0.1727
##      3        1.3387             nan     0.1000    0.1533
##      4        1.2400             nan     0.1000    0.1213
##      5        1.1636             nan     0.1000    0.1041
##      6        1.0979             nan     0.1000    0.1156
##      7        1.0257             nan     0.1000    0.0878
##      8        0.9708             nan     0.1000    0.0785
##      9        0.9230             nan     0.1000    0.0661
##     10        0.8814             nan     0.1000    0.0551
##     20        0.5878             nan     0.1000    0.0427
##     40        0.2998             nan     0.1000    0.0117
##     60        0.1683             nan     0.1000    0.0057
##     80        0.1034             nan     0.1000    0.0043
##    100        0.0674             nan     0.1000    0.0019
##    120        0.0461             nan     0.1000    0.0013
##    140        0.0332             nan     0.1000    0.0004
##    150        0.0283             nan     0.1000    0.0002
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5241             nan     0.1000    0.0880
##      3        1.4672             nan     0.1000    0.0674
##      4        1.4229             nan     0.1000    0.0540
##      5        1.3868             nan     0.1000    0.0476
##      6        1.3561             nan     0.1000    0.0453
##      7        1.3267             nan     0.1000    0.0440
##      8        1.2978             nan     0.1000    0.0439
##      9        1.2703             nan     0.1000    0.0399
##     10        1.2462             nan     0.1000    0.0363
##     20        1.0671             nan     0.1000    0.0246
##     40        0.8496             nan     0.1000    0.0133
##     60        0.7114             nan     0.1000    0.0094
##     80        0.6079             nan     0.1000    0.0066
##    100        0.5306             nan     0.1000    0.0041
##    120        0.4676             nan     0.1000    0.0039
##    140        0.4159             nan     0.1000    0.0043
##    150        0.3916             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1923
##      2        1.4841             nan     0.1000    0.1370
##      3        1.3952             nan     0.1000    0.1106
##      4        1.3237             nan     0.1000    0.1005
##      5        1.2599             nan     0.1000    0.0896
##      6        1.2032             nan     0.1000    0.0872
##      7        1.1484             nan     0.1000    0.0718
##      8        1.1043             nan     0.1000    0.0696
##      9        1.0615             nan     0.1000    0.0573
##     10        1.0256             nan     0.1000    0.0540
##     20        0.7656             nan     0.1000    0.0329
##     40        0.4776             nan     0.1000    0.0147
##     60        0.3135             nan     0.1000    0.0097
##     80        0.2192             nan     0.1000    0.0057
##    100        0.1572             nan     0.1000    0.0044
##    120        0.1156             nan     0.1000    0.0018
##    140        0.0904             nan     0.1000    0.0017
##    150        0.0794             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2545
##      2        1.4463             nan     0.1000    0.1838
##      3        1.3283             nan     0.1000    0.1503
##      4        1.2322             nan     0.1000    0.1293
##      5        1.1506             nan     0.1000    0.1105
##      6        1.0801             nan     0.1000    0.0958
##      7        1.0196             nan     0.1000    0.0808
##      8        0.9688             nan     0.1000    0.0831
##      9        0.9173             nan     0.1000    0.0728
##     10        0.8737             nan     0.1000    0.0551
##     20        0.5810             nan     0.1000    0.0377
##     40        0.3003             nan     0.1000    0.0155
##     60        0.1684             nan     0.1000    0.0087
##     80        0.1050             nan     0.1000    0.0049
##    100        0.0701             nan     0.1000    0.0022
##    120        0.0496             nan     0.1000    0.0016
##    140        0.0367             nan     0.1000    0.0006
##    150        0.0315             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1365
##      2        1.5198             nan     0.1000    0.0881
##      3        1.4604             nan     0.1000    0.0670
##      4        1.4150             nan     0.1000    0.0584
##      5        1.3758             nan     0.1000    0.0529
##      6        1.3418             nan     0.1000    0.0483
##      7        1.3111             nan     0.1000    0.0400
##      8        1.2859             nan     0.1000    0.0389
##      9        1.2595             nan     0.1000    0.0351
##     10        1.2344             nan     0.1000    0.0343
##     20        1.0470             nan     0.1000    0.0233
##     40        0.8284             nan     0.1000    0.0119
##     60        0.6919             nan     0.1000    0.0103
##     80        0.5931             nan     0.1000    0.0061
##    100        0.5172             nan     0.1000    0.0045
##    120        0.4527             nan     0.1000    0.0031
##    140        0.4016             nan     0.1000    0.0030
##    150        0.3793             nan     0.1000    0.0038
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1976
##      2        1.4786             nan     0.1000    0.1444
##      3        1.3861             nan     0.1000    0.1117
##      4        1.3141             nan     0.1000    0.1070
##      5        1.2460             nan     0.1000    0.0820
##      6        1.1928             nan     0.1000    0.0852
##      7        1.1389             nan     0.1000    0.0777
##      8        1.0914             nan     0.1000    0.0661
##      9        1.0486             nan     0.1000    0.0641
##     10        1.0087             nan     0.1000    0.0522
##     20        0.7494             nan     0.1000    0.0301
##     40        0.4694             nan     0.1000    0.0136
##     60        0.3121             nan     0.1000    0.0077
##     80        0.2140             nan     0.1000    0.0060
##    100        0.1541             nan     0.1000    0.0039
##    120        0.1139             nan     0.1000    0.0023
##    140        0.0868             nan     0.1000    0.0016
##    150        0.0764             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2591
##      2        1.4422             nan     0.1000    0.1966
##      3        1.3183             nan     0.1000    0.1429
##      4        1.2272             nan     0.1000    0.1245
##      5        1.1470             nan     0.1000    0.0987
##      6        1.0837             nan     0.1000    0.1008
##      7        1.0202             nan     0.1000    0.0893
##      8        0.9639             nan     0.1000    0.0798
##      9        0.9141             nan     0.1000    0.0803
##     10        0.8650             nan     0.1000    0.0696
##     20        0.5690             nan     0.1000    0.0396
##     40        0.2898             nan     0.1000    0.0144
##     60        0.1632             nan     0.1000    0.0080
##     80        0.1011             nan     0.1000    0.0029
##    100        0.0652             nan     0.1000    0.0013
##    120        0.0445             nan     0.1000    0.0006
##    140        0.0322             nan     0.1000    0.0005
##    150        0.0279             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1264
##      2        1.5225             nan     0.1000    0.0885
##      3        1.4628             nan     0.1000    0.0648
##      4        1.4186             nan     0.1000    0.0546
##      5        1.3821             nan     0.1000    0.0452
##      6        1.3517             nan     0.1000    0.0481
##      7        1.3219             nan     0.1000    0.0443
##      8        1.2940             nan     0.1000    0.0423
##      9        1.2661             nan     0.1000    0.0353
##     10        1.2408             nan     0.1000    0.0360
##     20        1.0593             nan     0.1000    0.0200
##     40        0.8374             nan     0.1000    0.0126
##     60        0.7006             nan     0.1000    0.0096
##     80        0.5988             nan     0.1000    0.0063
##    100        0.5168             nan     0.1000    0.0058
##    120        0.4527             nan     0.1000    0.0035
##    140        0.4027             nan     0.1000    0.0038
##    150        0.3815             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1968
##      2        1.4835             nan     0.1000    0.1417
##      3        1.3925             nan     0.1000    0.1092
##      4        1.3224             nan     0.1000    0.0958
##      5        1.2596             nan     0.1000    0.0798
##      6        1.2068             nan     0.1000    0.0857
##      7        1.1531             nan     0.1000    0.0834
##      8        1.1022             nan     0.1000    0.0732
##      9        1.0571             nan     0.1000    0.0590
##     10        1.0207             nan     0.1000    0.0492
##     20        0.7534             nan     0.1000    0.0273
##     40        0.4578             nan     0.1000    0.0155
##     60        0.3031             nan     0.1000    0.0092
##     80        0.2091             nan     0.1000    0.0037
##    100        0.1498             nan     0.1000    0.0042
##    120        0.1103             nan     0.1000    0.0021
##    140        0.0853             nan     0.1000    0.0014
##    150        0.0745             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2517
##      2        1.4479             nan     0.1000    0.1717
##      3        1.3348             nan     0.1000    0.1496
##      4        1.2401             nan     0.1000    0.1230
##      5        1.1603             nan     0.1000    0.1227
##      6        1.0846             nan     0.1000    0.0942
##      7        1.0261             nan     0.1000    0.0923
##      8        0.9682             nan     0.1000    0.0861
##      9        0.9154             nan     0.1000    0.0776
##     10        0.8686             nan     0.1000    0.0568
##     20        0.5657             nan     0.1000    0.0338
##     40        0.2877             nan     0.1000    0.0166
##     60        0.1610             nan     0.1000    0.0069
##     80        0.0997             nan     0.1000    0.0039
##    100        0.0663             nan     0.1000    0.0017
##    120        0.0454             nan     0.1000    0.0011
##    140        0.0329             nan     0.1000    0.0005
##    150        0.0286             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1288
##      2        1.5234             nan     0.1000    0.0873
##      3        1.4646             nan     0.1000    0.0694
##      4        1.4190             nan     0.1000    0.0549
##      5        1.3828             nan     0.1000    0.0507
##      6        1.3494             nan     0.1000    0.0471
##      7        1.3196             nan     0.1000    0.0437
##      8        1.2921             nan     0.1000    0.0386
##      9        1.2681             nan     0.1000    0.0369
##     10        1.2427             nan     0.1000    0.0348
##     20        1.0610             nan     0.1000    0.0211
##     40        0.8458             nan     0.1000    0.0113
##     60        0.7115             nan     0.1000    0.0085
##     80        0.6088             nan     0.1000    0.0065
##    100        0.5304             nan     0.1000    0.0044
##    120        0.4668             nan     0.1000    0.0028
##    140        0.4142             nan     0.1000    0.0035
##    150        0.3916             nan     0.1000    0.0035
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1995
##      2        1.4831             nan     0.1000    0.1409
##      3        1.3924             nan     0.1000    0.1120
##      4        1.3223             nan     0.1000    0.0987
##      5        1.2590             nan     0.1000    0.0795
##      6        1.2070             nan     0.1000    0.0860
##      7        1.1538             nan     0.1000    0.0712
##      8        1.1097             nan     0.1000    0.0721
##      9        1.0663             nan     0.1000    0.0659
##     10        1.0267             nan     0.1000    0.0548
##     20        0.7642             nan     0.1000    0.0279
##     40        0.4745             nan     0.1000    0.0177
##     60        0.3191             nan     0.1000    0.0106
##     80        0.2181             nan     0.1000    0.0041
##    100        0.1590             nan     0.1000    0.0037
##    120        0.1173             nan     0.1000    0.0028
##    140        0.0889             nan     0.1000    0.0024
##    150        0.0778             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2545
##      2        1.4466             nan     0.1000    0.1816
##      3        1.3303             nan     0.1000    0.1570
##      4        1.2315             nan     0.1000    0.1228
##      5        1.1536             nan     0.1000    0.0947
##      6        1.0915             nan     0.1000    0.0994
##      7        1.0299             nan     0.1000    0.0750
##      8        0.9830             nan     0.1000    0.0854
##      9        0.9307             nan     0.1000    0.0800
##     10        0.8818             nan     0.1000    0.0733
##     20        0.5856             nan     0.1000    0.0338
##     40        0.3037             nan     0.1000    0.0196
##     60        0.1704             nan     0.1000    0.0071
##     80        0.1032             nan     0.1000    0.0043
##    100        0.0691             nan     0.1000    0.0015
##    120        0.0485             nan     0.1000    0.0015
##    140        0.0352             nan     0.1000    0.0008
##    150        0.0303             nan     0.1000    0.0002
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5213             nan     0.1000    0.0913
##      3        1.4617             nan     0.1000    0.0689
##      4        1.4163             nan     0.1000    0.0533
##      5        1.3810             nan     0.1000    0.0471
##      6        1.3493             nan     0.1000    0.0457
##      7        1.3201             nan     0.1000    0.0409
##      8        1.2948             nan     0.1000    0.0436
##      9        1.2657             nan     0.1000    0.0361
##     10        1.2399             nan     0.1000    0.0359
##     20        1.0576             nan     0.1000    0.0243
##     40        0.8393             nan     0.1000    0.0122
##     60        0.7009             nan     0.1000    0.0088
##     80        0.5988             nan     0.1000    0.0057
##    100        0.5225             nan     0.1000    0.0047
##    120        0.4607             nan     0.1000    0.0052
##    140        0.4087             nan     0.1000    0.0032
##    150        0.3859             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1999
##      2        1.4811             nan     0.1000    0.1416
##      3        1.3903             nan     0.1000    0.1054
##      4        1.3213             nan     0.1000    0.1083
##      5        1.2529             nan     0.1000    0.0913
##      6        1.1959             nan     0.1000    0.0888
##      7        1.1413             nan     0.1000    0.0728
##      8        1.0966             nan     0.1000    0.0709
##      9        1.0522             nan     0.1000    0.0614
##     10        1.0147             nan     0.1000    0.0556
##     20        0.7611             nan     0.1000    0.0396
##     40        0.4761             nan     0.1000    0.0169
##     60        0.3158             nan     0.1000    0.0086
##     80        0.2196             nan     0.1000    0.0051
##    100        0.1580             nan     0.1000    0.0043
##    120        0.1167             nan     0.1000    0.0012
##    140        0.0895             nan     0.1000    0.0013
##    150        0.0787             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2564
##      2        1.4465             nan     0.1000    0.1905
##      3        1.3246             nan     0.1000    0.1478
##      4        1.2318             nan     0.1000    0.1261
##      5        1.1514             nan     0.1000    0.0990
##      6        1.0892             nan     0.1000    0.0966
##      7        1.0289             nan     0.1000    0.0794
##      8        0.9786             nan     0.1000    0.0914
##      9        0.9224             nan     0.1000    0.0755
##     10        0.8760             nan     0.1000    0.0621
##     20        0.5748             nan     0.1000    0.0339
##     40        0.2907             nan     0.1000    0.0155
##     60        0.1619             nan     0.1000    0.0059
##     80        0.0991             nan     0.1000    0.0042
##    100        0.0654             nan     0.1000    0.0017
##    120        0.0454             nan     0.1000    0.0011
##    140        0.0322             nan     0.1000    0.0010
##    150        0.0272             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1338
##      2        1.5193             nan     0.1000    0.0866
##      3        1.4602             nan     0.1000    0.0664
##      4        1.4153             nan     0.1000    0.0562
##      5        1.3779             nan     0.1000    0.0467
##      6        1.3472             nan     0.1000    0.0501
##      7        1.3167             nan     0.1000    0.0412
##      8        1.2911             nan     0.1000    0.0418
##      9        1.2635             nan     0.1000    0.0348
##     10        1.2413             nan     0.1000    0.0418
##     20        1.0530             nan     0.1000    0.0226
##     40        0.8327             nan     0.1000    0.0138
##     60        0.6952             nan     0.1000    0.0083
##     80        0.5938             nan     0.1000    0.0069
##    100        0.5155             nan     0.1000    0.0044
##    120        0.4531             nan     0.1000    0.0043
##    140        0.4027             nan     0.1000    0.0036
##    150        0.3834             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1997
##      2        1.4824             nan     0.1000    0.1434
##      3        1.3895             nan     0.1000    0.1173
##      4        1.3158             nan     0.1000    0.0883
##      5        1.2579             nan     0.1000    0.0934
##      6        1.1994             nan     0.1000    0.0833
##      7        1.1474             nan     0.1000    0.0742
##      8        1.1019             nan     0.1000    0.0655
##      9        1.0610             nan     0.1000    0.0618
##     10        1.0234             nan     0.1000    0.0560
##     20        0.7530             nan     0.1000    0.0290
##     40        0.4621             nan     0.1000    0.0127
##     60        0.3098             nan     0.1000    0.0093
##     80        0.2129             nan     0.1000    0.0060
##    100        0.1527             nan     0.1000    0.0039
##    120        0.1113             nan     0.1000    0.0027
##    140        0.0837             nan     0.1000    0.0025
##    150        0.0735             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2580
##      2        1.4433             nan     0.1000    0.1780
##      3        1.3292             nan     0.1000    0.1468
##      4        1.2345             nan     0.1000    0.1367
##      5        1.1490             nan     0.1000    0.1022
##      6        1.0832             nan     0.1000    0.0951
##      7        1.0234             nan     0.1000    0.0908
##      8        0.9673             nan     0.1000    0.0809
##      9        0.9163             nan     0.1000    0.0637
##     10        0.8758             nan     0.1000    0.0712
##     20        0.5750             nan     0.1000    0.0393
##     40        0.2865             nan     0.1000    0.0148
##     60        0.1588             nan     0.1000    0.0062
##     80        0.0975             nan     0.1000    0.0035
##    100        0.0645             nan     0.1000    0.0019
##    120        0.0445             nan     0.1000    0.0013
##    140        0.0321             nan     0.1000    0.0003
##    150        0.0276             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1261
##      2        1.5249             nan     0.1000    0.0876
##      3        1.4678             nan     0.1000    0.0678
##      4        1.4230             nan     0.1000    0.0552
##      5        1.3868             nan     0.1000    0.0465
##      6        1.3557             nan     0.1000    0.0495
##      7        1.3239             nan     0.1000    0.0363
##      8        1.3005             nan     0.1000    0.0414
##      9        1.2734             nan     0.1000    0.0395
##     10        1.2487             nan     0.1000    0.0344
##     20        1.0662             nan     0.1000    0.0226
##     40        0.8445             nan     0.1000    0.0112
##     60        0.7111             nan     0.1000    0.0106
##     80        0.6083             nan     0.1000    0.0075
##    100        0.5280             nan     0.1000    0.0056
##    120        0.4652             nan     0.1000    0.0047
##    140        0.4117             nan     0.1000    0.0033
##    150        0.3894             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1969
##      2        1.4828             nan     0.1000    0.1370
##      3        1.3936             nan     0.1000    0.1072
##      4        1.3241             nan     0.1000    0.0891
##      5        1.2661             nan     0.1000    0.0853
##      6        1.2117             nan     0.1000    0.0810
##      7        1.1611             nan     0.1000    0.0764
##      8        1.1128             nan     0.1000    0.0717
##      9        1.0693             nan     0.1000    0.0556
##     10        1.0333             nan     0.1000    0.0580
##     20        0.7680             nan     0.1000    0.0308
##     40        0.4810             nan     0.1000    0.0163
##     60        0.3175             nan     0.1000    0.0089
##     80        0.2232             nan     0.1000    0.0062
##    100        0.1577             nan     0.1000    0.0039
##    120        0.1167             nan     0.1000    0.0027
##    140        0.0878             nan     0.1000    0.0015
##    150        0.0775             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2491
##      2        1.4506             nan     0.1000    0.1851
##      3        1.3346             nan     0.1000    0.1549
##      4        1.2364             nan     0.1000    0.1257
##      5        1.1575             nan     0.1000    0.0976
##      6        1.0950             nan     0.1000    0.0986
##      7        1.0334             nan     0.1000    0.0869
##      8        0.9802             nan     0.1000    0.0646
##      9        0.9384             nan     0.1000    0.0820
##     10        0.8898             nan     0.1000    0.0620
##     20        0.5840             nan     0.1000    0.0378
##     40        0.2920             nan     0.1000    0.0165
##     60        0.1667             nan     0.1000    0.0069
##     80        0.1009             nan     0.1000    0.0046
##    100        0.0666             nan     0.1000    0.0017
##    120        0.0473             nan     0.1000    0.0015
##    140        0.0344             nan     0.1000    0.0004
##    150        0.0297             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1284
##      2        1.5250             nan     0.1000    0.0822
##      3        1.4687             nan     0.1000    0.0683
##      4        1.4243             nan     0.1000    0.0543
##      5        1.3891             nan     0.1000    0.0508
##      6        1.3568             nan     0.1000    0.0404
##      7        1.3309             nan     0.1000    0.0435
##      8        1.3044             nan     0.1000    0.0429
##      9        1.2755             nan     0.1000    0.0356
##     10        1.2530             nan     0.1000    0.0398
##     20        1.0642             nan     0.1000    0.0266
##     40        0.8407             nan     0.1000    0.0106
##     60        0.7019             nan     0.1000    0.0072
##     80        0.6011             nan     0.1000    0.0069
##    100        0.5198             nan     0.1000    0.0039
##    120        0.4584             nan     0.1000    0.0039
##    140        0.4077             nan     0.1000    0.0031
##    150        0.3849             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1989
##      2        1.4824             nan     0.1000    0.1369
##      3        1.3950             nan     0.1000    0.1190
##      4        1.3186             nan     0.1000    0.1046
##      5        1.2527             nan     0.1000    0.0873
##      6        1.1964             nan     0.1000    0.0714
##      7        1.1509             nan     0.1000    0.0682
##      8        1.1078             nan     0.1000    0.0681
##      9        1.0662             nan     0.1000    0.0533
##     10        1.0321             nan     0.1000    0.0617
##     20        0.7617             nan     0.1000    0.0336
##     40        0.4686             nan     0.1000    0.0181
##     60        0.3110             nan     0.1000    0.0107
##     80        0.2171             nan     0.1000    0.0050
##    100        0.1564             nan     0.1000    0.0027
##    120        0.1159             nan     0.1000    0.0023
##    140        0.0881             nan     0.1000    0.0017
##    150        0.0766             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2510
##      2        1.4501             nan     0.1000    0.1923
##      3        1.3279             nan     0.1000    0.1432
##      4        1.2366             nan     0.1000    0.1189
##      5        1.1606             nan     0.1000    0.1111
##      6        1.0901             nan     0.1000    0.0863
##      7        1.0345             nan     0.1000    0.0974
##      8        0.9752             nan     0.1000    0.0737
##      9        0.9288             nan     0.1000    0.0761
##     10        0.8818             nan     0.1000    0.0625
##     20        0.5769             nan     0.1000    0.0376
##     40        0.2976             nan     0.1000    0.0148
##     60        0.1604             nan     0.1000    0.0067
##     80        0.1007             nan     0.1000    0.0039
##    100        0.0662             nan     0.1000    0.0018
##    120        0.0463             nan     0.1000    0.0013
##    140        0.0340             nan     0.1000    0.0006
##    150        0.0295             nan     0.1000    0.0003
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1316
##      2        1.5214             nan     0.1000    0.0895
##      3        1.4629             nan     0.1000    0.0702
##      4        1.4167             nan     0.1000    0.0536
##      5        1.3814             nan     0.1000    0.0539
##      6        1.3472             nan     0.1000    0.0451
##      7        1.3182             nan     0.1000    0.0377
##      8        1.2938             nan     0.1000    0.0409
##      9        1.2682             nan     0.1000    0.0363
##     10        1.2431             nan     0.1000    0.0378
##     20        1.0557             nan     0.1000    0.0218
##     40        0.8409             nan     0.1000    0.0113
##     60        0.7034             nan     0.1000    0.0085
##     80        0.6039             nan     0.1000    0.0058
##    100        0.5209             nan     0.1000    0.0045
##    120        0.4613             nan     0.1000    0.0054
##    140        0.4064             nan     0.1000    0.0028
##    150        0.3854             nan     0.1000    0.0036
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1986
##      2        1.4815             nan     0.1000    0.1419
##      3        1.3899             nan     0.1000    0.1057
##      4        1.3208             nan     0.1000    0.0926
##      5        1.2616             nan     0.1000    0.0895
##      6        1.2056             nan     0.1000    0.0946
##      7        1.1468             nan     0.1000    0.0777
##      8        1.0989             nan     0.1000    0.0659
##      9        1.0568             nan     0.1000    0.0665
##     10        1.0163             nan     0.1000    0.0525
##     20        0.7613             nan     0.1000    0.0314
##     40        0.4748             nan     0.1000    0.0166
##     60        0.3210             nan     0.1000    0.0106
##     80        0.2223             nan     0.1000    0.0086
##    100        0.1579             nan     0.1000    0.0038
##    120        0.1166             nan     0.1000    0.0033
##    140        0.0879             nan     0.1000    0.0017
##    150        0.0759             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2535
##      2        1.4453             nan     0.1000    0.1820
##      3        1.3284             nan     0.1000    0.1464
##      4        1.2362             nan     0.1000    0.1248
##      5        1.1586             nan     0.1000    0.1046
##      6        1.0934             nan     0.1000    0.1025
##      7        1.0296             nan     0.1000    0.0895
##      8        0.9740             nan     0.1000    0.0755
##      9        0.9265             nan     0.1000    0.0760
##     10        0.8794             nan     0.1000    0.0718
##     20        0.5748             nan     0.1000    0.0325
##     40        0.3001             nan     0.1000    0.0148
##     60        0.1670             nan     0.1000    0.0068
##     80        0.1010             nan     0.1000    0.0036
##    100        0.0673             nan     0.1000    0.0026
##    120        0.0466             nan     0.1000    0.0013
##    140        0.0339             nan     0.1000    0.0003
##    150        0.0295             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1331
##      2        1.5212             nan     0.1000    0.0919
##      3        1.4615             nan     0.1000    0.0691
##      4        1.4162             nan     0.1000    0.0584
##      5        1.3788             nan     0.1000    0.0524
##      6        1.3456             nan     0.1000    0.0479
##      7        1.3154             nan     0.1000    0.0432
##      8        1.2884             nan     0.1000    0.0407
##      9        1.2610             nan     0.1000    0.0373
##     10        1.2359             nan     0.1000    0.0340
##     20        1.0548             nan     0.1000    0.0236
##     40        0.8345             nan     0.1000    0.0117
##     60        0.6982             nan     0.1000    0.0089
##     80        0.5946             nan     0.1000    0.0060
##    100        0.5156             nan     0.1000    0.0055
##    120        0.4571             nan     0.1000    0.0032
##    140        0.4050             nan     0.1000    0.0034
##    150        0.3826             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1974
##      2        1.4820             nan     0.1000    0.1457
##      3        1.3869             nan     0.1000    0.1120
##      4        1.3141             nan     0.1000    0.0926
##      5        1.2531             nan     0.1000    0.0842
##      6        1.1984             nan     0.1000    0.0871
##      7        1.1443             nan     0.1000    0.0803
##      8        1.0952             nan     0.1000    0.0672
##      9        1.0536             nan     0.1000    0.0593
##     10        1.0173             nan     0.1000    0.0593
##     20        0.7545             nan     0.1000    0.0334
##     40        0.4660             nan     0.1000    0.0192
##     60        0.3090             nan     0.1000    0.0096
##     80        0.2151             nan     0.1000    0.0058
##    100        0.1562             nan     0.1000    0.0030
##    120        0.1164             nan     0.1000    0.0021
##    140        0.0883             nan     0.1000    0.0019
##    150        0.0776             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2526
##      2        1.4488             nan     0.1000    0.1925
##      3        1.3259             nan     0.1000    0.1373
##      4        1.2376             nan     0.1000    0.1215
##      5        1.1608             nan     0.1000    0.1082
##      6        1.0930             nan     0.1000    0.1094
##      7        1.0257             nan     0.1000    0.0784
##      8        0.9761             nan     0.1000    0.0808
##      9        0.9265             nan     0.1000    0.0730
##     10        0.8813             nan     0.1000    0.0651
##     20        0.5646             nan     0.1000    0.0293
##     40        0.2922             nan     0.1000    0.0156
##     60        0.1631             nan     0.1000    0.0075
##     80        0.1008             nan     0.1000    0.0035
##    100        0.0673             nan     0.1000    0.0009
##    120        0.0469             nan     0.1000    0.0011
##    140        0.0342             nan     0.1000    0.0008
##    150        0.0295             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1277
##      2        1.5230             nan     0.1000    0.0900
##      3        1.4637             nan     0.1000    0.0662
##      4        1.4193             nan     0.1000    0.0523
##      5        1.3844             nan     0.1000    0.0531
##      6        1.3503             nan     0.1000    0.0424
##      7        1.3222             nan     0.1000    0.0461
##      8        1.2930             nan     0.1000    0.0456
##      9        1.2633             nan     0.1000    0.0412
##     10        1.2357             nan     0.1000    0.0372
##     20        1.0530             nan     0.1000    0.0220
##     40        0.8327             nan     0.1000    0.0116
##     60        0.6949             nan     0.1000    0.0087
##     80        0.5932             nan     0.1000    0.0053
##    100        0.5148             nan     0.1000    0.0064
##    120        0.4531             nan     0.1000    0.0063
##    140        0.4028             nan     0.1000    0.0037
##    150        0.3809             nan     0.1000    0.0037
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1930
##      2        1.4830             nan     0.1000    0.1438
##      3        1.3895             nan     0.1000    0.1232
##      4        1.3115             nan     0.1000    0.1053
##      5        1.2427             nan     0.1000    0.0932
##      6        1.1834             nan     0.1000    0.0681
##      7        1.1384             nan     0.1000    0.0749
##      8        1.0927             nan     0.1000    0.0622
##      9        1.0536             nan     0.1000    0.0685
##     10        1.0120             nan     0.1000    0.0547
##     20        0.7465             nan     0.1000    0.0295
##     40        0.4643             nan     0.1000    0.0176
##     60        0.3117             nan     0.1000    0.0088
##     80        0.2171             nan     0.1000    0.0062
##    100        0.1549             nan     0.1000    0.0033
##    120        0.1160             nan     0.1000    0.0025
##    140        0.0876             nan     0.1000    0.0014
##    150        0.0782             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2495
##      2        1.4472             nan     0.1000    0.1903
##      3        1.3265             nan     0.1000    0.1379
##      4        1.2378             nan     0.1000    0.1318
##      5        1.1531             nan     0.1000    0.1172
##      6        1.0799             nan     0.1000    0.1012
##      7        1.0180             nan     0.1000    0.0822
##      8        0.9661             nan     0.1000    0.0832
##      9        0.9154             nan     0.1000    0.0706
##     10        0.8719             nan     0.1000    0.0558
##     20        0.5754             nan     0.1000    0.0295
##     40        0.2909             nan     0.1000    0.0117
##     60        0.1686             nan     0.1000    0.0042
##     80        0.1036             nan     0.1000    0.0036
##    100        0.0697             nan     0.1000    0.0013
##    120        0.0485             nan     0.1000    0.0010
##    140        0.0353             nan     0.1000    0.0006
##    150        0.0304             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1314
##      2        1.5193             nan     0.1000    0.0887
##      3        1.4610             nan     0.1000    0.0695
##      4        1.4150             nan     0.1000    0.0549
##      5        1.3780             nan     0.1000    0.0540
##      6        1.3436             nan     0.1000    0.0475
##      7        1.3128             nan     0.1000    0.0474
##      8        1.2838             nan     0.1000    0.0385
##      9        1.2573             nan     0.1000    0.0401
##     10        1.2307             nan     0.1000    0.0373
##     20        1.0467             nan     0.1000    0.0235
##     40        0.8250             nan     0.1000    0.0110
##     60        0.6883             nan     0.1000    0.0078
##     80        0.5889             nan     0.1000    0.0049
##    100        0.5090             nan     0.1000    0.0053
##    120        0.4487             nan     0.1000    0.0044
##    140        0.3989             nan     0.1000    0.0034
##    150        0.3766             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1996
##      2        1.4786             nan     0.1000    0.1421
##      3        1.3868             nan     0.1000    0.1155
##      4        1.3142             nan     0.1000    0.0988
##      5        1.2506             nan     0.1000    0.0910
##      6        1.1925             nan     0.1000    0.0854
##      7        1.1398             nan     0.1000    0.0898
##      8        1.0856             nan     0.1000    0.0641
##      9        1.0455             nan     0.1000    0.0651
##     10        1.0056             nan     0.1000    0.0533
##     20        0.7506             nan     0.1000    0.0279
##     40        0.4646             nan     0.1000    0.0204
##     60        0.3025             nan     0.1000    0.0116
##     80        0.2111             nan     0.1000    0.0056
##    100        0.1515             nan     0.1000    0.0043
##    120        0.1123             nan     0.1000    0.0032
##    140        0.0863             nan     0.1000    0.0016
##    150        0.0749             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2558
##      2        1.4429             nan     0.1000    0.1889
##      3        1.3231             nan     0.1000    0.1549
##      4        1.2248             nan     0.1000    0.1270
##      5        1.1447             nan     0.1000    0.1045
##      6        1.0792             nan     0.1000    0.1014
##      7        1.0169             nan     0.1000    0.0932
##      8        0.9612             nan     0.1000    0.0764
##      9        0.9132             nan     0.1000    0.0670
##     10        0.8711             nan     0.1000    0.0728
##     20        0.5599             nan     0.1000    0.0307
##     40        0.2844             nan     0.1000    0.0103
##     60        0.1616             nan     0.1000    0.0077
##     80        0.0988             nan     0.1000    0.0035
##    100        0.0641             nan     0.1000    0.0026
##    120        0.0447             nan     0.1000    0.0010
##    140        0.0330             nan     0.1000    0.0007
##    150        0.0285             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1300
##      2        1.5228             nan     0.1000    0.0881
##      3        1.4654             nan     0.1000    0.0684
##      4        1.4207             nan     0.1000    0.0570
##      5        1.3836             nan     0.1000    0.0501
##      6        1.3513             nan     0.1000    0.0479
##      7        1.3219             nan     0.1000    0.0396
##      8        1.2966             nan     0.1000    0.0411
##      9        1.2710             nan     0.1000    0.0401
##     10        1.2446             nan     0.1000    0.0337
##     20        1.0636             nan     0.1000    0.0236
##     40        0.8458             nan     0.1000    0.0122
##     60        0.7089             nan     0.1000    0.0091
##     80        0.6068             nan     0.1000    0.0055
##    100        0.5288             nan     0.1000    0.0049
##    120        0.4662             nan     0.1000    0.0035
##    140        0.4128             nan     0.1000    0.0039
##    150        0.3902             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1952
##      2        1.4853             nan     0.1000    0.1375
##      3        1.3975             nan     0.1000    0.1101
##      4        1.3254             nan     0.1000    0.0903
##      5        1.2673             nan     0.1000    0.0899
##      6        1.2081             nan     0.1000    0.0799
##      7        1.1586             nan     0.1000    0.0739
##      8        1.1126             nan     0.1000    0.0745
##      9        1.0677             nan     0.1000    0.0610
##     10        1.0302             nan     0.1000    0.0564
##     20        0.7653             nan     0.1000    0.0290
##     40        0.4724             nan     0.1000    0.0177
##     60        0.3169             nan     0.1000    0.0109
##     80        0.2164             nan     0.1000    0.0050
##    100        0.1567             nan     0.1000    0.0049
##    120        0.1150             nan     0.1000    0.0034
##    140        0.0870             nan     0.1000    0.0018
##    150        0.0765             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2460
##      2        1.4494             nan     0.1000    0.1804
##      3        1.3369             nan     0.1000    0.1587
##      4        1.2360             nan     0.1000    0.1218
##      5        1.1602             nan     0.1000    0.1039
##      6        1.0926             nan     0.1000    0.0956
##      7        1.0329             nan     0.1000    0.0883
##      8        0.9790             nan     0.1000    0.0795
##      9        0.9302             nan     0.1000    0.0831
##     10        0.8794             nan     0.1000    0.0621
##     20        0.5791             nan     0.1000    0.0403
##     40        0.2967             nan     0.1000    0.0168
##     60        0.1638             nan     0.1000    0.0077
##     80        0.1026             nan     0.1000    0.0040
##    100        0.0670             nan     0.1000    0.0022
##    120        0.0464             nan     0.1000    0.0011
##    140        0.0337             nan     0.1000    0.0008
##    150        0.0289             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1262
##      2        1.5248             nan     0.1000    0.0850
##      3        1.4673             nan     0.1000    0.0678
##      4        1.4225             nan     0.1000    0.0518
##      5        1.3881             nan     0.1000    0.0497
##      6        1.3563             nan     0.1000    0.0504
##      7        1.3253             nan     0.1000    0.0371
##      8        1.3011             nan     0.1000    0.0436
##      9        1.2734             nan     0.1000    0.0388
##     10        1.2472             nan     0.1000    0.0342
##     20        1.0678             nan     0.1000    0.0224
##     40        0.8484             nan     0.1000    0.0113
##     60        0.7112             nan     0.1000    0.0084
##     80        0.6084             nan     0.1000    0.0059
##    100        0.5285             nan     0.1000    0.0037
##    120        0.4669             nan     0.1000    0.0045
##    140        0.4165             nan     0.1000    0.0033
##    150        0.3946             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1907
##      2        1.4838             nan     0.1000    0.1433
##      3        1.3921             nan     0.1000    0.1109
##      4        1.3190             nan     0.1000    0.0907
##      5        1.2602             nan     0.1000    0.0948
##      6        1.2002             nan     0.1000    0.0776
##      7        1.1518             nan     0.1000    0.0758
##      8        1.1050             nan     0.1000    0.0794
##      9        1.0574             nan     0.1000    0.0539
##     10        1.0235             nan     0.1000    0.0490
##     20        0.7654             nan     0.1000    0.0331
##     40        0.4781             nan     0.1000    0.0178
##     60        0.3180             nan     0.1000    0.0073
##     80        0.2269             nan     0.1000    0.0064
##    100        0.1627             nan     0.1000    0.0031
##    120        0.1207             nan     0.1000    0.0032
##    140        0.0910             nan     0.1000    0.0019
##    150        0.0804             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2556
##      2        1.4467             nan     0.1000    0.1753
##      3        1.3360             nan     0.1000    0.1515
##      4        1.2404             nan     0.1000    0.1230
##      5        1.1632             nan     0.1000    0.0998
##      6        1.0986             nan     0.1000    0.1035
##      7        1.0336             nan     0.1000    0.0887
##      8        0.9786             nan     0.1000    0.0845
##      9        0.9281             nan     0.1000    0.0753
##     10        0.8820             nan     0.1000    0.0734
##     20        0.5737             nan     0.1000    0.0415
##     40        0.2963             nan     0.1000    0.0169
##     60        0.1684             nan     0.1000    0.0075
##     80        0.1034             nan     0.1000    0.0041
##    100        0.0685             nan     0.1000    0.0014
##    120        0.0492             nan     0.1000    0.0012
##    140        0.0358             nan     0.1000    0.0004
##    150        0.0310             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1286
##      2        1.5227             nan     0.1000    0.0877
##      3        1.4645             nan     0.1000    0.0661
##      4        1.4204             nan     0.1000    0.0557
##      5        1.3826             nan     0.1000    0.0487
##      6        1.3509             nan     0.1000    0.0459
##      7        1.3209             nan     0.1000    0.0460
##      8        1.2924             nan     0.1000    0.0367
##      9        1.2665             nan     0.1000    0.0327
##     10        1.2453             nan     0.1000    0.0409
##     20        1.0563             nan     0.1000    0.0215
##     40        0.8387             nan     0.1000    0.0112
##     60        0.7030             nan     0.1000    0.0090
##     80        0.5983             nan     0.1000    0.0067
##    100        0.5182             nan     0.1000    0.0059
##    120        0.4566             nan     0.1000    0.0049
##    140        0.4056             nan     0.1000    0.0031
##    150        0.3841             nan     0.1000    0.0031
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1906
##      2        1.4841             nan     0.1000    0.1421
##      3        1.3921             nan     0.1000    0.1117
##      4        1.3194             nan     0.1000    0.0884
##      5        1.2623             nan     0.1000    0.1086
##      6        1.1958             nan     0.1000    0.0710
##      7        1.1500             nan     0.1000    0.0852
##      8        1.0978             nan     0.1000    0.0704
##      9        1.0548             nan     0.1000    0.0599
##     10        1.0184             nan     0.1000    0.0516
##     20        0.7493             nan     0.1000    0.0292
##     40        0.4669             nan     0.1000    0.0143
##     60        0.3130             nan     0.1000    0.0099
##     80        0.2182             nan     0.1000    0.0073
##    100        0.1576             nan     0.1000    0.0049
##    120        0.1162             nan     0.1000    0.0025
##    140        0.0874             nan     0.1000    0.0017
##    150        0.0767             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2510
##      2        1.4485             nan     0.1000    0.1858
##      3        1.3319             nan     0.1000    0.1417
##      4        1.2421             nan     0.1000    0.1468
##      5        1.1525             nan     0.1000    0.1035
##      6        1.0869             nan     0.1000    0.1061
##      7        1.0191             nan     0.1000    0.0816
##      8        0.9681             nan     0.1000    0.0823
##      9        0.9183             nan     0.1000    0.0822
##     10        0.8676             nan     0.1000    0.0607
##     20        0.5647             nan     0.1000    0.0327
##     40        0.2920             nan     0.1000    0.0133
##     60        0.1656             nan     0.1000    0.0090
##     80        0.1011             nan     0.1000    0.0048
##    100        0.0674             nan     0.1000    0.0021
##    120        0.0476             nan     0.1000    0.0008
##    140        0.0345             nan     0.1000    0.0005
##    150        0.0300             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2405
##      2        1.4526             nan     0.1000    0.1878
##      3        1.3333             nan     0.1000    0.1511
##      4        1.2358             nan     0.1000    0.1274
##      5        1.1558             nan     0.1000    0.1081
##      6        1.0876             nan     0.1000    0.0915
##      7        1.0304             nan     0.1000    0.0777
##      8        0.9812             nan     0.1000    0.0857
##      9        0.9280             nan     0.1000    0.0681
##     10        0.8865             nan     0.1000    0.0663
##     20        0.5796             nan     0.1000    0.0386
##     40        0.2913             nan     0.1000    0.0153
##     60        0.1613             nan     0.1000    0.0064
##     80        0.0995             nan     0.1000    0.0031
##    100        0.0656             nan     0.1000    0.0015
##    120        0.0458             nan     0.1000    0.0012
##    140        0.0331             nan     0.1000    0.0002
##    150        0.0290             nan     0.1000    0.0006
```

![](Machine-Learning-Predicting-Exercise_files/figure-html/model fit-1.png)<!-- -->

```
##                                                           var      rel.inf
## raw_timestamp_part_1                     raw_timestamp_part_1 2.945201e+01
## roll_belt                                           roll_belt 1.295682e+01
## num_window                                         num_window 7.401351e+00
## pitch_forearm                                   pitch_forearm 6.445592e+00
## magnet_dumbbell_z                           magnet_dumbbell_z 3.990439e+00
## roll_forearm                                     roll_forearm 3.186387e+00
## pitch_belt                                         pitch_belt 3.173748e+00
## cvtd_timestamp30/11/2011 17:12 cvtd_timestamp30/11/2011 17:12 3.086897e+00
## cvtd_timestamp28/11/2011 14:15 cvtd_timestamp28/11/2011 14:15 2.829702e+00
## magnet_dumbbell_y                           magnet_dumbbell_y 2.462349e+00
## cvtd_timestamp02/12/2011 13:33 cvtd_timestamp02/12/2011 13:33 1.981993e+00
## roll_dumbbell                                   roll_dumbbell 1.872875e+00
## cvtd_timestamp02/12/2011 13:34 cvtd_timestamp02/12/2011 13:34 1.499319e+00
## yaw_belt                                             yaw_belt 1.428216e+00
## magnet_belt_z                                   magnet_belt_z 1.366140e+00
## accel_forearm_x                               accel_forearm_x 1.318281e+00
## gyros_belt_z                                     gyros_belt_z 1.277372e+00
## accel_dumbbell_y                             accel_dumbbell_y 1.276314e+00
## cvtd_timestamp02/12/2011 14:58 cvtd_timestamp02/12/2011 14:58 1.177572e+00
## cvtd_timestamp30/11/2011 17:11 cvtd_timestamp30/11/2011 17:11 1.147433e+00
## cvtd_timestamp02/12/2011 14:57 cvtd_timestamp02/12/2011 14:57 1.129539e+00
## gyros_dumbbell_y                             gyros_dumbbell_y 1.066843e+00
## cvtd_timestamp05/12/2011 11:24 cvtd_timestamp05/12/2011 11:24 9.534109e-01
## accel_dumbbell_x                             accel_dumbbell_x 7.705248e-01
## accel_dumbbell_z                             accel_dumbbell_z 6.157362e-01
## cvtd_timestamp05/12/2011 14:23 cvtd_timestamp05/12/2011 14:23 5.395772e-01
## cvtd_timestamp02/12/2011 13:35 cvtd_timestamp02/12/2011 13:35 4.654885e-01
## roll_arm                                             roll_arm 4.324860e-01
## yaw_arm                                               yaw_arm 4.237024e-01
## cvtd_timestamp05/12/2011 14:24 cvtd_timestamp05/12/2011 14:24 4.176247e-01
## cvtd_timestamp05/12/2011 14:22 cvtd_timestamp05/12/2011 14:22 4.063869e-01
## magnet_belt_y                                   magnet_belt_y 4.015627e-01
## total_accel_forearm                       total_accel_forearm 3.471324e-01
## magnet_dumbbell_x                           magnet_dumbbell_x 2.711895e-01
## magnet_arm_z                                     magnet_arm_z 2.569371e-01
## magnet_forearm_x                             magnet_forearm_x 2.540118e-01
## gyros_belt_y                                     gyros_belt_y 2.434105e-01
## cvtd_timestamp05/12/2011 11:25 cvtd_timestamp05/12/2011 11:25 2.432527e-01
## cvtd_timestamp02/12/2011 14:59 cvtd_timestamp02/12/2011 14:59 2.363126e-01
## accel_belt_z                                     accel_belt_z 1.940983e-01
## accel_arm_x                                       accel_arm_x 1.813097e-01
## magnet_forearm_z                             magnet_forearm_z 1.445793e-01
## magnet_belt_x                                   magnet_belt_x 1.168869e-01
## gyros_arm_y                                       gyros_arm_y 8.641914e-02
## yaw_dumbbell                                     yaw_dumbbell 6.216074e-02
## accel_arm_z                                       accel_arm_z 5.825676e-02
## user_namecharles                             user_namecharles 5.339411e-02
## accel_forearm_z                               accel_forearm_z 5.156797e-02
## gyros_arm_x                                       gyros_arm_x 2.364896e-02
## gyros_forearm_y                               gyros_forearm_y 2.119366e-02
## yaw_forearm                                       yaw_forearm 2.080279e-02
## total_accel_belt                             total_accel_belt 1.986797e-02
## pitch_arm                                           pitch_arm 1.841387e-02
## cvtd_timestamp30/11/2011 17:10 cvtd_timestamp30/11/2011 17:10 1.619130e-02
## magnet_arm_y                                     magnet_arm_y 1.507678e-02
## cvtd_timestamp28/11/2011 14:14 cvtd_timestamp28/11/2011 14:14 1.504073e-02
## magnet_arm_x                                     magnet_arm_x 1.468009e-02
## pitch_dumbbell                                 pitch_dumbbell 1.357565e-02
## gyros_dumbbell_x                             gyros_dumbbell_x 1.115535e-02
## raw_timestamp_part_2                     raw_timestamp_part_2 8.946560e-03
## accel_belt_y                                     accel_belt_y 7.684338e-03
## gyros_belt_x                                     gyros_belt_x 7.217846e-03
## total_accel_dumbbell                     total_accel_dumbbell 7.193564e-03
## total_accel_arm                               total_accel_arm 6.859187e-03
## accel_forearm_y                               accel_forearm_y 6.496834e-03
## accel_belt_x                                     accel_belt_x 4.267331e-03
## accel_arm_y                                       accel_arm_y 3.776837e-03
## gyros_arm_z                                       gyros_arm_z 2.310553e-03
## gyros_dumbbell_z                             gyros_dumbbell_z 5.008206e-04
## gyros_forearm_z                               gyros_forearm_z 4.899702e-04
## user_namecarlitos                           user_namecarlitos 0.000000e+00
## user_nameeurico                               user_nameeurico 0.000000e+00
## user_namejeremy                               user_namejeremy 0.000000e+00
## user_namepedro                                 user_namepedro 0.000000e+00
## cvtd_timestamp02/12/2011 14:56 cvtd_timestamp02/12/2011 14:56 0.000000e+00
## cvtd_timestamp05/12/2011 11:23 cvtd_timestamp05/12/2011 11:23 0.000000e+00
## cvtd_timestamp28/11/2011 14:13 cvtd_timestamp28/11/2011 14:13 0.000000e+00
## gyros_forearm_x                               gyros_forearm_x 0.000000e+00
## magnet_forearm_y                             magnet_forearm_y 0.000000e+00
```

Reviewing the summary of the model, we realize that 9 of the predictors have zero influence, which would allow us to simplify the model by eliminating them without any loss of information or predictive ability.

## Prediction and Cross Validation

Using the fitted model I generate a prediction using the `trainTest` data for Cross Validation purposes.


```r
predGBM<-predict(modelGBM,newdata = trainTest)
ConfMatrix<-confusionMatrix(predGBM,trainTest$classe)
print(ConfMatrix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  945    2    0    0
##          C    0    2  849    3    0
##          D    0    2    4  800    5
##          E    0    0    0    1  896
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9961         
##                  95% CI : (0.994, 0.9977)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9951         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9958   0.9930   0.9950   0.9945
## Specificity            1.0000   0.9995   0.9988   0.9973   0.9998
## Pos Pred Value         1.0000   0.9979   0.9941   0.9864   0.9989
## Neg Pred Value         1.0000   0.9990   0.9985   0.9990   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1927   0.1731   0.1631   0.1827
## Detection Prevalence   0.2845   0.1931   0.1741   0.1654   0.1829
## Balanced Accuracy      1.0000   0.9976   0.9959   0.9962   0.9971
```

```r
#plotdata<-data.frame(predGBM,trainTest$classe)
#plot1<-ggplot(plotdata, aes(y=predGBM,x=trainTest$classe))+
#    stat_sum(alpha=0.8)+scale_size(range=c(0,20))
#finalplot<-plot1+geom_text(data = ggplot_build(plot1)$data[[1]], 
#              aes(x, y, label = n), color = "white")
#print(finalplot)
```

The Confusion Matrix including the actual values of `classe` in `trainTest` vs. the predicted values of our fitted model using GBM has an accuracy of 99.6%. 

The plot also shows the consistent performance of the fitted model. 

