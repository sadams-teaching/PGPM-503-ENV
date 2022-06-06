import pandas as pd
import numpy as np

# Generate a dataset with the following: 

## ID (1000 patients)
## Age (18-96)
## Average resting heart rate (60-100)
### Heart rate and age correlated with r of 0.4
## Height (48 - 84 inches)
## Weight (98 - 250 pounds)
### Height and weight correlated with r 0.64
## Average Systolic Blood Pressure (90-140) with a mean of 120 and a standard deviation of 10 and 5% missing data
## Serum Creatinine (0.5-1.5) with a mean of 0.7 and a standard deviation of 0.1,
### Creatinine and systolic blood pressure correlated with r of 0.3



IDS = np.arange(start = 1, stop = 1001, step = 1)

height_range = np.array([48, 84])
weight_range = np.array([98, 250])
means = [height_range.mean(), weight_range.mean()]  
stds = [height_range.std() / 3, weight_range.std() / 3]
corr = 0.64        # correlation
covs = [[stds[0]**2, stds[0]*stds[1]*corr], 
        [stds[0]*stds[1]*corr, stds[1]**2]] 

m = np.random.multivariate_normal(means, covs, 1000).T

HEIGHT = m[0].round().astype(int)
WEIGHT = m[1].round().astype(int)

hr_range = np.array([60, 100])
age_range = np.array([18, 96])
means = [hr_range.mean(), age_range.mean()]
stds = [hr_range.std() / 3, age_range.std() / 3]
corr = 0.4
covs = [[stds[0]**2, stds[0]*stds[1]*corr],
        [stds[0]*stds[1]*corr, stds[1]**2]]

m = np.random.multivariate_normal(means, covs, 1000).T

HR = m[0].round().astype(int)
AGE = m[1].round().astype(int)

creatinine_range = np.array([0.5, 1.5])
sbp_range = np.array([90, 140])
means = [creatinine_range.mean(), sbp_range.mean()]
stds = [creatinine_range.std() / 3, sbp_range.std() / 3]
corr = 0.3
covs = [[stds[0]**2, stds[0]*stds[1]*corr],
        [stds[0]*stds[1]*corr, stds[1]**2]]

m = np.random.multivariate_normal(means, covs, 1000).T

CREATININE = m[0].round(2)
SYS_BLOOD_PRESSURE = m[1].round(0)

SYS_BLOOD_PRESSURE[np.random.randint(0, 1000, size = 55)] = np.nan

pd.DataFrame(
    {
        'ID': IDS, 
        'Age': AGE, 
        'Height': HEIGHT, 
        'Weight': WEIGHT, 
        'Heart Rate': HR, 
        'Systolic Blood Pressure': SYS_BLOOD_PRESSURE,
        'Serum Creatinine': CREATININE})\
    .to_csv('../data/size_data.csv', index = False)




