"""Results using grey-matter maps."""

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import KFold

from utils import plot_results, GridSearchCVRBFKernel


# Load the data
df = pd.read_csv('gm/Kernels/euclidean_norm.csv', index_col=0)
X = df.values
y = pd.read_csv('gm/Target/age.csv', index_col=0, header=None).squeeze().values


# Perform 5-fold cross-validation
estimator = SVR(kernel='precomputed', max_iter=1e6, tol=1e-2)
param_grid = {'gamma': [j * 10**k for k in range(-8, -6) for j in [1, 4, 7]],
              'C': [j * 10**k for k in range(2, 4) for j in [1, 4, 7]]}


y_trues, y_preds = [], []
gridsearchs = []

for i in range(5):
    df_train = pd.read_csv(
        '.../data/splits/training/split-{0}.tsv'.format(i),
        sep='\t', index_col=0
    )
    df_val = pd.read_csv(
        '.../data/splits/validation/split-{0}.tsv'.format(i),
        sep='\t', index_col=0
    )
    train_index = np.where(
        np.in1d(df.index.values, df_train['subject_ID'].values))[0]
    val_index = np.where(
        np.in1d(df.index.values, df_val['subject_ID'].values))[0]

    gridsearch = GridSearchCVRBFKernel(
        estimator, param_grid, cv=KFold(5, shuffle=True, random_state=42),
        age_range=(18, 90)
    )

    gridsearch.fit(X, y, train_index)
    y_pred = gridsearch.predict(val_index)
    y_preds.append(y_pred)
    y_trues.append(y[val_index])
    gridsearchs.append(gridsearch)

    # Put the results in a DataFrame
    df_pred = pd.DataFrame(y_preds[i], index=df.index.values[val_index],
                           columns=['predicted age'])
    df_res = pd.concat([df_val.set_index('subject_ID'), df_pred], axis=1,
                       sort=False).reset_index().rename(
                           columns={'index': 'subject_ID'})

    # Fill in missing values with the mean age
    # from the corresponding (site, gender) pair
    df_res.loc[df_res['predicted age'].isna(), 'predicted age'] = (
        df_res[df_res['predicted age'].isna()].apply(
            lambda x: df_train.groupby(['site', 'gender'])['age'].mean().loc[
                (x['site'], x['gender'])], axis=1)
    )

    # Save the results in a tsv file
    df_res = df_res.drop(['gender', 'site'], axis=1)
    df_res.to_csv('gm/Predictions/split-{0}.tsv'.format(i), sep='\t')

# Plot the performance on each fold
plot_results(y_trues, y_preds, marker='o', fillstyle='none',
             output_file='gm/SVM.png')
