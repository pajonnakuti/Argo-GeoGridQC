PS D:\INCOIS\Agro_project> python -u "d:\INCOIS\Agro_project\scripts\12_0_grid_29_ML.py"
=================================================================
  6-MODEL QC CLASSIFICATION  —  Grid_29
=================================================================

📂 Loading train : D:\INCOIS\Agro_project\data\Arabian_sea_gridwise_split\train\Grid_29_train.csv
📂 Loading test  : D:\INCOIS\Agro_project\data\Arabian_sea_gridwise_split\test\Grid_29_test.csv

  Train shape : (1059734, 26)
  Test  shape : (1219452, 26)

  Features used : 20

=================================================================
  TARGET : temp_qc
=================================================================

  Train rows  : 1,059,734
  Test  rows  : 1,219,452
  QC classes  : [1, 3, 4]

  Train class distribution:
    QC=1  →  1,004,273  (94.77%)
    QC=3  →    54,702  (5.16%)
    QC=4  →       759  (0.07%)

  Test class distribution:
    QC=1  →  1,219,093  (99.97%)
    QC=3  →         7  (0.00%)
    QC=4  →       352  (0.03%)

  ───────────────────────────────────────────────────────
  ▶  Model : RandomForest
  ───────────────────────────────────────────────────────
     Train time   : 220.2s
     Accuracy     : 0.9977  (99.77%)
     F1 Macro     : 0.3329
     F1 Weighted  : 0.9985

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      1.00      1.00   1219093
                  3       0.00      0.00      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           1.00   1219452
          macro avg       0.33      0.33      0.33   1219452
       weighted avg       1.00      1.00      1.00   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : ExtraTrees
  ───────────────────────────────────────────────────────
     Train time   : 53.6s
     Accuracy     : 0.9959  (99.59%)
     F1 Macro     : 0.3327
     F1 Weighted  : 0.9977

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      1.00      1.00   1219093
                  3       0.00      0.00      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           1.00   1219452
          macro avg       0.33      0.33      0.33   1219452
       weighted avg       1.00      1.00      1.00   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : XGBoost
  ───────────────────────────────────────────────────────
     Train time   : 95.1s
     Accuracy     : 0.9991  (99.91%)
     F1 Macro     : 0.3332
     F1 Weighted  : 0.9992

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      1.00      1.00   1219093
                  3       0.00      0.00      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           1.00   1219452
          macro avg       0.33      0.33      0.33   1219452
       weighted avg       1.00      1.00      1.00   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : CatBoost
  ───────────────────────────────────────────────────────
     Train time   : 317.6s
     Accuracy     : 0.9889  (98.89%)
     F1 Macro     : 0.3315
     F1 Weighted  : 0.9941

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.99      0.99   1219093
                  3       0.00      0.14      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           0.99   1219452
          macro avg       0.33      0.38      0.33   1219452
       weighted avg       1.00      0.99      0.99   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : LightGBM
  ───────────────────────────────────────────────────────
     Train time   : 51.1s
     Accuracy     : 0.9931  (99.31%)
     F1 Macro     : 0.3322
     F1 Weighted  : 0.9963

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.99      1.00   1219093
                  3       0.00      0.00      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           0.99   1219452
          macro avg       0.33      0.33      0.33   1219452
       weighted avg       1.00      0.99      1.00   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : LogisticRegression
  ───────────────────────────────────────────────────────
     Train time   : 1457.2s
     Accuracy     : 0.9415  (94.15%)
     F1 Macro     : 0.3234
     F1 Weighted  : 0.9696

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.94      0.97   1219093
                  3       0.00      0.43      0.00         7
                  4       0.00      0.00      0.00       352
       
           accuracy                           0.94   1219452
          macro avg       0.33      0.46      0.32   1219452
       weighted avg       1.00      0.94      0.97   1219452
       

=================================================================
  TARGET : psal_qc
=================================================================

  Train rows  : 1,059,734
  Test  rows  : 1,219,452
  QC classes  : [1, 2, 3, 4]

  Train class distribution:
    QC=1  →   972,343  (91.75%)
    QC=2  →    11,646  (1.10%)
    QC=3  →    65,470  (6.18%)
    QC=4  →    10,275  (0.97%)

  Test class distribution:
    QC=1  →  1,219,071  (99.97%)
    QC=4  →       381  (0.03%)

  ───────────────────────────────────────────────────────
  ▶  Model : RandomForest
  ───────────────────────────────────────────────────────
     Train time   : 242.0s
     Accuracy     : 0.9634  (96.34%)
     F1 Macro     : 0.2459
     F1 Weighted  : 0.9811

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.96      0.98   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.00      0.01      0.00       381
       
           accuracy                           0.96   1219452
          macro avg       0.25      0.24      0.25   1219452
       weighted avg       1.00      0.96      0.98   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : ExtraTrees
  ───────────────────────────────────────────────────────
     Train time   : 65.6s
     Accuracy     : 0.9049  (90.49%)
     F1 Macro     : 0.2379
     F1 Weighted  : 0.9498

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.91      0.95   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.00      0.02      0.00       381
       
           accuracy                           0.90   1219452
          macro avg       0.25      0.23      0.24   1219452
       weighted avg       1.00      0.90      0.95   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : XGBoost
  ───────────────────────────────────────────────────────
     Train time   : 135.1s
     Accuracy     : 0.9889  (98.89%)
     F1 Macro     : 0.2505
     F1 Weighted  : 0.9941

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.99      0.99   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.01      0.01      0.01       381
       
           accuracy                           0.99   1219452
          macro avg       0.25      0.25      0.25   1219452
       weighted avg       1.00      0.99      0.99   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : CatBoost
  ───────────────────────────────────────────────────────
     Train time   : 401.1s
     Accuracy     : 0.8132  (81.32%)
     F1 Macro     : 0.2250
     F1 Weighted  : 0.8967

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.81      0.90   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.00      0.12      0.00       381
       
           accuracy                           0.81   1219452
          macro avg       0.25      0.23      0.22   1219452
       weighted avg       1.00      0.81      0.90   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : LightGBM
  ───────────────────────────────────────────────────────
     Train time   : 61.4s
     Accuracy     : 0.9015  (90.15%)
     F1 Macro     : 0.2377
     F1 Weighted  : 0.9479

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.90      0.95   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.00      0.06      0.00       381
       
           accuracy                           0.90   1219452
          macro avg       0.25      0.24      0.24   1219452
       weighted avg       1.00      0.90      0.95   1219452
       

  ───────────────────────────────────────────────────────
  ▶  Model : LogisticRegression
  ───────────────────────────────────────────────────────
     Train time   : 1617.4s
     Accuracy     : 0.6141  (61.41%)
     F1 Macro     : 0.1906
     F1 Weighted  : 0.7608

     Classification Report:
                     precision    recall  f1-score   support
       
                  1       1.00      0.61      0.76   1219071
                  2       0.00      0.00      0.00         0
                  3       0.00      0.00      0.00         0
                  4       0.00      0.04      0.00       381
       
           accuracy                           0.61   1219452
          macro avg       0.25      0.16      0.19   1219452
       weighted avg       1.00      0.61      0.76   1219452
       

=================================================================
  FINAL RESULTS SUMMARY
=================================================================

  TARGET : temp_qc  (sorted by F1 Weighted ↓)
                model  accuracy  f1_macro  f1_weighted  train_time_s
1             XGBoost    0.9991    0.3332       0.9992         95.14
2        RandomForest    0.9977    0.3329       0.9985        220.24
3          ExtraTrees    0.9959    0.3327       0.9977         53.60
4            LightGBM    0.9931    0.3322       0.9963         51.12
5            CatBoost    0.9889    0.3315       0.9941        317.59
6  LogisticRegression    0.9415    0.3234       0.9696       1457.18

  TARGET : psal_qc  (sorted by F1 Weighted ↓)
                model  accuracy  f1_macro  f1_weighted  train_time_s
1             XGBoost    0.9889    0.2505       0.9941        135.07
2        RandomForest    0.9634    0.2459       0.9811        242.02
3          ExtraTrees    0.9049    0.2379       0.9498         65.64
4            LightGBM    0.9015    0.2377       0.9479         61.45
5            CatBoost    0.8132    0.2250       0.8967        401.10
6  LogisticRegression    0.6141    0.1906       0.7608       1617.44

✅ Results CSV saved → D:\INCOIS\Agro_project\results\Grid_29_models\Grid_29_6model_results.csv

=================================================================
  BEST MODEL PER TARGET  (by F1 Weighted)
=================================================================

  temp_qc
    🏆 Best Model  : XGBoost
       Accuracy    : 0.9991  (99.91%)
       F1 Weighted : 0.9992
       F1 Macro    : 0.3332
       Train time  : 95.14s

  psal_qc
    🏆 Best Model  : XGBoost
       Accuracy    : 0.9889  (98.89%)
       F1 Weighted : 0.9941
       F1 Macro    : 0.2505
       Train time  : 135.07s

=================================================================
  ✅ COMPLETE — Grid_29  |  6 Models  |  2 Targets
=================================================================
PS D:\INCOIS\Agro_project> 