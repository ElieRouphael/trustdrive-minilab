# TrustDrive MiniLab - benchmark results

Perception backend: **synthetic**

### Aggregated over all scenarios

| method   | RMSE e_y [m] | max|e_y| [m] | lane exits | steer jerk | unsafe AI [%] | auth A/S/H [%] |
| -------- | ------------ | ------------ | ---------- | ---------- | ------------- | -------------- |
| naive    | 0.126        | 0.617        | 1          | 30.56      | 16.4          | 100/0/0        |
| handover | 0.101        | 0.400        | 0          | 13.93      | 8.3           | 89/0/11        |
| proposed | 0.042        | 0.315        | 0          | 8.79       | 0.0           | 81/2/17        |

### Scenario: nominal

| method   | RMSE e_y [m] | max|e_y| [m] | lane exits | steer jerk | unsafe AI [%] | auth A/S/H [%] |
| -------- | ------------ | ------------ | ---------- | ---------- | ------------- | -------------- |
| naive    | 0.035        | 0.315        | 0          | 12.10      | 0.0           | 100/0/0        |
| handover | 0.035        | 0.315        | 0          | 12.10      | 0.0           | 100/0/0        |
| proposed | 0.036        | 0.315        | 0          | 6.20       | 0.0           | 100/0/0        |

### Scenario: fog

| method   | RMSE e_y [m] | max|e_y| [m] | lane exits | steer jerk | unsafe AI [%] | auth A/S/H [%] |
| -------- | ------------ | ------------ | ---------- | ---------- | ------------- | -------------- |
| naive    | 0.245        | 1.298        | 4          | 82.08      | 32.2          | 100/0/0        |
| handover | 0.145        | 0.430        | 0          | 15.57      | 0.0           | 57/0/42        |
| proposed | 0.045        | 0.315        | 0          | 10.46      | 0.0           | 58/8/34        |

### Scenario: ood_curve

| method   | RMSE e_y [m] | max|e_y| [m] | lane exits | steer jerk | unsafe AI [%] | auth A/S/H [%] |
| -------- | ------------ | ------------ | ---------- | ---------- | ------------- | -------------- |
| naive    | 0.187        | 0.539        | 0          | 15.96      | 33.3          | 100/0/0        |
| handover | 0.187        | 0.539        | 0          | 15.96      | 33.3          | 100/0/0        |
| proposed | 0.052        | 0.315        | 0          | 11.14      | 0.2           | 66/1/34        |

### Scenario: conflict

| method   | RMSE e_y [m] | max|e_y| [m] | lane exits | steer jerk | unsafe AI [%] | auth A/S/H [%] |
| -------- | ------------ | ------------ | ---------- | ---------- | ------------- | -------------- |
| naive    | 0.035        | 0.315        | 0          | 12.10      | 0.0           | 100/0/0        |
| handover | 0.035        | 0.315        | 0          | 12.10      | 0.0           | 100/0/0        |
| proposed | 0.036        | 0.315        | 0          | 7.38       | 0.0           | 100/0/0        |
