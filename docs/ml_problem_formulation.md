# Machine Learning Problem Formulation, Evaluation Metrics, and Baseline Specification

## 1. Context and Target Climate Hazards

The intelligent prediction subsystem monitors continuous environmental telemetry to forecast climate hazards across three critical categories:

1. **Agricultural & Hydrological Drought:** Monitored through prolonged deficits in precipitation, elevated vapor pressure deficits, and depleted root-zone soil moisture ($0\text{--}20\text{ cm}$ and $20\text{--}100\text{ cm}$). Standardized Precipitation Indices (SPI-30/SPI-90) and soil moisture depletion ratios serve as target proxies.
2. **Extreme Temperature Events (Heatwaves & Sudden Freezes):** Operationalized as sustained anomalies exceeding localized historical percentiles ($T_{\max} > 95\text{th percentile}$ for heatwaves; $T_{\min} < 0^\circ\text{C}$ or rapid drop $> 10^\circ\text{C} / 24\text{h}$ for sudden freezes).
3. **Flash Floods and Extreme Precipitation:** Monitored via cumulative hourly precipitation intensity, soil saturation thresholds, and rapid atmospheric pressure drops ($\Delta P / \Delta t$).

---

## 2. Mathematical Problem Formulation

The prediction task is framed as a **hybrid multi-horizon multivariate time-series forecasting and sequential risk classification** problem.

### 2.1 Input Representation
Let continuous sensor telemetry collected over discrete time intervals $t \in \{1, 2, \dots, T\}$ be defined as an observation vector $\mathbf{x}_t \in \mathbb{R}^D$:

$$\mathbf{x}_t = [T_t, RH_t, P_t, SM_t, R_t, WS_t, \dots]^\top$$

where:
- $T_t$: Ambient air temperature ($^\circ\text{C}$)
- $RH_t$: Relative humidity ($\%$)
- $P_t$: Atmospheric surface pressure ($\text{hPa}$)
- $SM_t$: Volumetric soil moisture ($\text{m}^3/\text{m}^3$)
- $R_t$: Rainfall accumulation ($\text{mm}$)
- $WS_t$: Wind speed ($\text{m/s}$)

For a lookback window of length $W \in \mathbb{N}$, the input history matrix $\mathbf{X}_t \in \mathbb{R}^{W \times D}$ is defined as:

$$\mathbf{X}_t = [\mathbf{x}_{t-W+1}, \mathbf{x}_{t-W+2}, \dots, \mathbf{x}_t]^\top$$

### 2.2 Task A: Multi-Horizon Regression (Continuous Values)
The model maps historical observations $\mathbf{X}_t$ to predicted physical parameters over a forecast horizon $H$:

$$\hat{\mathbf{Y}}_{t+1:t+H} = f_\theta(\mathbf{X}_t), \quad \hat{\mathbf{Y}} \in \mathbb{R}^{H \times M}$$

where $M \le D$ is the subset of target physical variables (e.g., Temperature, Soil Moisture, Precipitation), and $\theta$ denotes learnable model parameters.

### 2.3 Task B: Climate Risk Classification (Hazard Tiers)
Given predicted physical trajectories $\hat{\mathbf{Y}}_{t+1:t+H}$ and environmental context $\mathbf{X}_t$, the secondary mapping estimates the probability distribution across discrete threat severity levels:

$$\hat{\mathbf{p}}_{t+h} = g_\phi(\mathbf{X}_t, \hat{\mathbf{y}}_{t+h}) \in [0, 1]^K, \quad \sum_{k=1}^K \hat{p}_{t+h}^{(k)} = 1$$

where $K = 3$ represents the risk classes:
- **Class 0 (Normal):** Telemetry within safe operating climatological boundaries.
- **Class 1 (Warning):** Approaching dangerous conditions (e.g., $90\text{th percentile}$ thresholds, progressive soil moisture depletion). Preventive actions advised.
- **Class 2 (Critical):** Immediate hazardous conditions confirmed (e.g., torrential rain $> 30\text{ mm/h}$, extreme heat $> 40^\circ\text{C}$). Automated emergency alerts triggered.

The final predicted risk category $\hat{C}_{t+h}$ is resolved using risk-sensitive decision thresholds $\tau$:

$$\hat{C}_{t+h} = \arg\max_{k \in \{0, 1, 2\}} \left( \hat{p}_{t+h}^{(k)} \cdot w_k \right)$$

where $w_k$ denotes the asymmetry penalty weight ($w_{\text{Critical}} > w_{\text{Warning}} > w_{\text{Normal}}$) to minimize false negatives for disaster events.

---

## 3. Evaluation Metrics

Because climate extremes are characterized by high consequence and low frequency (severe class imbalance), standard symmetrical metrics are insufficient. The subsystem employs dual evaluation criteria.

### 3.1 Continuous Regression Metrics

1. **Mean Absolute Error (MAE):** Quantifies baseline prediction error without overly penalizing isolated extreme spikes:

$$\text{MAE} = \frac{1}{H} \sum_{h=1}^H |y_{t+h} - \hat{y}_{t+h}|$$

2. **Root Mean Squared Error (RMSE):** Strongly penalizes large prediction errors, critical for detecting rapid temperature or precipitation spikes:

$$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{h=1}^H (y_{t+h} - \hat{y}_{t+h})^2}$$

3. **Continuous Ranked Probability Score (CRPS):** (For probabilistic forecasts) Measures both sharpness and calibration of predictive uncertainty distributions.

### 3.2 Hazard Classification and Alert Metrics

Standard accuracy is explicitly rejected due to majority class dominance ($>90\%$ Normal).

1. **Precision & Recall (Per-Class & Macro):**
   - $\text{Recall}_{\text{Critical}} = \frac{TP}{TP + FN}$: Core safety metric. Demonstrates the proportion of actual hazardous events successfully caught.
   - $\text{Precision}_{\text{Critical}} = \frac{TP}{TP + FP}$: Operational usability metric. Prevents warning fatigue caused by false alarms.

2. **Macro-Averaged and Minor-Class $F_1$-Score ($F_{\beta}$ with $\beta = 2$):**

$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

Setting $\beta = 2$ weights recall twice as heavily as precision to strictly penalize missed extreme events ($FN$).

3. **Precision-Recall Area Under Curve (PR-AUC):** Preferred over ROC-AUC due to high skewness in disaster event frequencies.

---

## 4. Baseline Benchmarks

All subsequent machine learning models (e.g., Gradient Boosted Trees, LSTM, PatchTST) will be compared against standardized, reproducible baselines:

### 4.1 Naive Persistence Model (Random Walk)
Assumes environmental conditions remain constant over the prediction horizon:

$$\hat{y}_{t+h} = y_t, \quad \forall h \in \{1, \dots, H\}$$

*Purpose:* Serves as the zero-intelligence benchmark. Any valid ML model must demonstrate statistically significant improvement over persistence, especially across horizons $h > 3\text{h}$.

### 4.2 Moving Average (Climatological MA)
Computes the rolling mean of the target variable over an empirical historical window $k$:

$$\hat{y}_{t+h} = \frac{1}{k} \sum_{i=0}^{k-1} y_{t-i}$$

*Purpose:* Captures stationary diurnal trends without dynamic adjustment to abrupt atmospheric transitions.

### 4.3 Static Threshold Heuristic (Rule-Based Classifier)
A non-ML deterministic rule that classifies risk based purely on instantaneous sensor values breaching fixed thresholds:

$$\hat{C}_t = \begin{cases} 
\text{Critical}, & \text{if } T_t > T_{\text{crit}} \lor R_t > R_{\text{crit}} \lor SM_t < SM_{\text{crit}} \\
\text{Warning}, & \text{if } T_t > T_{\text{warn}} \lor R_t > R_{\text{warn}} \lor SM_t < SM_{\text{warn}} \\
\text{Normal}, & \text{otherwise}
\end{cases}$$

*Purpose:* Demonstrates the added value of predictive ML lead times ($12\text{--}48\text{ hours}$ in advance) versus purely reactive threshold monitoring.