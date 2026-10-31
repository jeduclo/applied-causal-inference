/**
 * ─────────────────────────────────────────────────────────────────────────────
 *  YouTube video for each day: paste a link between the quotes.
 *
 *  Any of these formats work:
 *    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
 *    "https://youtu.be/dQw4w9WgXcQ"
 *    "https://www.youtube.com/embed/dQw4w9WgXcQ"
 *    "dQw4w9WgXcQ"
 *
 *  Leave "" to show the "coming soon" placeholder.
 *  Tip: "Unlisted" YouTube videos embed fine and do not appear in search.
 * ─────────────────────────────────────────────────────────────────────────────
 */
export const VIDEOS: Record<number, string> = {

  // ── Module 1: Causal-Ready Data Engineering ──
   1: "", // From a Causal Question to a Dataset
   2: "", // Cleaning and Shaping Data for a Causal Question
   3: "", // DuckDB and the Customer × Week Panel
   4: "", // APIs and JSON for Economists
   5: "", // Ingestion Pipelines with dlt
   6: "", // dbt Foundations: Sources, Staging and Tests
   7: "", // dbt Marts: The Panel as a Model
   8: "", // Data Contracts: Leakage and Treatment Timing
   9: "", // Orchestration with Prefect
  10: "", // Capstone: The Causal Data Mart

  // ── Module 2: Causal Thinking, Estimands & DAGs ──
  11: "", // Potential Outcomes and the Fundamental Problem
  12: "", // Estimands: ATE, ATT, CATE, LATE
  13: "", // Randomization as the Benchmark
  14: "", // Drawing DAGs
  15: "", // The Backdoor Criterion and Adjustment Sets
  16: "", // Colliders and Selection Bias
  17: "", // Frontdoor Paths, Mediators and Bad Controls
  18: "", // Encoding Assumptions with DoWhy
  19: "", // Simpson's Paradox in Business Data
  20: "", // Capstone: Defending the DAG

  // ── Module 3: Experimentation & A/B Testing ──
  21: "", // Designing an A/B Test
  22: "", // Power and Sample Size
  23: "", // Randomization, Stratification and Balance
  24: "", // Analysing Experiments with Regression
  25: "", // CUPED and Variance Reduction
  26: "", // Multiple Testing and Peeking
  27: "", // Sequential Testing
  28: "", // Interference and Network Effects
  29: "", // Switchback and Geo Experiments
  30: "", // Capstone: The Free-Shipping Experiment

  // ── Module 4: Selection on Observables & Instrumental Variables ──
  31: "", // Regression Adjustment Done Right
  32: "", // Propensity Scores
  33: "", // Matching
  34: "", // Inverse Probability Weighting
  35: "", // Overlap and Trimming
  36: "", // Doubly Robust Estimation (AIPW)
  37: "", // Instrumental Variables: Intuition and 2SLS
  38: "", // Weak Instruments and LATE
  39: "", // Comparing Estimators
  40: "", // Capstone: The Loyalty Programme Effect

  // ── Module 5: Quasi-Experiments: DiD, Synthetic Control, RDD ──
  41: "", // Difference-in-Differences Basics
  42: "", // Event Studies and Pre-Trends
  43: "", // Staggered Adoption: What Breaks TWFE
  44: "", // Modern DiD Estimators
  45: "", // Synthetic Control
  46: "", // Synthetic Difference-in-Differences
  47: "", // Bayesian Structural Time Series with CausalPy
  48: "", // Regression Discontinuity
  49: "", // Placebo and Falsification Tests
  50: "", // Capstone: Regional Campaign ROI

  // ── Module 6: Causal Machine Learning & Heterogeneity ──
  51: "", // From Prediction to Causal ML
  52: "", // Neyman Orthogonality and Cross-Fitting
  53: "", // DoubleML: Partially Linear Models
  54: "", // DoubleML: Interactive Models and IV
  55: "", // Heterogeneous Effects and Meta-Learners
  56: "", // Causal Forests
  57: "", // Validating CATEs: RATE, Qini and Calibration
  58: "", // Price Elasticity Estimation
  59: "", // Segment-Level Elasticities
  60: "", // Capstone: The Price Elasticity Engine

  // ── Module 7: Sensitivity, Robustness & Decisions ──
  61: "", // Sensitivity to Hidden Confounding
  62: "", // E-values and Robustness Values
  63: "", // Refutation Tests and Specification Curves
  64: "", // Uplift Modelling
  65: "", // Policy Learning
  66: "", // Cost-Benefit Analysis
  67: "", // Decisions under Uncertainty
  68: "", // The Value of Information
  69: "", // Writing the Decision Memo
  70: "", // Capstone: The Targeting Policy

  // ── Module 8: Decision Intelligence Delivery ──
  71: "", // Serving Causal Models with FastAPI
  72: "", // Scenario and What-If Design
  73: "", // Communicating Uncertainty to Executives
  74: "", // Building the Simulator Front End
  75: "", // Charts for Counterfactuals
  76: "", // Testing and Deploying the App
  77: "", // Monitoring Estimates over Time
  78: "", // When to Re-Run the Experiment
  79: "", // Final Capstone Workshop
  80: "", // Final Capstone Defence
};
