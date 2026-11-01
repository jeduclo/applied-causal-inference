import {
  BrainCircuit, ChartLine, Database, FlaskConical, LayoutDashboard, Network, Scale, ShieldCheck,
  type LucideIcon,
} from "lucide-react";

/** Course-wide information (edit freely). */
export const COURSE = {
  title: "Applied Causal Inference & Decision Science",
  program: "Specialty course",
  tagline: "An 80-day, full-stack course that takes economists from raw data to a defensible causal estimate to a decision.",
  author: "Jean Alegue, PhD",
  hoursPerDay: 5,
} as const;

export type Part = { id: number; title: string; days: string };

export const PARTS: Part[] = [
  { id: 1, title: "Causal Foundations", days: "Days 1–30" },
  { id: 2, title: "Identification & Estimation", days: "Days 31–60" },
  { id: 3, title: "Decisions & Delivery", days: "Days 61–80" },
];

export type Topic = { days: string; text: string };

export type ModuleMeta = {
  number: number;
  slug: string;
  title: string;
  short: string;          // label on the tab bar
  part: number;
  firstDay: number;
  lastDay: number;
  accent: string;         // UI accent for this module
  icon: LucideIcon;
  focus: string;
  tools: string[];
  topics: Topic[];        // the curriculum plan, three blocks per module
  plan: string[];         // planned title of each of the 10 days (used until the notes are written)
  capstone: string;       // the module's mini-capstone
};

export const MODULES: ModuleMeta[] = [
  {
    number: 1, slug: "causal-data", title: "Causal-Ready Data Engineering", short: "Causal Data", part: 1,
    firstDay: 1, lastDay: 10, accent: "#2dd4bf", icon: Database,
    focus: "Turning messy transactions into tested panels with treatment, timing, outcomes and pre-period covariates.",
    tools: ["Python", "Polars", "DuckDB", "dlt", "dbt", "Prefect", "Git"],
    topics: [
      { days: "Days 1–3", text: "Causal vocabulary, Polars cleaning, DuckDB and the customer × week panel." },
      { days: "Days 4–7", text: "Ingesting API data with dlt and modelling the panel in dbt." },
      { days: "Days 8–10", text: "Data contracts for leakage and treatment timing, orchestration with Prefect." },
    ],
    plan: [
      "From a Causal Question to a Dataset", "Cleaning and Shaping Data for a Causal Question",
      "DuckDB and the Customer × Week Panel", "APIs and JSON for Economists", "Ingestion Pipelines with dlt",
      "dbt Foundations: Sources, Staging and Tests", "dbt Marts: The Panel as a Model",
      "Data Contracts: Leakage and Treatment Timing", "Orchestration with Prefect", "Capstone: The Causal Data Mart",
    ],
    capstone: "Causal Data Mart: a dbt-modelled customer × week panel with contracts that catch post-treatment leakage.",
  },
  {
    number: 2, slug: "dags", title: "Causal Thinking, Estimands & DAGs", short: "DAGs", part: 1,
    firstDay: 11, lastDay: 20, accent: "#38bdf8", icon: Network,
    focus: "Stating the question precisely and making every causal assumption explicit.",
    tools: ["DoWhy", "NetworkX", "statsmodels", "Polars", "matplotlib"],
    topics: [
      { days: "Days 11–13", text: "Potential outcomes, estimands (ATE, ATT, CATE, LATE) and randomization as the benchmark." },
      { days: "Days 14–17", text: "DAGs, backdoor and frontdoor criteria, colliders and bad controls." },
      { days: "Days 18–20", text: "Encoding assumptions in DoWhy and defending a DAG in writing." },
    ],
    plan: [
      "Potential Outcomes and the Fundamental Problem", "Estimands: ATE, ATT, CATE, LATE",
      "Randomization as the Benchmark", "Drawing DAGs", "The Backdoor Criterion and Adjustment Sets",
      "Colliders and Selection Bias", "Frontdoor Paths, Mediators and Bad Controls",
      "Encoding Assumptions with DoWhy", "Simpson's Paradox in Business Data", "Capstone: Defending the DAG",
    ],
    capstone: "Defending the DAG: map pricing and email decisions, find adjustment sets, and write an assumptions memo.",
  },
  {
    number: 3, slug: "experiments", title: "Experimentation & A/B Testing", short: "Experiments", part: 1,
    firstDay: 21, lastDay: 30, accent: "#a78bfa", icon: FlaskConical,
    focus: "Designing, running and analysing randomized experiments well.",
    tools: ["statsmodels", "SciPy", "Polars", "NumPy"],
    topics: [
      { days: "Days 21–23", text: "Test design, power and sample size, stratified randomization and balance." },
      { days: "Days 24–27", text: "Regression analysis, CUPED, multiple testing, peeking and sequential tests." },
      { days: "Days 28–30", text: "Interference, switchback and geo experiments, and the free-shipping capstone." },
    ],
    plan: [
      "Designing an A/B Test", "Power and Sample Size", "Randomization, Stratification and Balance",
      "Analysing Experiments with Regression", "CUPED and Variance Reduction", "Multiple Testing and Peeking",
      "Sequential Testing", "Interference and Network Effects", "Switchback and Geo Experiments",
      "Capstone: The Free-Shipping Experiment",
    ],
    capstone: "Free-shipping experiment: pre-registered plan, CUPED readout, non-inferiority guardrail and a launch decision by segment.",
  },
  {
    number: 4, slug: "observables-iv", title: "Selection on Observables & Instrumental Variables", short: "Matching & IV",
    part: 2, firstDay: 31, lastDay: 40, accent: "#fbbf24", icon: Scale,
    focus: "Estimating effects without randomization when confounders are measured, or an instrument exists.",
    tools: ["statsmodels", "scikit-learn", "linearmodels", "Polars"],
    topics: [
      { days: "Days 31–33", text: "Regression adjustment, propensity scores and matching." },
      { days: "Days 34–36", text: "Inverse probability weighting, overlap and doubly robust (AIPW) estimators." },
      { days: "Days 37–40", text: "Instrumental variables, 2SLS, weak instruments and LATE." },
    ],
    plan: [
      "Regression Adjustment Done Right", "Propensity Scores", "Matching", "Inverse Probability Weighting",
      "Overlap and Trimming", "Doubly Robust Estimation (AIPW)", "Instrumental Variables: Intuition and 2SLS",
      "Weak Instruments and LATE", "Comparing Estimators", "Capstone: The Loyalty Programme Effect",
    ],
    capstone: "Loyalty programme effect: compare AIPW and IV estimates of the effect on 12-month spend, then decide whether the programme and its $20 invitations pay off.",
  },
  {
    number: 5, slug: "quasi-experiments", title: "Quasi-Experiments: DiD, Synthetic Control, RDD", short: "Quasi-Experiments",
    part: 2, firstDay: 41, lastDay: 50, accent: "#fb7185", icon: ChartLine,
    focus: "Evaluating rollouts and campaigns when you can't randomize.",
    tools: ["pyfixest", "CausalPy", "PyMC", "rdrobust", "rddensity"],
    topics: [
      { days: "Days 41–44", text: "Difference-in-differences, event studies and modern staggered estimators." },
      { days: "Days 45–47", text: "Synthetic control, synthetic DiD and Bayesian structural time series." },
      { days: "Days 48–50", text: "Regression discontinuity and placebo tests." },
    ],
    plan: [
      "Difference-in-Differences Basics", "Event Studies and Pre-Trends", "Staggered Adoption: What Breaks TWFE",
      "Modern DiD Estimators", "Synthetic Control", "Synthetic Difference-in-Differences",
      "Bayesian Structural Time Series with CausalPy", "Regression Discontinuity", "Placebo and Falsification Tests",
      "Capstone: Regional Campaign ROI",
    ],
    capstone: "Regional campaign ROI: DiD, synthetic control, synthetic DiD and a Bayesian model, stress-tested with placebos and turned into a profit decision.",
  },
  {
    number: 6, slug: "causal-ml", title: "Causal Machine Learning & Heterogeneity", short: "Causal ML", part: 2,
    firstDay: 51, lastDay: 60, accent: "#34d399", icon: BrainCircuit,
    focus: "Using machine learning to estimate effects in high dimensions and find who responds.",
    tools: ["DoubleML", "EconML", "LightGBM"],
    topics: [
      { days: "Days 51–54", text: "Orthogonalization, cross-fitting and DoubleML for partially linear and interactive models." },
      { days: "Days 55–57", text: "Meta-learners, causal forests and validating heterogeneous effects." },
      { days: "Days 58–60", text: "Segment-level price elasticities." },
    ],
    plan: [
      "From Prediction to Causal ML", "Neyman Orthogonality and Cross-Fitting", "DoubleML: Partially Linear Models",
      "DoubleML: Interactive Models and IV", "Heterogeneous Effects and Meta-Learners", "Causal Forests",
      "Validating CATEs: RATE, Qini and Calibration", "Price Elasticity Estimation", "Segment-Level Elasticities",
      "Capstone: The Price Elasticity Engine",
    ],
    capstone: "Price elasticity engine: which customer segments can absorb a 5% price rise?",
  },
  {
    number: 7, slug: "robustness-decisions", title: "Sensitivity, Robustness & Decisions", short: "Decisions", part: 3,
    firstDay: 61, lastDay: 70, accent: "#fb923c", icon: ShieldCheck,
    focus: "Stress-testing estimates and turning them into targeting policies and choices under uncertainty.",
    tools: ["DoubleML", "scikit-uplift", "EconML"],
    topics: [
      { days: "Days 61–63", text: "Sensitivity to hidden confounding, E-values, refutation and specification curves." },
      { days: "Days 64–66", text: "Uplift modelling, policy learning and cost-benefit analysis." },
      { days: "Days 67–70", text: "Decisions under uncertainty, value of information and the decision memo." },
    ],
    plan: [
      "Sensitivity to Hidden Confounding", "E-values and Robustness Values", "Refutation Tests and Specification Curves",
      "Uplift Modelling", "Policy Learning", "Cost-Benefit Analysis", "Decisions under Uncertainty",
      "The Value of Information", "Writing the Decision Memo", "Capstone: The Targeting Policy",
    ],
    capstone: "Targeting policy: an uplift-based discount policy versus treat-all and treat-none, with robustness checks.",
  },
  {
    number: 8, slug: "delivery", title: "Decision Intelligence Delivery", short: "Delivery", part: 3,
    firstDay: 71, lastDay: 80, accent: "#e879f9", icon: LayoutDashboard,
    focus: "Putting causal estimates in front of decision-makers as tools they can use.",
    tools: ["FastAPI", "Streamlit", "Next.js", "Recharts"],
    topics: [
      { days: "Days 71–73", text: "Serving causal models with FastAPI, scenario design and uncertainty for executives." },
      { days: "Days 74–77", text: "Building, testing and deploying the scenario simulator; monitoring estimates." },
      { days: "Days 78–80", text: "When to re-run experiments, and the final capstone defence." },
    ],
    plan: [
      "Serving Causal Models with FastAPI", "Scenario and What-If Design", "Communicating Uncertainty to Executives",
      "Building the Simulator Front End", "Charts for Counterfactuals", "Testing and Deploying the App",
      "Monitoring Estimates over Time", "When to Re-Run the Experiment", "Final Capstone Workshop",
      "Final Capstone Defence",
    ],
    capstone: "Scenario simulator: pricing and marketing sliders showing projected vs counterfactual revenue.",
  },
];

export const TOTAL_DAYS = 80;

export function moduleBySlug(slug: string) {
  return MODULES.find((m) => m.slug === slug);
}

export function moduleForDay(day: number) {
  return MODULES.find((m) => day >= m.firstDay && day <= m.lastDay);
}

export function moduleHref(m: ModuleMeta) {
  return `/module/${m.slug}`;
}

export function dayHref(day: number) {
  const m = moduleForDay(day);
  return m ? `/module/${m.slug}/day/${day}` : "/";
}

/** The topic block ("Days 4–7") a day belongs to. */
export function topicForDay(day: number): Topic | undefined {
  const m = moduleForDay(day);
  return m?.topics.find((t) => {
    const [a, b] = t.days.replace("Days ", "").split("–").map(Number);
    return day >= a && day <= b;
  });
}
