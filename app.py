# app.py — Insurance Analytics Chatbot (Complete Updated Version)

import streamlit as st
import pandas as pd
import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px

load_dotenv()

st.set_page_config(
    page_title="Insurance Analytics Chatbot",
    page_icon="📊",
    layout="wide"
)

# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("insurance_data.csv")
    df.columns = df.columns.str.strip().str.upper()
    return df

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error("❌ File not found: data/insurance_data.csv — place your CSV inside the data/ folder.")
    data_loaded = False
    df = None
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    data_loaded = False
    df = None

# ══════════════════════════════════════════════════════
# 2. GROQ CLIENT
# ══════════════════════════════════════════════════════
load_dotenv()

# Works locally (.env) AND on Streamlit Cloud (secrets)
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY", "")

# ── GROQ CLIENT ──
try:
    client = Groq(api_key=groq_api_key)
    groq_ready = True
except Exception as e:
    st.error(f"❌ Groq init failed: {e}")
    groq_ready = False

# ══════════════════════════════════════════════════════
# 3. SYSTEM PROMPT — Dynamic, works with any data size
# ══════════════════════════════════════════════════════
def build_system_prompt(df):
    fy_years = sorted(df['FY_YEAR'].dropna().unique().tolist())

    prompt = f"""You are an expert insurance data analyst assistant.
You have access to a pandas DataFrame called `df` with {len(df):,} rows.

COLUMN REFERENCE GUIDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSIONAL COLUMNS (use for groupby / filter):
  FY_YEAR              → Financial Year. Known values: {fy_years}
  IMD_CHANNEL          → IMD Channel / Channel Name
  REN_ROLL_NB_FLAG     → Business Type (e.g. NEW, RENEWAL)
  M_VEHICLE_SEGMENT    → Vehicle Segment
  V_VEHICLE_TYPE       → Vehicle Type
  GEO_FLAG             → Geo Category (e.g. Urban, Rural)
  I_IMD_NAME           → IMD Name
  RISKZONE             → Risk Zone Category
  FLAG_POLICYTYPE      → Policy Type
  FUEL_TYPE            → Fuel Type (e.g. Petrol, Diesel, CNG, Electric)
  VEHICLE_CATEGORY     → Vehicle Category
  VEHICLE_AGE_GROUP    → Vehicle Age Group
  P_ACC_LOB            → Line of Business (LOB)
  CUSTOMER_STATE       → State (Indian state name)
  CUSTOMER_DISTRICT    → District
  PT_PARTNER_PIN_CODE  → Pin Code
  ORG_SUB_CHANNEL_NAME → Sub Channel Name
  SUBIMD_NAME          → Sub IMD Name

METRIC COLUMNS (numeric — use for aggregation):
  OD_NET_EARNED_PREMIUM → OD Net Earned Premium
  OD_CLAIM_AMOUNT       → OD Claim Amount
  OD_NOC                → OD Claim Count
  OD_NOP                → OD Number of Policies
  TP_NET_EARNED_PREMIUM → TP Net Earned Premium
  TP_CLAIM_AMOUNT       → TP Claim Amount
  TP_NOC                → TP Claim Count
  TP_NOP                → TP Number of Policies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEASURE DEFINITIONS (use these EXACT formulas inside lambda x):
  OD NOP           → int(x['OD_NOP'].sum())
  OD Premium       → round(x['OD_NET_EARNED_PREMIUM'].sum(), 2)
  OD Claims        → int(x['OD_NOC'].sum())
  OD Claim Amount  → round(x['OD_CLAIM_AMOUNT'].sum(), 2)
  OD Frequency %   → round(x['OD_NOC'].sum() / x['OD_NOP'].sum() * 100, 2)
  OD Loss Ratio %  → round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)
  TP NOP           → int(x['TP_NOP'].sum())
  TP Premium       → round(x['TP_NET_EARNED_PREMIUM'].sum(), 2)
  TP Claims        → int(x['TP_NOC'].sum())
  TP Claim Amount  → round(x['TP_CLAIM_AMOUNT'].sum(), 2)
  TP Frequency %   → round(x['TP_NOC'].sum() / x['TP_NOP'].sum() * 100, 2)
  TP Loss Ratio %  → round(x['TP_CLAIM_AMOUNT'].sum() / x['TP_NET_EARNED_PREMIUM'].sum() * 100, 2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER ALIASES — map user words to correct columns:
  "channel"              → IMD_CHANNEL
  "business type"        → REN_ROLL_NB_FLAG
  "new/renewal"          → REN_ROLL_NB_FLAG
  "state"                → CUSTOMER_STATE
  "district"             → CUSTOMER_DISTRICT
  "vehicle type"         → V_VEHICLE_TYPE
  "lob"                  → P_ACC_LOB
  "zone" / "risk zone"   → RISKZONE
  "imd"                  → I_IMD_NAME
  "sub channel"          → ORG_SUB_CHANNEL_NAME
  "OD LR" / "OD loss ratio" / "loss ratio" → OD Loss Ratio formula (default)
  "TP LR" / "TP loss ratio"                → TP Loss Ratio formula
  "claims"               → OD_CLAIM_AMOUNT (default unless user says TP)
  "premium"              → OD_NET_EARNED_PREMIUM (default unless user says TP)
  "NOP"                  → OD_NOP (default unless user says TP)
  "frequency"            → OD Frequency formula (default unless user says TP)
  "FY 2024" / "2024-25"  → 'FY-2024-25'
  "FY 2023" / "2023-24"  → 'FY-2023-24'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL PANDAS RULES:

RULE 0 — OUTPUT FORMAT:
  Return ONLY valid JSON on ONE LINE — no markdown, no backslashes, no newlines inside values:
  {{"pandas_code": "your_single_line_code", "explanation": "plain English explanation"}}

RULE 1 — FY_YEAR filter must use EXACT values from {fy_years}:
  ✅ df[df['FY_YEAR'] == 'FY-2024-25']
  ❌ df[df['FY_YEAR'] == '2024-25']

RULE 2 — NEVER hardcode state/channel/district names.
  Always use .str.upper() so it works with any data:
  ✅ df[df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA']
  ✅ Dynamic: df[df['CUSTOMER_STATE'] == df[...].idxmax()]

RULE 3 — For MULTIPLE METRICS in ONE result, use single groupby + apply + pd.Series:
  df[df['FY_YEAR'] == 'FY-2024-25'].groupby('CUSTOMER_STATE').apply(lambda x: pd.Series({{'OD_LR_%': round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2), 'OD_PREMIUM': round(x['OD_NET_EARNED_PREMIUM'].sum(), 2), 'OD_NOP': int(x['OD_NOP'].sum())}})).sort_values('OD_LR_%', ascending=False).head(5).reset_index()

RULE 4 — Loss Ratio ALWAYS multiplied by 100 and rounded to 2 decimal places.

RULE 5 — Year-on-year comparison (both years as columns):
  Use groupby with FY_YEAR + unstack — do NOT filter by year:
  df.groupby(['CUSTOMER_STATE', 'FY_YEAR']).apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).unstack('FY_YEAR').rename(columns={{'FY-2023-24': 'OD_LR_FY23-24_%', 'FY-2024-25': 'OD_LR_FY24-25_%'}}).reset_index()

RULE 6 — Two-step questions (find top X THEN drill down) = TWO semicolon-separated statements:
  Statement 1 → find the top entity
  Statement 2 → drill into sub-dimension dynamically using idxmax()
  Example (top state then its districts):
  df[df['FY_YEAR'] == 'FY-2024-25'].groupby('CUSTOMER_STATE').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).sort_values(ascending=False).head(1).reset_index().rename(columns={{0: 'OD_LR_FY24-25_%'}}); df[df['CUSTOMER_STATE'] == df[df['FY_YEAR'] == 'FY-2024-25'].groupby('CUSTOMER_STATE').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).idxmax()].groupby(['CUSTOMER_DISTRICT', 'FY_YEAR']).apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).unstack('FY_YEAR').rename(columns={{'FY-2023-24': 'OD_LR_FY23-24_%', 'FY-2024-25': 'OD_LR_FY24-25_%'}}).reset_index()

RULE 7 — Always call .reset_index() at end.
RULE 8 — Never use import statements inside the code.

RULE 9 — For SINGLE VALUE queries (no groupby needed):
  When user asks for metric of ONE specific entity (one state, one channel etc.),
  do NOT use .apply() or .groupby(). Use direct column aggregation instead.

  ✅ CORRECT — single state loss ratio:
  round(df[df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA']['OD_CLAIM_AMOUNT'].sum() / df[df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA']['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)

  ✅ CORRECT — single state with FY filter:
  round(df[(df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA') & (df['FY_YEAR'] == 'FY-2024-25')]['OD_CLAIM_AMOUNT'].sum() / df[(df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA') & (df['FY_YEAR'] == 'FY-2024-25')]['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)

  ✅ CORRECT — single state multiple metrics:
  df[df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA'].agg(OD_LR_PCT=('OD_CLAIM_AMOUNT', 'sum'))

  ❌ WRONG — never use .apply() without .groupby() for single entity:
  df[df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA'].apply(lambda x: x['OD_CLAIM_AMOUNT'].sum() ...)

  DECISION RULE:
  - User mentions ONE specific name + asks for a metric  → direct aggregation, NO groupby
  - User asks "top N" or "by state/channel" → use groupby + apply
  - User asks "compare" or "all states" → use groupby + apply

  RULE 11 — sort_values() ALWAYS needs 'by' argument when result is a DataFrame:
  When using groupby + apply returning pd.Series({{}}) → result is a DataFrame → use:
  .sort_values('COLUMN_NAME', ascending=False).head(1).reset_index()

  When using groupby + apply returning a scalar (single number) → result is a Series → use:
  .sort_values(ascending=False).head(1).reset_index()

  ✅ CORRECT — single metric (Series result):
  df[...].groupby('I_IMD_NAME').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).sort_values(ascending=False).head(1).reset_index().rename(columns={{0: 'OD_LR_%'}})

  ✅ CORRECT — multiple metrics (DataFrame result):
  df[...].groupby('I_IMD_NAME').apply(lambda x: pd.Series({{'OD_LR_%': round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2), 'OD_PREMIUM': round(x['OD_NET_EARNED_PREMIUM'].sum(), 2)}})).sort_values('OD_LR_%', ascending=False).head(1).reset_index()

  ❌ WRONG — sort_values() without 'by' on DataFrame:
  df[...].groupby(...).apply(lambda x: pd.Series({{...}})).sort_values(ascending=False)

    RULE 12 — VEHICLE_AGE_GROUP filter:
    The column VEHICLE_AGE_GROUP contains text values like '> 10 YEARS', 'MORE THAN 10 YEARS' etc.
    Since final data may have different formats, always use .str.contains() for vehicle age filters:
    ✅ df[df['VEHICLE_AGE_GROUP'].str.upper().str.contains('10')]
    ✅ df[df['VEHICLE_AGE_GROUP'].str.upper().str.contains('MORE THAN 10|> 10|>10')]
    ❌ df[df['VEHICLE_AGE_GROUP'] == 'MORE THAN 10 YEARS']  ← breaks if value format changes


    RULE 13 — For "districts/entities WHERE metric > average/threshold" questions:
  NEVER use .query() with dynamic string concatenation — it always breaks.
  Instead use a TWO-STATEMENT approach with semicolon:
  
  Statement 1 → calculate the threshold value (average loss ratio)
  Statement 2 → filter districts above that threshold

  ✅ CORRECT PATTERN:
  df[(df['FY_YEAR'] == 'FY-2024-25') & (df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA')].groupby('CUSTOMER_DISTRICT').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).reset_index().rename(columns={{0: 'OD_LR_%'}}).pipe(lambda t: t[t['OD_LR_%'] > round(df[(df['FY_YEAR'] == 'FY-2024-25') & (df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA')]['OD_CLAIM_AMOUNT'].sum() / df[(df['FY_YEAR'] == 'FY-2024-25') & (df['CUSTOMER_STATE'].str.upper() == 'MAHARASHTRA')]['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)]).sort_values('OD_LR_%', ascending=False)

  EXPLANATION OF PATTERN:
  Step 1 → compute district loss ratios → reset_index → rename column 0
  Step 2 → use .pipe(lambda t: t[t['OD_LR_%'] > THRESHOLD]) to filter inline
  Step 3 → THRESHOLD = state average = direct sum calculation on filtered df

  ❌ WRONG — .query() with string + str():
  .query('OD_LR_% > ' + str(round(...)))

  ❌ WRONG — .query() with f-string:
  .query(f'OD_LR_% > {{threshold}}')

  RULE 14 — Always name output columns descriptively using rename():
  When result is a Series (single metric), always rename column 0 to the correct metric name.

  ✅ For OD Premium query:
  .reset_index().rename(columns={{0: 'OD_PREMIUM'}})

  ✅ For OD Loss Ratio query:
  .reset_index().rename(columns={{0: 'OD_LR_%'}})

  ✅ For TP Claim Amount query:
  .reset_index().rename(columns={{0: 'TP_CLAIM_AMOUNT'}})

  ✅ For OD NOP query:
  .reset_index().rename(columns={{0: 'OD_NOP'}})

  ✅ For OD Frequency query:
  .reset_index().rename(columns={{0: 'OD_FREQ_%'}})

  RULE: The column name must match what the user asked for — never leave it as 0.

  RULE 15 — For "IMDs/districts WHERE metric > state average" questions:
  Use SINGLE statement with .pipe() to filter inline — no semicolons needed.
  
  ✅ CORRECT PATTERN (single statement, filter using pipe):
  df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH'].groupby('I_IMD_NAME').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).reset_index().rename(columns={{0: 'OD_LR_%'}}).pipe(lambda t: t[t['OD_LR_%'] > round(df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH']['OD_CLAIM_AMOUNT'].sum() / df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH']['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)]).sort_values('OD_LR_%', ascending=False).head(5)

  STEP BY STEP BREAKDOWN:
  Step 1 → filter state: df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH']
  Step 2 → group by IMD and calculate loss ratio
  Step 3 → reset_index and rename column 0 to 'OD_LR_%'
  Step 4 → .pipe(lambda t: t[t['OD_LR_%'] > STATE_LR]) to filter above state average
  Step 5 → STATE_LR = direct calculation on same state filter
  Step 6 → .sort_values('OD_LR_%', ascending=False).head(5)

  ❌ WRONG — multi-statement with variable assignment:
  state_lr = round(df[...]['OD_CLAIM_AMOUNT'].sum() / df[...]['OD_NET_EARNED_PREMIUM'].sum() * 100, 2); df[...].groupby(...)...

  ❌ WRONG — .query() with string concatenation:
  .query('OD_LR_% > ' + str(state_lr))
  
  ❌ WRONG — any code with = assignment inside eval:
  state_lr = ...  ← assignment breaks eval()

RULE 16 — NEVER use variable assignment inside pandas_code:
  eval() cannot execute assignment statements (=).
  Everything must be a single chained expression.
  ✅ Use .pipe() for intermediate filtering
  ✅ Use nested expressions for threshold values
  ❌ Never: var = value; use_var_here

 RULE 17 — NEVER use % symbol in .assign() column names — Python syntax error:
  .assign() uses keyword arguments — % is illegal in Python variable names.
  
  ❌ WRONG:
  .assign(STATE_OD_LR_%=round(...))   ← % breaks Python syntax

  ✅ CORRECT — use dict form of assign instead:
  .assign(**{{'STATE_OD_LR_%': round(...)}})

  ✅ FULL CORRECT EXAMPLE for IMDs above state loss ratio:
  df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH'].groupby('I_IMD_NAME').apply(lambda x: round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)).reset_index().rename(columns={{0: 'OD_LR_%'}}).assign(**{{'STATE_OD_LR_%': round(df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH']['OD_CLAIM_AMOUNT'].sum() / df[df['CUSTOMER_STATE'].str.upper() == 'UTTAR PRADESH']['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)}}).pipe(lambda t: t[t['OD_LR_%'] > t['STATE_OD_LR_%']]).sort_values('OD_LR_%', ascending=False).head(5)

  KEY PATTERN:
  .assign(**{{'COLUMN_%': value}})   ← double curly braces because inside f-string

  RULE 18 — Always include the groupby dimension column in the output:
  When using groupby + apply returning pd.Series, always call .reset_index()
  at the end so the groupby column (e.g. V_VEHICLE_TYPE) appears as a column.

  ✅ CORRECT — vehicle type shows as first column:
  df.groupby('V_VEHICLE_TYPE').apply(lambda x: pd.Series({{'OD_LR_%': round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2), 'TP_LR_%': round(x['TP_CLAIM_AMOUNT'].sum() / x['TP_NET_EARNED_PREMIUM'].sum() * 100, 2)}})).reset_index()

  ❌ WRONG — groupby column lost:
  df.groupby('V_VEHICLE_TYPE').apply(lambda x: pd.Series({{...}}))  ← missing .reset_index()

  RULE 19 — For "IMDs where loss ratio > their state average" (all states):
  Use a SINGLE statement with merge + filter — no semicolons needed:

  ✅ CORRECT PATTERN:
  df.groupby(['CUSTOMER_STATE', 'I_IMD_NAME']).apply(lambda x: pd.Series({{'IMD_OD_LR_%': round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)}})).reset_index().merge(df.groupby('CUSTOMER_STATE').apply(lambda x: pd.Series({{'STATE_OD_LR_%': round(x['OD_CLAIM_AMOUNT'].sum() / x['OD_NET_EARNED_PREMIUM'].sum() * 100, 2)}})).reset_index(), on='CUSTOMER_STATE').pipe(lambda t: t[t['IMD_OD_LR_%'] > t['STATE_OD_LR_%']]).sort_values('IMD_OD_LR_%', ascending=False).head(10)

  This returns a single DataFrame with columns:
  CUSTOMER_STATE | I_IMD_NAME | IMD_OD_LR_% | STATE_OD_LR_%
  No need for two-step — single merge handles it.

"""
    return prompt

# ══════════════════════════════════════════════════════
# 4. JSON PARSER — handles Groq multiline/backslash responses
# ══════════════════════════════════════════════════════
def parse_groq_response(raw):
    raw = raw.replace("```json", "").replace("```python", "").replace("```", "").strip()
    raw = re.sub(r'\\\s*\n\s*', ' ', raw)   # collapse backslash line continuations
    raw = re.sub(r'(?<!\\)\n', ' ', raw)    # remove stray newlines inside JSON
    raw = re.sub(r'  +', ' ', raw).strip()  # collapse multiple spaces

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract with regex
        code_match = re.search(r'"pandas_code"\s*:\s*"(.*?)(?<!\\)",\s*"explanation"', raw, re.DOTALL)
        expl_match = re.search(r'"explanation"\s*:\s*"(.*?)"[\s\n]*\}', raw, re.DOTALL)
        if code_match and expl_match:
            return {
                "pandas_code": code_match.group(1).replace('\\"', '"'),
                "explanation":  expl_match.group(1)
            }
        raise ValueError(f"Could not parse Groq response:\n{raw}")

# ══════════════════════════════════════════════════════
# 5. RUN CODE — safe eval with all builtins needed
# ══════════════════════════════════════════════════════
def run_code(code, df):
    import builtins

    # Use full builtins (safe for internal/local app usage)
    # Only block genuinely dangerous functions
    safe_builtins = vars(builtins).copy()
    for dangerous in ["exec", "eval", "compile", "open",
                      "breakpoint", "memoryview", "__loader__"]:
        safe_builtins.pop(dangerous, None)

    eval_globals = {"df": df, "pd": pd, "__builtins__": safe_builtins}

    def normalize(res, label=None):
        """Convert any eval result into a clean flat DataFrame."""
        if isinstance(res, (int, float, str)):
            return pd.DataFrame({"Result": [res]})

        if isinstance(res, pd.Series):
            out = res.reset_index()
            out.columns = [str(c) for c in out.columns]
            # Only rename "0" column — keep name neutral, format_df will handle display
            out.columns = ["VALUE" if c == "0" else c for c in out.columns]
            # If last column looks like a loss ratio (values 0-300), label it
            last_col = out.columns[-1]
            if last_col == "OD_LR_%":
                pass  # already named
            elif pd.to_numeric(out[last_col], errors="coerce").between(0, 300).all():
                out.rename(columns={last_col: "OD_LR_%"}, inplace=True)
            if label and "FY_YEAR" not in out.columns:
                out.insert(0, "FY_YEAR", label)
            return out
        # ── Handle scalar ──
        if isinstance(res, (int, float)):
            # Try to make it a readable result
            label_name = "Result"
            return pd.DataFrame({label_name: [res]})

        if isinstance(res, str):
            return pd.DataFrame({"Result": [res]})

        if isinstance(res, pd.DataFrame):
            out = res.copy()
            # Flatten MultiIndex columns (from unstack)
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = [
                    str(col[-1]) if isinstance(col, tuple) else str(col)
                    for col in out.columns
                ]
            # Keep named index as column (groupby dimension)
            if out.index.name and out.index.name not in out.columns:
                out = out.reset_index()   # ← keeps groupby column
            else:
                out = out.reset_index(drop=True)  # ← drops numeric index only
            out.columns = [str(c) for c in out.columns]
            out.columns = ["LOSS_RATIO_%" if c == "0" else c for c in out.columns]
            # Fix _x/_y merge artifacts
            rename_map = {}
            for col in out.columns:
                if col.endswith("_x"):
                    rename_map[col] = col.replace("_x", "_FY23-24_%")
                elif col.endswith("_y"):
                    rename_map[col] = col.replace("_y", "_FY24-25_%")
            if rename_map:
                out.rename(columns=rename_map, inplace=True)
            if label and "FY_YEAR" not in out.columns:
                out.insert(0, "FY_YEAR", label)
            return out

        return pd.DataFrame({"Result": [str(res)]})

    def auto_pivot(data):
        """If FY_YEAR column with 2+ years exists, pivot to wide format."""
        if not isinstance(data, pd.DataFrame):
            return data
        if "FY_YEAR" not in data.columns or data["FY_YEAR"].nunique() < 2:
            return data
        lr_col = next((c for c in data.columns if "LOSS_RATIO" in c.upper() or c == "0"), None)
        grp_col = next((c for c in data.columns if c not in ["FY_YEAR", lr_col] and data[c].dtype == object), None)
        if not lr_col or not grp_col:
            return data
        try:
            pivot = data.pivot_table(
                index=grp_col, columns="FY_YEAR", values=lr_col, aggfunc="first"
            ).reset_index()
            pivot.columns.name = None
            rename = {}
            for col in pivot.columns:
                if "2023" in str(col):
                    rename[col] = "OD_LR_FY23-24_%"
                elif "2024" in str(col):
                    rename[col] = "OD_LR_FY24-25_%"
            pivot.rename(columns=rename, inplace=True)
            return pivot
        except Exception:
            return data

    statements = [s.strip() for s in code.split(";") if s.strip()]

    if len(statements) == 1:
        res = eval(statements[0], eval_globals)
        return auto_pivot(normalize(res))

    else:
        results = []
        for stmt in statements:
            label = None
            if "FY-2024-25" in stmt:
                label = "FY-2024-25"
            elif "FY-2023-24" in stmt:
                label = "FY-2023-24"
            try:
                res = eval(stmt, eval_globals)
                results.append({"label": label, "data": auto_pivot(normalize(res, label))})
            except Exception as e:
                results.append({"label": label or "Error", "data": pd.DataFrame({"Error": [str(e)]})})
        return results

# ══════════════════════════════════════════════════════
# 6. FORMAT DATAFRAME — works with any column names
# ══════════════════════════════════════════════════════
def format_df(df_in):
    out = df_in.copy()
    out.columns = [str(c) for c in out.columns]
    for col in list(out.columns):
        col_up = col.upper()

        # Format as % ONLY if column name explicitly contains LR or LOSS_RATIO
        # NOT for generic VALUE column
        if any(k in col_up for k in ["LOSS_RATIO", "_LR_%", "LR_%", "_LR",
                                      "FREQUENCY", "FREQ", "STATE_OD", "STATE_TP"]) \
                and col_up != "VALUE":
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
            )

        # Format as Crores if column name has PREMIUM or CLAIM_AMOUNT
        elif any(k in col_up for k in ["PREMIUM", "CLAIM_AMOUNT"]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].apply(
                lambda x: f"₹{x/1e7:.2f} Cr" if pd.notnull(x) else "-"
            )

        # Format counts with commas
        elif any(k in col_up for k in ["_NOP", "_NOC"]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].apply(
                lambda x: f"{int(x):,}" if pd.notnull(x) else "-"
            )

        # VALUE column — detect type from actual data values
        elif col_up == "VALUE":
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.max() > 10000:
                # Large number → treat as amount → show as Crores
                out[col] = numeric.apply(
                    lambda x: f"₹{x/1e7:.2f} Cr" if pd.notnull(x) else "-"
                )
            elif numeric.max() <= 300:
                # Small number → treat as ratio/% → show as %
                out[col] = numeric.apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
                )
    # Catch-all: any remaining numeric column with values between 0-500
        # that hasn't been formatted yet → likely a ratio/% column
        elif col_up not in ["FY_YEAR", "VALUE"] and \
             not any(k in col_up for k in ["STATE", "DISTRICT", "IMD", "CHANNEL",
                                            "VEHICLE", "FUEL", "ZONE", "PIN",
                                            "SEGMENT", "CATEGORY", "TYPE", "LOB",
                                            "FLAG", "NAME", "CODE", "GROUP"]):
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().any():
                col_max = numeric.max()
                col_min = numeric.min()
                if 0 <= col_min and col_max <= 500:
                    # Likely a percentage/ratio
                    out[col] = numeric.apply(
                        lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
                    )
                elif col_max > 10000:
                    # Likely an amount
                    out[col] = numeric.apply(
                        lambda x: f"₹{x/1e7:.2f} Cr" if pd.notnull(x) else "-"
                    )
    return out

# ══════════════════════════════════════════════════════
# 7. SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.title("📊 Insurance Bot")
    st.markdown("---")
    if data_loaded:
        fy_vals = sorted(df['FY_YEAR'].dropna().unique().tolist())
        st.success(f"✅ {len(df):,} rows loaded")
        st.write(f"**FY Years:** {fy_vals}")
        st.write(f"**States:** {df['CUSTOMER_STATE'].nunique()}")
        st.write(f"**Channels:** {df['IMD_CHANNEL'].nunique()}")
        st.write(f"**Vehicle Types:** {df['V_VEHICLE_TYPE'].nunique()}")
    st.markdown("---")
    st.markdown("**📈 Quick Charts**")
    if st.button("OD Loss Ratio by State"):      st.session_state["quick"] = "od_lr_state"
    if st.button("TP Loss Ratio by State"):      st.session_state["quick"] = "tp_lr_state"
    if st.button("OD Premium by Channel"):       st.session_state["quick"] = "od_prem_channel"
    if st.button("OD Claims by Vehicle Type"):   st.session_state["quick"] = "od_claim_vehicle"
    if st.button("NEW vs RENEWAL"):              st.session_state["quick"] = "new_renewal"
    if st.button("Claims by Risk Zone"):         st.session_state["quick"] = "riskzone"
    if st.button("Claims by Fuel Type"):         st.session_state["quick"] = "fuel"
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════
# 8. MAIN AREA
# ══════════════════════════════════════════════════════
st.title("📊 Insurance Analytics Chatbot")

if data_loaded:

    # ── KPI Row ──
    od_claim   = df["OD_CLAIM_AMOUNT"].sum()
    od_premium = df["OD_NET_EARNED_PREMIUM"].sum()
    tp_claim   = df["TP_CLAIM_AMOUNT"].sum()
    tp_premium = df["TP_NET_EARNED_PREMIUM"].sum()
    od_lr      = (od_claim / od_premium * 100)   if od_premium > 0 else 0
    tp_lr      = (tp_claim / tp_premium * 100)   if tp_premium > 0 else 0
    od_freq    = (df["OD_NOC"].sum() / df["OD_NOP"].sum() * 100) if df["OD_NOP"].sum() > 0 else 0
    tp_freq    = (df["TP_NOC"].sum() / df["TP_NOP"].sum() * 100) if df["TP_NOP"].sum() > 0 else 0

    k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
    k1.metric("OD Premium",     f"₹{od_premium/1e7:.1f}Cr")
    k2.metric("OD Claim Amt",   f"₹{od_claim/1e7:.1f}Cr")
    k3.metric("OD Loss Ratio",  f"{od_lr:.1f}%",  delta="⚠️ High" if od_lr  > 100 else "✅ OK")
    k4.metric("OD Frequency",   f"{od_freq:.2f}%")
    k5.metric("TP Premium",     f"₹{tp_premium/1e7:.1f}Cr")
    k6.metric("TP Claim Amt",   f"₹{tp_claim/1e7:.1f}Cr")
    k7.metric("TP Loss Ratio",  f"{tp_lr:.1f}%",  delta="⚠️ High" if tp_lr  > 100 else "✅ OK")
    k8.metric("TP Frequency",   f"{tp_freq:.2f}%")

    st.markdown("---")

    # ── Quick Charts ──
    qc = st.session_state.get("quick", None)
    if qc:
        st.subheader("📈 Quick Chart")
        try:
            if qc == "od_lr_state":
                g = df.groupby("CUSTOMER_STATE").apply(
                    lambda x: x["OD_CLAIM_AMOUNT"].sum() / x["OD_NET_EARNED_PREMIUM"].sum() * 100
                ).sort_values(ascending=False).head(15).reset_index()
                g.columns = ["STATE", "OD_LR_%"]
                fig = px.bar(g, x="STATE", y="OD_LR_%",
                             title="OD Loss Ratio by State (Top 15)",
                             color="OD_LR_%", color_continuous_scale="Reds",
                             template="plotly_dark")
                fig.add_hline(y=100, line_dash="dash", line_color="white", annotation_text="100% Break-even")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "tp_lr_state":
                g = df.groupby("CUSTOMER_STATE").apply(
                    lambda x: x["TP_CLAIM_AMOUNT"].sum() / x["TP_NET_EARNED_PREMIUM"].sum() * 100
                ).sort_values(ascending=False).head(15).reset_index()
                g.columns = ["STATE", "TP_LR_%"]
                fig = px.bar(g, x="STATE", y="TP_LR_%",
                             title="TP Loss Ratio by State (Top 15)",
                             color="TP_LR_%", color_continuous_scale="Oranges",
                             template="plotly_dark")
                fig.add_hline(y=100, line_dash="dash", line_color="white")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "od_prem_channel":
                g = df.groupby("IMD_CHANNEL")[["OD_NET_EARNED_PREMIUM","TP_NET_EARNED_PREMIUM"]].sum().reset_index()
                fig = px.bar(g, x="IMD_CHANNEL", y=["OD_NET_EARNED_PREMIUM","TP_NET_EARNED_PREMIUM"],
                             title="Premium by Channel (OD vs TP)",
                             template="plotly_dark", barmode="group")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "od_claim_vehicle":
                g = df.groupby("V_VEHICLE_TYPE")[["OD_CLAIM_AMOUNT","TP_CLAIM_AMOUNT"]].sum().reset_index()
                fig = px.bar(g, x="V_VEHICLE_TYPE", y=["OD_CLAIM_AMOUNT","TP_CLAIM_AMOUNT"],
                             title="Claims by Vehicle Type (OD vs TP)",
                             template="plotly_dark", barmode="group")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "new_renewal":
                g = df.groupby("REN_ROLL_NB_FLAG").apply(
                    lambda x: pd.Series({
                        "OD_LR_%":   round(x["OD_CLAIM_AMOUNT"].sum() / x["OD_NET_EARNED_PREMIUM"].sum() * 100, 2),
                        "OD_NOP":    int(x["OD_NOP"].sum()),
                        "OD_PREMIUM":round(x["OD_NET_EARNED_PREMIUM"].sum(), 2)
                    })
                ).reset_index()
                st.dataframe(g, use_container_width=True)
                fig = px.bar(g, x="REN_ROLL_NB_FLAG", y="OD_LR_%",
                             title="OD Loss Ratio: NEW vs RENEWAL",
                             color="REN_ROLL_NB_FLAG", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "riskzone":
                g = df.groupby("RISKZONE").apply(
                    lambda x: round(x["OD_CLAIM_AMOUNT"].sum() / x["OD_NET_EARNED_PREMIUM"].sum() * 100, 2)
                ).reset_index()
                g.columns = ["RISKZONE", "OD_LR_%"]
                fig = px.bar(g, x="RISKZONE", y="OD_LR_%",
                             title="OD Loss Ratio by Risk Zone",
                             color="OD_LR_%", color_continuous_scale="RdYlGn_r",
                             template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            elif qc == "fuel":
                g = df.groupby("FUEL_TYPE").apply(
                    lambda x: pd.Series({
                        "OD_CLAIM_AMOUNT": round(x["OD_CLAIM_AMOUNT"].sum(), 2),
                        "OD_FREQ_%":       round(x["OD_NOC"].sum() / x["OD_NOP"].sum() * 100, 2)
                    })
                ).reset_index()
                fig = px.bar(g, x="FUEL_TYPE", y="OD_CLAIM_AMOUNT",
                             title="OD Claims by Fuel Type",
                             color="OD_FREQ_%", color_continuous_scale="Blues",
                             template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Chart error: {e}")

        if st.button("✖ Close Chart"):
            st.session_state.pop("quick", None)
            st.rerun()

        st.markdown("---")

    # ── Data Preview ──
    # with st.expander("👁️ Preview Raw Data (first 5 rows)"):
    #     st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    #     st.write(f"**Columns:** {list(df.columns)}")
    #     st.dataframe(df.head(5), use_container_width=True)

    # ── Suggested Questions ──
    st.markdown("**💡 Suggested Questions — click to ask:**")
    suggestions = [
        "Top 5 states by OD loss ratio in FY-2024-25",
        "Which channel has highest OD loss ratio in FY-2024-25?",
        "Compare NEW vs RENEWAL OD loss ratio",
        "Year on year OD loss ratio by state",
        "Top 5 districts by OD claim amount in FY-2024-25",
        "OD and TP loss ratio by vehicle type",
        "Which risk zone has highest OD loss ratio?",
        "OD frequency by fuel type",
    ]
    r1 = st.columns(4)
    r2 = st.columns(4)
    for i, s in enumerate(suggestions[:4]):
        if r1[i].button(s, key=f"sq{i}"):
            st.session_state["prefill"] = s
    for i, s in enumerate(suggestions[4:]):
        if r2[i].button(s, key=f"sq{i+4}"):
            st.session_state["prefill"] = s

    st.markdown("---")

    # ── Chat History ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)
            # if "code" in msg:
            #     st.caption(f"🔧 `{msg['code']}`")

    # ── Chat Input ──
    prefill  = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask anything about your insurance data…")
    if prefill:
        question = prefill

    if question and groq_ready:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Analysing your data..."):
                pandas_code = ""
                raw         = ""
                try:
                    system_prompt = build_system_prompt(df)
                    MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
                    resp = None
                    last_error = None
                    for model_name in MODELS:
                        try:
                            resp = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": question}
                                ],
                                temperature=0,
                                max_tokens=600
                            )
                            break
                        except Exception as e:
                            last_error = e
                            if "rate_limit" in str(e).lower() or "429" in str(e):
                                continue
                            else:
                                raise e
                    if resp is None:
                        raise Exception(f"All models rate limited. Try again in 10 min.")
                    raw         = resp.choices[0].message.content.strip()
                    parsed      = parse_groq_response(raw)
                    pandas_code = parsed.get("pandas_code", "")
                    explanation = parsed.get("explanation", "")
                    result      = run_code(pandas_code, df)

                    st.markdown(f"**{explanation}**")
                    # st.caption(f"🔧 `{pandas_code}`")

                    msg_entry = {"role": "assistant", "content": explanation, "code": pandas_code}

                    # ── CASE 1: Multi-result list (two-step queries) ──
                    if isinstance(result, list):
                        for i, item in enumerate(result):
                            data  = item["data"]
                            label = item.get("label", f"Result {i+1}")

                            if i == 0:
                                st.markdown("---")
                                # Check if first result has multiple rows → show as table not metric card
                                if isinstance(data, pd.DataFrame) and len(data) > 1:
                                    # Multiple rows — show as table directly
                                    st.markdown(f"#### 📊 {label} Summary")
                                    st.dataframe(format_df(data), use_container_width=True)
                                    msg_entry["dataframe"] = format_df(data)
                                else:
                                    # Single row — show as metric card
                                    try:
                                        dim_col = next((c for c in data.columns if data[c].dtype == object and c != "FY_YEAR"), None)
                                        lr_col  = next((c for c in data.columns if "LR" in c.upper() or "LOSS" in c.upper()), None)
                                        entity  = data.iloc[0][dim_col] if dim_col else "Top Entry"
                                        lr_val  = pd.to_numeric(data.iloc[0][lr_col], errors="coerce") if lr_col else None
                                        st.markdown(f"### 🏆 Highest Loss Ratio — {label}")
                                        m1, m2 = st.columns(2)
                                        m1.metric(dim_col or "Entity", entity)
                                        if lr_val:
                                            m2.metric("OD Loss Ratio", f"{lr_val:.2f}%",
                                                      delta="⚠️ Highest" if lr_val > 100 else "📊 Tracked")
                                    except Exception:
                                        st.dataframe(format_df(data), use_container_width=True)
                                st.markdown("---")

                            else:
                                # Second result = drill-down table + chart
                                st.markdown("#### 📍 Detailed Breakdown")
                                formatted = format_df(data)
                                st.dataframe(formatted, use_container_width=True)
                                msg_entry["dataframe"] = formatted

                                try:
                                    raw_num = data.select_dtypes("number")
                                    cat_col = next((c for c in data.columns if data[c].dtype == object), None)
                                    if cat_col and not raw_num.empty:
                                        chart_data = data.copy()
                                        num_cols   = raw_num.columns.tolist()
                                        if len(num_cols) >= 2:
                                            melted = chart_data.melt(
                                                id_vars=cat_col,
                                                value_vars=num_cols,
                                                var_name="FY_YEAR",
                                                value_name="OD_LR_%"
                                            )
                                            fig = px.bar(melted, x=cat_col, y="OD_LR_%",
                                                         color="FY_YEAR", barmode="group",
                                                         title="Year-on-Year Breakdown",
                                                         template="plotly_dark",
                                                         color_discrete_sequence=["#3b82f6","#ef4444"])
                                        else:
                                            fig = px.bar(chart_data, x=cat_col, y=num_cols[0],
                                                         title="Breakdown", template="plotly_dark",
                                                         color=num_cols[0], color_continuous_scale="Reds")
                                        st.plotly_chart(fig, use_container_width=True)
                                        msg_entry["chart"] = fig
                                except Exception:
                                    pass
                        st.session_state.messages.append(msg_entry)  # ← save CASE 1

                    # ── CASE 2: Single DataFrame ──
                    elif isinstance(result, pd.DataFrame):
                        formatted = format_df(result)
                        st.dataframe(formatted, use_container_width=True)
                        msg_entry["dataframe"] = formatted

                        try:
                            num_cols = result.select_dtypes("number").columns.tolist()
                            cat_cols = result.select_dtypes("object").columns.tolist()
                            if cat_cols and num_cols:
                                if len(num_cols) >= 2:
                                    melted = result.melt(
                                        id_vars=cat_cols[0],
                                        value_vars=num_cols,
                                        var_name="Metric",
                                        value_name="Value"
                                    )
                                    fig = px.bar(melted, x=cat_cols[0], y="Value",
                                                 color="Metric", barmode="group",
                                                 title=question, template="plotly_dark",
                                                 color_discrete_sequence=["#3b82f6","#ef4444","#22c55e"])
                                else:
                                    fig = px.bar(result, x=cat_cols[0], y=num_cols[0],
                                                 title=question, template="plotly_dark",
                                                 color=num_cols[0], color_continuous_scale="Reds")
                                st.plotly_chart(fig, use_container_width=True)
                                msg_entry["chart"] = fig
                        except Exception:
                            pass
                        st.session_state.messages.append(msg_entry)  # ← save CASE 2

                    # ── CASE 3: Scalar ──
                    else:
                        # Format nicely based on likely type
                        val = result
                        if isinstance(val, pd.DataFrame) and len(val) == 1 and len(val.columns) == 1:
                            val = val.iloc[0, 0]

                        # Detect if it's a loss ratio (0-200 range)
                        try:
                            num = float(val) if not isinstance(val, pd.DataFrame) else None
                            if num is not None:
                                if 0 < num < 200:
                                    display = f"**{num:.2f}%**"
                                    label   = "OD Loss Ratio"
                                elif num > 10000:
                                    display = f"**₹{num/1e7:.2f} Cr**"
                                    label   = "Amount"
                                else:
                                    display = f"**{num:,.2f}**"
                                    label   = "Result"
                                st.metric(label, display)
                            else:
                                st.success(f"**Result:** {val}")
                        except Exception:
                            st.success(f"**Result:** {val}")

                        msg_entry["content"] = f"{explanation}\n\n**Result: {val}**"
                        st.session_state.messages.append(msg_entry)

                except json.JSONDecodeError:
                    st.error("⚠️ AI returned unexpected format. Try rephrasing your question.")
                    st.code(raw if raw else "No response captured")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    if pandas_code:
                        st.caption(f"Code attempted: `{pandas_code}`")

else:
    st.error("❌ Data not loaded. Place your file at: insurance_chatbot/data/insurance_data.csv")
