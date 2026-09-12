




  


import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


st.set_page_config(
    page_title="Student Withdrawal Prediction System",
    page_icon="🔍",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF5FA 100%);
    }
    h1, h2, h3 {
        color: #C2185B !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FDF2F8;
        border-right: 1px solid #FBCFE8;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #FBCFE8;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(236, 72, 153, 0.08);
    }
    .result-card {
        border-radius: 18px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(236, 72, 153, 0.15);
        margin-bottom: 18px;
    }
    .risk-high {
        background: linear-gradient(135deg, #FBCFE8 0%, #F9A8D4 100%);
        border: 2px solid #EC4899;
    }
    .risk-low {
        background: linear-gradient(135deg, #FCE7F3 0%, #FFFFFF 100%);
        border: 2px solid #F9A8D4;
    }
    .stButton>button {
        background-color: #EC4899;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 26px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #DB2777;
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


DATA_FILE = "master_df.csv"


HIGHEST_EDUCATION_MAP = {
    0: "A Level or Equivalent",
    1: "HE Qualification",
    2: "Lower Than A Level",
    3: "No Formal quals",
    4: "Post Graduate Qualification",
}
IMD_BAND_MAP = {
    0: "0-10%",
    1: "10-20%",
    2: "20-30%",
    3: "30-40%",
    4: "40-50%",
    5: "50-60%",
    6: "60-70%",
    7: "70-80%",
    8: "80-90%",
    9: "90-100%",
    10: "Unknown",
}

HIGHEST_EDUCATION_REVERSE = {v: k for k, v in HIGHEST_EDUCATION_MAP.items()}
IMD_BAND_REVERSE = {v: k for k, v in IMD_BAND_MAP.items()}

FEATURES = [
    "mean_assessment_score",
    "total_vle_clicks",
    "date_registration",
    "imd_band",
    "highest_education",
]



@st.cache_data(show_spinner="Loading data...")
def load_master_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df



@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(master_df: pd.DataFrame):
    df = master_df[FEATURES + ["final_result"]].copy()
    df["target_withdrawn"] = (df["final_result"] == "Withdrawn").astype(int)

    X = df[FEATURES]
    y = df["target_withdrawn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", max_depth=12
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    importance_df = pd.DataFrame(
        {"feature": FEATURES, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    return rf, acc, f1, importance_df



if not os.path.exists(DATA_FILE):
    st.title("🔍 Student Withdrawal Prediction System")
    st.error(
        f'⚠️ Could not find "{DATA_FILE}" next to app.py.\n\n'
        "Developer note: place your master_df.csv file in the same folder as "
        "app.py, or update the DATA_FILE variable at the top of the script."
    )
    st.stop()

master_df = load_master_df(DATA_FILE)
rf_model, model_acc, model_f1, importance_df = train_model(master_df)


st.sidebar.title("Control Panel")
st.sidebar.markdown("### 🎛️ Student Filters")

score_min = float(master_df["mean_assessment_score"].min())
score_max = float(master_df["mean_assessment_score"].max())
clicks_min = float(master_df["total_vle_clicks"].min())
clicks_max = float(master_df["total_vle_clicks"].max())
reg_min = int(master_df["date_registration"].min())
reg_max = int(master_df["date_registration"].max())

mean_assessment_score = st.sidebar.slider(
    "📊 Mean assessment score",
    min_value=score_min,
    max_value=score_max,
    value=float(master_df["mean_assessment_score"].median()),
)

total_vle_clicks = st.sidebar.slider(
    "🖱️ Total VLE clicks",
    min_value=clicks_min,
    max_value=clicks_max,
    value=float(master_df["total_vle_clicks"].median()),
)

date_registration = st.sidebar.slider(
    "📅 Registration date (in days, relative to course start)",
    min_value=reg_min,
    max_value=reg_max,
    value=int(master_df["date_registration"].median()),
    help="A negative value means the student registered that many days before the course started.",
)

imd_label = st.sidebar.selectbox(
    "🏘️ IMD band (socio-economic deprivation index)",
    list(IMD_BAND_REVERSE.keys()),
)

education_label = st.sidebar.selectbox(
    "🎓 Highest education",
    list(HIGHEST_EDUCATION_REVERSE.keys()),
)

predict_clicked = st.sidebar.button("🔮 Predict now", use_container_width=True)


st.title("Student Withdrawal Prediction System")
st.caption("Random Forest model trained on the OULAD dataset (master_df)")

col_a, col_b = st.columns(2)
col_a.metric("Model Accuracy", f"{model_acc * 100:.1f}%")
col_b.metric("F1-Score", f"{model_f1 * 100:.1f}%")

st.markdown("---")

if predict_clicked:
    imd_code = IMD_BAND_REVERSE[imd_label]
    education_code = HIGHEST_EDUCATION_REVERSE[education_label]

    input_df = pd.DataFrame(
        [
            {
                "mean_assessment_score": mean_assessment_score,
                "total_vle_clicks": total_vle_clicks,
                "date_registration": date_registration,
                "imd_band": imd_code,
                "highest_education": education_code,
            }
        ]
    )[FEATURES]

    proba = rf_model.predict_proba(input_df)[0]
    withdraw_prob = proba[1] * 100
    not_withdraw_prob = proba[0] * 100
    will_withdraw = withdraw_prob >= 50

    result_col, chart_col = st.columns([1, 1.2])

    with result_col:
        if will_withdraw:
            st.markdown(
                f"""
                <div class="result-card risk-high">
                    <h2>⚠️ Student is at risk of withdrawing</h2>
                    <h1 style="color:#BE185D; font-size:56px;">{withdraw_prob:.1f}%</h1>
                    <p style="color:#831843;">Probability of withdrawal</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "🔔 Recommended: reach out early and offer academic or motivational "
                "support based on the weakest factor in the student's profile."
            )
        else:
            st.markdown(
                f"""
                <div class="result-card risk-low">
                    <h2>✅ Student is on track (low withdrawal risk)</h2>
                    <h1 style="color:#DB2777; font-size:56px;">{not_withdraw_prob:.1f}%</h1>
                    <p style="color:#9D174D;">Probability of continuing</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("🔍 Keep monitoring engagement periodically to maintain this level.")

    with chart_col:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Withdrawal probability", "Continuation probability"],
                    values=[withdraw_prob, not_withdraw_prob],
                    hole=0.55,
                    marker=dict(colors=["#EC4899", "#FBCFE8"]),
                    textinfo="label+percent",
                    textfont=dict(size=14),
                )
            ]
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"{withdraw_prob:.0f}%",
                    x=0.5,
                    y=0.5,
                    font_size=28,
                    font_color="#BE185D",
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Top factors driving the model's decision")
    fig_imp = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(color="#EC4899"),
        )
    )
    fig_imp.update_layout(
        xaxis_title="Importance score",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("⬅️ Set the student's data using the filters in the sidebar, then click «Predict now».")
