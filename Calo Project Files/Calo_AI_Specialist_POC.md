# Calo AI Specialist POC - Subscription Growth Agent

## Executive Summary
- Reduce churn risk visibility from hours to seconds.
- Generate tailored Arabic or UK English offers with a single click.
- Keep outputs automation ready using a JSON block while showing a clean human plan.
- Model choice stays flexible via Hugging Face Inference Router.

# Architecture and Design

- Frontend - Streamlit single file app with three tabs: Data, Scoring, Recommendations.
- Data - CSV upload or synthetic generator with realistic distributions for last_order_days, orders_month, lifetime_months.
- Risk Model - Simple interpretable score combined and normalised to 0-100.
- LLM Layer - Hugging Face Inference Router with OpenAI compatible client. Default model gpt-oss-120b from Fireworks.
- Output Contract - Human plan text plus machine JSON. UI renders plan with bold emphasis while JSON stays internal.
- i18n - Arabic RTL and UK English LTR with CSS toggles and language switcher.


# Product Requirements Document - Subscription Growth Agent

## Problem
Meal subscriptions face churn due to waning engagement. Teams need a fast way to spot at-risk users and push the right offer at the right time without manual analysis.

## Goals
- Identify top at-risk subscribers in seconds.
- Generate human readable plan text plus structured JSON for automation.
- Support Arabic Gulf tone and UK English tone with a single codebase.
- Keep the model choice abstract through the HF Inference Router.

## Non-goals
- No PII ingestion in this POC.
- No production CRM integration in this POC.

## Users and Personas
- Retention Marketer - wants targeted offers and copy quickly.
- CRM Operator - needs structured payloads for campaigns.
- Growth PM - wants measurable lift and iteration speed.

## User Flow
1. Load or generate sample data.
2. Review data quality tips.
3. Calculate churn and select top N.
4. Generate AI recommendations.
5. Read plan text and inspect per-customer actions.
6. Export or plug into CRM in a next iteration.

## Functional Requirements
- Upload CSV with required schema.
- Create synthetic dataset when needed.
- Compute churn_score in range 0 to 100.
- Select top N customers: 5 or 10 or 15 or 20.
- Generate bilingual plan and JSON block based on language.
- Render bold emphasis for offers and percentages.
- Show per customer expander with message, expected impact, rationale.

## Acceptance Criteria
- With sample data, generation completes under 5 seconds locally.
- Plan text renders with consistent spacing and visible bold.
- JSON is parsed and drives the recommendation UI without exposing raw JSON.
- Switching language updates UI labels and sample data language.

## Metrics
- Time to first recommendation.
- CTR proxy: count of high impact offers.
- Recall of risky customers vs. baseline heuristic.

## Risks
- Model variance - mitigated by fixed prompts and constrained outputs.
- JSON parsing - mitigated by dual extraction of <JSON> or bare array.
- UI RTL LTR styling - mitigated by conditional CSS blocks.

## Next Steps
- Connect to CRM segments and A B tests.
- Add guardrails and observability.
- Introduce configurable weights for churn once validated.


# Prompts

## System - Arabic Gulf
أنت مساعد تسويق يتحدث العربية ومخصص لدول الخليج. اكتب بالعربية برسالة ودية ومهنية.

## User - Arabic Gulf
استخدم اللهجة: {dialect}. 1) اكتب فقرة ودية قصيرة بالعربية لكل عميل تشرح العرض المقترح مع رموز تعبيرية.
2) بعد الفقرات، أضف كتلة <JSON>...</JSON> فيها مصفوفة كائنات بالحقل:
customer_id, action, message, expected_lift, rationale.
القيم بالعربية فقط، وطول message ≤ 120 حرفًا.
العملاء: {records}

## System - UK English
You are a marketing assistant specialised for the UK market. Write in professional yet warm UK English.

## User - UK English
1) For each customer, write a short friendly paragraph in English that explains the retention offer with emojis.
2) Then add a <JSON>...</JSON> block containing an array of objects with keys:
customer_id, action, message, expected_lift, rationale.
Use UK English tone. Message length ≤ 120 characters.
Customers: {records}


## 60s Demo Scripts
### English
60s Demo Script — English

Goal - show end to end value for Calo - reduce churn risk visibility and generate tailored offers in seconds.

0-5s
Voiceover: "Hi, I am [Your Name]. This is a 1 minute AI prototype for Calo focused on reducing churn and lifting repeat orders."
Action: Launch the app. App title visible.

5-12s
Voiceover: "I can switch Arabic or English. I will pick English for the UK market."
Action: Sidebar, choose Language -> English. Theme change optional.

12-18s
Voiceover: "Data can come from CSV or sample data. I will generate a clean sample."
Action: Click Generate sample data. Show table. Open "Data quality tips".

18-28s
Voiceover: "Next, we calculate churn risk."
Action: Open Scoring. "The score blends time since last order, order frequency, lifetime, and promo usage, then normalises 0 to 100."
Action: Select 10 in "Show top at-risk customers".

28-45s
Voiceover: "Now AI generates tailored retention offers."
Action: Go to Recommendations. Set top 5. Click "Generate retention recommendations".
Voiceover: "Under the hood I use Hugging Face Inference Router with an OpenAI compatible client calling GPT-OSS-120B. Data here is synthetic."

45-55s
Voiceover: "We get a clean plan plus structured JSON behind the scenes. Key parts are bold and each recommendation shows expected impact."
Action: Scroll the plan, expand two recommendations.

55-60s
Voiceover: "This can ship as a POC and plug into CRM and campaigns. Thank you."
Action: End on plan view.


### Arabic
نص الديمو 60 ثانية — العربية

الهدف - إظهار قيمة فورية لكالو - رؤية واضحة لخطر الانسحاب وتوليد عروض مخصصة خلال ثوانٍ.

0-5 ث
التعليق الصوتي: "مرحباً، أنا [اسمك]. هذا نموذج ذكاء اصطناعي سريع لمدة دقيقة يركز على خفض الانسحاب وزيادة تكرار الطلب."
الإجراء: تشغيل التطبيق وظهور العنوان.

5-12 ث
التعليق الصوتي: "أبدّل بين العربية والإنجليزية. سأختار الإنجليزية للسوق البريطاني."
الإجراء: من الشريط الجانبي اختيار English. تغيير اللون اختياري.

12-18 ث
التعليق الصوتي: "البيانات من CSV أو بيانات تجريبية. سأولّد عينة نقية."
الإجراء: ضغط "إنشاء بيانات تجريبية". عرض الجدول. فتح "نصائح جودة البيانات".

18-28 ث
التعليق الصوتي: "الآن نحسب درجة الانسحاب."
الإجراء: فتح تبويب الحساب. "الدرجة تمزج تأخر آخر طلب وتكرار الشراء وطول العلاقة واستخدام العروض ثم نطبّعها من 0 إلى 100."
الإجراء: اختيار 10 من "عرض الأعلى خطراً".

28-45 ث
التعليق الصوتي: "نولّد عروض احتفاظ مخصصة."
الإجراء: تبويب التوصيات. اختيار أعلى 5. ضغط "توليد توصيات الاحتفاظ".
التعليق الصوتي: "في الخلفية نستخدم Hugging Face Inference Router بواجهة متوافقة مع OpenAI ونموذج GPT-OSS-120B. البيانات تجريبية."

45-55 ث
التعليق الصوتي: "تظهر خطة منسقة مع إبراز العناصر المهمة بخط عريض ولكل توصية تأثير متوقّع."
الإجراء: تمرير الخطة وتوسيع توصيتين.

55-60 ث
التعليق الصوتي: "جاهزة كإثبات مفهوم ويمكن ربطها مع CRM والحملات. شكراً."
الإجراء: إنهاء على عرض الخطة.


## Full Source Code
```python
# app.py
# export HF_TOKEN="hf_..."
# streamlit run app.py

import os
import re
import json
import html
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# -----------------------------
# Settings
# -----------------------------
HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b:fireworks-ai"


# -----------------------------
# Client and state
# -----------------------------
def make_client() -> OpenAI:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Please set your Hugging Face token.")
    return OpenAI(base_url=HF_ROUTER, api_key=token)


def init_state():
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "plan_text" not in st.session_state:
        st.session_state.plan_text = ""
    if "parsed_recs" not in st.session_state:
        st.session_state.parsed_recs = None
    if "lang" not in st.session_state:
        st.session_state.lang = "Arabic"
    if "dialect" not in st.session_state:
        st.session_state.dialect = "الخليجية"


# -----------------------------
# LLM helpers
# -----------------------------
def extract_text_from_completion(completion) -> str:
    try:
        choices = getattr(completion, "choices", None) or completion.get("choices")
        first = choices[0]
        msg = getattr(first, "message", None) or first.get("message")
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "") or getattr(first, "text", "") or ""
    except Exception:
        try:
            return json.dumps(completion, ensure_ascii=False)
        except Exception:
            return str(completion)


def split_natural_and_json(full_text: str):
    if not full_text:
        return "", None
    m = re.search(r"<JSON>([\s\S]*?)</JSON>", full_text, re.IGNORECASE)
    if m:
        natural = (full_text[:m.start()] + full_text[m.end():]).strip()
        return natural, m.group(1).strip()
    m2 = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", full_text)
    if m2:
        json_text = m2.group(1).strip()
        natural = (full_text[:m2.start()] + full_text[m2.end():]).strip()
        return natural, json_text
    return full_text, None


# -----------------------------
# Data helpers
# -----------------------------
def validate_uploaded_df(df: pd.DataFrame):
    required = {
        "customer_id",
        "last_order_days",
        "avg_spend",
        "orders_month",
        "lifetime_months",
        "preference",
        "promo_used_recently",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ("Missing columns: " if st.session_state.lang == "English" else "الأعمدة الناقصة: ") + ", ".join(missing)
    return None


def generate_sample_subscribers(n=60, lang="Arabic"):
    np.random.seed(42)
    prefs = (
        ["نباتي", "نباتي صارم", "محب اللحوم", "منخفض الكربوهيدرات", "متوازن"]
        if lang == "Arabic"
        else ["Vegetarian", "Vegan", "Meat Lover", "Low Carb", "Balanced"]
    )
    rows = []
    for i in range(1, n + 1):
        last_order_days = int(np.clip(np.random.exponential(12), 0, 120))
        avg_spend = round(np.random.uniform(1.5, 10), 2)
        orders_month = int(np.random.poisson(3))
        lifetime_months = int(np.random.exponential(8))
        pref = np.random.choice(prefs)
        promo_used = np.random.choice([0, 1], p=[0.7, 0.3])
        rows.append(
            {
                "customer_id": f"C{i:04d}",
                "last_order_days": last_order_days,
                "avg_spend": avg_spend,
                "orders_month": orders_month,
                "lifetime_months": lifetime_months,
                "preference": pref,
                "promo_used_recently": int(promo_used),
            }
        )
    return pd.DataFrame(rows)


def churn_score(df: pd.DataFrame) -> pd.Series:
    score = (
        (df["last_order_days"] * 1.5)
        - (df["orders_month"] * 8)
        - (df["lifetime_months"] * 0.5)
        + (5 * (1 - (df["promo_used_recently"])))
    ) / (df["avg_spend"] + 1)
    s = 100 * (score - score.min()) / (score.max() - score.min() + 1e-6)
    return s.round(1)


# -----------------------------
# UI styling and rendering
# -----------------------------
def style_primary(color: str, lang: str):
    direction_css = (
        """
        .stApp { direction: rtl; }
        .stMarkdown, .stText { text-align: right; }
        [data-testid="stSidebar"] { direction: rtl; }
        """
        if lang == "Arabic"
        else """
        .stApp { direction: ltr; }
        .stMarkdown, .stText { text-align: left; }
        [data-testid="stSidebar"] { direction: ltr; }
        """
    )
    st.markdown(
        f"""
        <style>
        {direction_css}
        body, div, p, span {{
            font-family: "Tajawal","Cairo","Noto Kufi Arabic","Segoe UI",Arial,sans-serif;
        }}
        div.stButton>button {{
            background: {color};
            color: white;
            border-radius: 8px;
            border: 0;
            padding: 0.5rem 0.8rem;
        }}
        .highlight {{
            background: rgba(0,0,0,0.03);
            padding: 0.9rem 1.1rem;
            border-radius: 10px;
            border: 1px solid #eee;
        }}
        .plan-wrapper {{
            background: #ffffff;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 1rem 1.25rem;
        }}
        .plan-wrapper h3 {{
            margin: 0 0 12px 0;
            font-weight: 700;
        }}
        .plan-line {{
            line-height: 1.9;
            margin: 0 0 8px 0;
            font-size: 1.05rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bold(text: str) -> str:
    esc = html.escape(text)
    esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
    esc = re.sub(r"__(.+?)__", r"<strong>\1</strong>", esc)
    esc = re.sub(r"(\d+(?:\.\d+)?\s*[٪%])", r"<strong>\1</strong>", esc)
    for pat in [r"\bdiscount\b", r"\boffer\b", r"\bfree\s+(?:delivery|shipping)\b", r"\bvoucher\b", r"\bcoupon\b", r"\bpromo\b", r"\bcode\b", r"\bpoints?\b", r"\bdouble\s+points?\b", r"\bsave\b"]:
        esc = re.sub(pat, lambda m: f"<strong>{m.group(0)}</strong>", esc, flags=re.IGNORECASE)
    for pat in [r"خصم", r"عرض", r"كوبون", r"قسيمة", r"رمز", r"نقاط", r"مضاعفة\s+النقاط", r"(?:توصيل|شحن)\s+مجاني", r"مجاني"]:
        esc = re.sub(pat, lambda m: f"<strong>{m.group(0)}</strong>", esc)
    return esc


def render_plan_text(text: str, lang: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = [l for l in lines if l]
    paras_html = [f"<p class='plan-line'>{render_bold(l)}</p>" for l in cleaned]
    title = "Plan - English text" if lang == "English" else "الخطة - نص عربي"
    return f"<div class='plan-wrapper'><h3>{title}</h3>{''.join(paras_html)}</div>"


# -----------------------------
# Sidebar
# -----------------------------
def sidebar(lang: str):
    st.header("Settings" if lang == "English" else "الإعدادات")

    lang_choice = st.radio("Language / اللغة", ["Arabic", "English"], index=0 if lang == "Arabic" else 1)
    st.session_state.lang = lang_choice
    lang = lang_choice

    if lang == "Arabic":
        st.session_state.dialect = st.selectbox("اللهجة", ["الفصحى", "الخليجية"], index=1)

    if lang == "English":
        theme = st.selectbox("Theme color", ["Green", "Blue", "Purple"])
        color_map = {"Green": "#22c55e", "Blue": "#3b82f6", "Purple": "#8b5cf6"}
    else:
        theme = st.selectbox("لون الواجهة", ["أخضر", "أزرق", "بنفسجي"])
        color_map = {"أخضر": "#22c55e", "أزرق": "#3b82ف6", "بنفسجي": "#8b5cf6"}

    style_primary(color_map.get(theme, "#22c55e"), lang)

    st.session_state.model = st.text_input("Model ID" if lang == "English" else "معرف النموذج", value=st.session_state.model)

    st.markdown("**Data source**" if lang == "English" else "**مصدر البيانات**")
    uploaded = st.file_uploader("Upload CSV" if lang == "English" else "ارفع ملف CSV", type=["csv"])
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            err = validate_uploaded_df(up_df)
            if err:
                st.error(err)
            else:
                st.session_state["subscribers_df"] = up_df
                st.success("CSV uploaded successfully." if lang == "English" else "تم تحميل بيانات CSV بنجاح.")
        except Exception as e:
            st.error(("Failed to upload file: " if lang == "English" else "فشل تحميل الملف: ") + str(e))

    if st.button("Generate sample data" if lang == "English" else "إنشاء بيانات تجريبية"):
        st.session_state["subscribers_df"] = generate_sample_subscribers(lang=lang)

    if st.button("Clear data" if lang == "English" else "مسح البيانات"):
        st.session_state.pop("subscribers_df", None)
        st.session_state.plan_text = ""
        st.session_state.parsed_recs = None
        st.success("Cleared." if lang == "English" else "تم المسح.")


# -----------------------------
# Tabs
# -----------------------------
def tab_data_view(df: pd.DataFrame, lang: str):
    st.subheader("Data preview" if lang == "English" else "معاينة البيانات")
    st.dataframe(df, use_container_width=True)
    with st.expander("Data quality tips" if lang == "English" else "نصائح جودة البيانات"):
        st.markdown(
            "- Ensure all required columns exist.\n- Boolean values like promo_used_recently must be 0 or 1.\n- Numeric columns should not contain text."
            if lang == "English"
            else "- تأكد من وجود جميع الأعمدة المطلوبة.\n- القيم المنطقية كـ promo_used_recently يجب أن تكون 0 أو 1.\n- الأعمدة الرقمية بدون قيم نصية."
        )


def tab_scoring_view(df: pd.DataFrame, lang: str):
    st.subheader("Churn score calculation" if lang == "English" else "حساب درجة الانسحاب")
    try:
        df["churn_score"] = churn_score(df)
    except Exception as e:
        st.error(f"Failed to calculate churn score: {e}" if lang == "English" else f"تعذر حساب درجة الانسحاب: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers" if lang == "English" else "عدد العملاء", len(df))
    c2.metric("Avg score" if lang == "English" else "متوسط الدرجة", f"{df['churn_score'].mean():.1f}%")
    c3.metric("Max score" if lang == "English" else "أعلى درجة", f"{df['churn_score'].max():.1f}%")
    c4.metric("Min score" if lang == "English" else "أدنى درجة", f"{df['churn_score'].min():.1f}%")

    top_n = st.selectbox("Show top at-risk customers" if lang == "English" else "عرض الأعلى خطراً", [5, 10, 15, 20], index=0)

    st.dataframe(
        df.sort_values("churn_score", ascending=False).head(top_n),
        use_container_width=True,
        column_config={
            "churn_score": st.column_config.ProgressColumn(
                "Churn score" if lang == "English" else "درجة الانسحاب",
                help="Relative score 0 to 100" if lang == "English" else "قيمة نسبية من 0 إلى 100",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    with st.expander("How churn score is calculated" if lang == "English" else "كيف نحسب درجة الانسحاب"):
        st.markdown(
            "Composite score: longer since last order raises risk, frequent monthly orders and long lifetime reduce risk, and recent promo use reduces risk. Normalised to 0 to 100."
            if lang == "English"
            else "نحسب درجة مركبة تعتمد على عدة عوامل: تأخر آخر طلب يرفع الخطر، تكرار الشراء وطول العمر يقللان الخطر، واستخدام عرض ترويجي مؤخراً يقلل الخطر. نطبع النتيجة إلى نطاق 0 حتى 100."
        )


def _emoji_for_lift(lift: str) -> str:
    s = str(lift or "").strip().lower()
    if s in ["high", "عالي", "مرتفع"]:
        return "🔥"
    if s in ["medium", "متوسط"]:
        return "👍"
    if s in ["low", "منخفض"]:
        return "🔔"
    return "📈"


def tab_recommendations_view(df: pd.DataFrame, lang: str, dialect: str):
    st.subheader("Generate recommendations" if lang == "English" else "توليد التوصيات")

    df_scored = df.copy()
    if "churn_score" not in df_scored.columns:
        df_scored["churn_score"] = churn_score(df_scored)

    top_k = st.selectbox("Number of top at-risk customers" if lang == "English" else "عدد العملاء الأعلى خطراً", [5, 10, 15, 20], index=0)

    selected = df_scored.sort_values("churn_score", ascending=False).head(top_k)
    st.dataframe(
        selected[["customer_id", "last_order_days", "orders_month", "avg_spend", "churn_score"]],
        use_container_width=True,
        column_config={
            "churn_score": st.column_config.ProgressColumn(
                "Churn score" if lang == "English" else "درجة الانسحاب",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    if st.button("Generate retention recommendations" if lang == "English" else "توليد توصيات الاحتفاظ"):
        top_records = selected.to_dict(orient="records")

        if lang == "Arabic":
            sys_prompt = "أنت مساعد تسويق يتحدث العربية ومخصص لدول الخليج. اكتب بالعربية برسالة ودية ومهنية."
            user_prompt = (
                f"استخدم اللهجة: {dialect}. "
                "1) اكتب فقرة ودية قصيرة بالعربية لكل عميل تشرح العرض المقترح مع رموز تعبيرية. "
                "2) بعد الفقرات، أضف كتلة <JSON>...</JSON> فيها مصفوفة كائنات بالحقل: "
                "customer_id, action, message, expected_lift, rationale. "
                "القيم بالعربية فقط، وطول message ≤ 120 حرفًا. "
                f"العملاء: {json.dumps(top_records, ensure_ascii=False)}"
            )
        else:
            sys_prompt = "You are a marketing assistant specialised for the UK market. Write in professional yet warm UK English."
            user_prompt = (
                "1) For each customer, write a short friendly paragraph in English that explains the retention offer with emojis. "
                "2) Then add a <JSON>...</JSON> block containing an array of objects with keys: "
                "customer_id, action, message, expected_lift, rationale. "
                "Use UK English tone. Message length ≤ 120 characters. "
                f"Customers: {json.dumps(top_records, ensure_ascii=False)}"
            )

        with st.spinner("Generating plan and recommendations..." if lang == "English" else "جاري توليد الخطة والتوصيات"):
            try:
                client = make_client()
                completion = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                )
                full_text = extract_text_from_completion(completion)
                natural, json_text = split_natural_and_json(full_text)

                if natural:
                    st.session_state.plan_text = natural

                parsed = None
                if json_text:
                    try:
                        parsed = json.loads(json_text)
                    except Exception:
                        parsed = None
                        st.warning("Could not parse JSON." if lang == "English" else "تعذر قراءة JSON.")
                st.session_state.parsed_recs = parsed
                st.success("Generated successfully." if lang == "English" else "تم التوليد بنجاح.")
            except Exception as e:
                st.error(("Failed: " if lang == "English" else "فشل: ") + str(e))

    if st.session_state.plan_text:
        st.markdown(render_plan_text(st.session_state.plan_text, lang), unsafe_allow_html=True)

    if isinstance(st.session_state.parsed_recs, list):
        st.markdown("### Recommendations" if lang == "English" else "### التوصيات")
        for rec in st.session_state.parsed_recs:
            cid = rec.get("customer_id", "-")
            action = rec.get("action", "")
            message_raw = rec.get("message", "")
            lift = rec.get("expected_lift", "")
            rationale = rec.get("rationale", "")
            emoji = _emoji_for_lift(lift)

            with st.expander(f"{emoji} {cid} - {action}"):
                st.markdown(("**Message:** " if lang == "English" else "**الرسالة:** ") + render_bold(message_raw), unsafe_allow_html=True)
                st.markdown(("**Expected impact:** " if lang == "English" else "**التأثير المتوقع:** ") + f"**{html.escape(str(lift))}**")
                if rationale:
                    st.markdown(("**Rationale:** " if lang == "English" else "**السبب:** ") + html.escape(str(rationale)))


# -----------------------------
# App
# -----------------------------
def app():
    st.set_page_config(page_title="Subscription Growth Agent", layout="wide", initial_sidebar_state="expanded")
    init_state()

    with st.sidebar:
        sidebar(st.session_state.lang)

    if st.session_state.lang == "English":
        st.title("🌍 Subscription Growth Agent - Calo")
        st.markdown("🤖 Helps identify at-risk customers and generate retention recommendations.")
    else:
        st.title("🌍 وكيل نمو الاشتراكات - Calo")
        st.markdown("🤖 يساعد هذا التطبيق على التعرف على العملاء ذوي خطر الانسحاب وتوليد توصيات احتفاظ باللغة العربية.")

    if "subscribers_df" not in st.session_state:
        st.info("No data yet. Upload a CSV or generate sample data from the sidebar." if st.session_state.lang == "English" else "لا توجد بيانات بعد. ارفع CSV أو أنشئ بيانات تجريبية من الشريط الجانبي.")
        st.stop()

    df = st.session_state["subscribers_df"]
    tabs = (["Data", "Scoring", "Recommendations"] if st.session_state.lang == "English" else ["البيانات", "الحساب", "التوصيات"])
    tab_data, tab_scoring, tab_reco = st.tabs(tabs)

    with tab_data:
        tab_data_view(df, st.session_state.lang)
    with tab_scoring:
        tab_scoring_view(df, st.session_state.lang)
    with tab_reco:
        tab_recommendations_view(df, st.session_state.lang, st.session_state.dialect if st.session_state.lang == "Arabic" else "")


if __name__ == "__main__":
    try:
        app()
    except EnvironmentError as e:
        st.error(str(e))
    except Exception as e:
        st.exception(e)

```

# Alignment to Calo AI Specialist Role

- Agent development and prompt engineering - stable prompts produce both human plan and JSON payloads.
- POC prototyping - Streamlit app demonstrates end to end value in under 60 seconds.
- Data preparation - synthetic generator ensures safe demos with realistic fields.
- Model flexibility - HF Router keeps the app model agnostic for future swaps or fine tuned models.
- Knowledge sharing - PRD and demo script included for team handoff and lunch and learn sessions.


# Setup and Run

1. Install requirements:
   - streamlit, openai, pandas, numpy
2. Set environment variable:
   - export HF_TOKEN="hf_xxx"
3. Run:
   - streamlit run app.py
4. Optional:
   - Upload CSV with required columns or use sample data.

