import json
import os
import re
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SelfAttention.in AI Holiday Planner",
    page_icon="✈️",
    layout="wide",
)


# ---------- UI styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #07111f 0%, #101c33 45%, #0f172a 100%);
        color: #e2e8f0;
    }

    .stApp > header {
        background: transparent;
    }

    .top-panel {
        background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(14,116,144,0.18), rgba(15,23,42,0.55));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 24px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
        animation: float 6s ease-in-out infinite;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.2);
    }

    .shine {
        background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(96,165,250,0.45), rgba(255,255,255,0.05));
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.7);
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #38bdf8, #2563eb);
        color: white;
    }

    .day-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def get_api_key() -> str:
    env_value = os.getenv("GEMINI_API_KEY", "")
    if env_value:
        return env_value
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def get_serper_api_key() -> str:
    env_value = os.getenv("SERPER_API_KEY", "")
    if env_value:
        return env_value
    try:
        return st.secrets["SERPER_API_KEY"]
    except Exception:
        return ""


def get_login_credentials() -> tuple[str, str]:
    username = os.getenv("APP_USERNAME", "selfattention").strip()
    password = os.getenv("APP_PASSWORD", "selfattention@123").strip()
    return username, password


@st.cache_data
def search_serper(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    api_key = get_serper_api_key()
    if not api_key:
        return []

    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=25)
        response.raise_for_status()
        data = response.json()

        organic = data.get("organic", []) or data.get("organic_results", [])
        results: List[Dict[str, str]] = []

        for item in organic[:num_results]:
            title = item.get("title") or item.get("name") or "Result"
            snippet = item.get("snippet") or item.get("description") or ""
            link = item.get("link") or ""
            if title or snippet:
                results.append({"title": title, "snippet": snippet, "link": link})

        if not results and data.get("answer_box"):
            answer = data["answer_box"]
            results.append({
                "title": answer.get("title") or "Quick answer",
                "snippet": answer.get("snippet") or answer.get("answer") or "",
                "link": answer.get("link") or "",
            })

        return results
    except Exception:
        return []


@st.cache_resource
def get_model():
    api_key = get_api_key()
    if not api_key:
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-3.5-flash-lite")


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text


def safe_json_parse(raw_text: str) -> Dict[str, Any]:
    cleaned = clean_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("The AI response was not valid JSON.")


def build_travel_prompt(form_data: Dict[str, Any]) -> str:
    destination = form_data.get("destination", "")
    start_date = form_data.get("start_date", "")
    travelers = form_data.get("travelers", 2)
    budget = form_data.get("budget", 2000)
    currency = form_data.get("currency", "USD")
    trip_style = form_data.get("trip_style", "Balanced")
    interests = form_data.get("interests", "scenic views and local food")
    trip_length = form_data.get("trip_length", 5)

    return """
You are a senior travel strategist and itinerary designer.
Create a realistic, exciting holiday plan for the following traveler profile.

Return ONLY valid JSON with this exact schema:
{{
  "destination": "string",
  "country": "string",
  "summary": "string",
  "best_time_to_visit": "string",
  "travel_style": "string",
  "trip_duration_days": 5,
  "travelers": 2,
  "budget_total": 2000,
  "currency": "USD",
  "budget_breakdown": [
    {{"category": "Flights", "amount": 500}},
    {{"category": "Stay", "amount": 700}},
    {{"category": "Food", "amount": 300}},
    {{"category": "Activities", "amount": 250}},
    {{"category": "Local transport", "amount": 250}}
  ],
  "highlights": ["string"],
  "daily_itinerary": [
    {{"day": 1, "title": "Arrival and city orientation", "details": "Detailed description", "cost": 110}}
  ],
  "packing_list": ["string"],
  "flight_strategy": "string",
  "local_tips": ["string"]
}}

Rules:
- Keep the total daily itinerary within the stated budget.
- Use realistic costs in the same currency.
- Add local, practical tips that feel authentic.
- Match the travelers and trip style.
- Keep each day short but detailed.
- The destination must be {destination}.
- Trip starts around {start_date}.
- Travelers: {travelers}.
- Budget: {budget} {currency}.
- Trip style: {trip_style}.
- Trip duration: {trip_length} days.
- Interests: {interests}.

Only respond with valid JSON, no markdown fences and no extra notes.
""".format(
        destination=destination,
        start_date=start_date,
        travelers=travelers,
        budget=budget,
        currency=currency,
        trip_style=trip_style,
        interests=interests,
        trip_length=trip_length,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class='top-panel'>
            <div style='display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:20px;'>
                <div>
                    <div style='font-size:12px; letter-spacing:0.18em; text-transform:uppercase; color:#7dd3fc; margin-bottom:8px;'>AI-powered travel design</div>
                    <h1 style='margin:0; font-size:3rem; line-height:1.05; color:#f8fafc;'>SelfAttention.in AI Holiday Planner</h1>
                    <div style='margin-top:10px; font-size:0.9rem; color:#bae6fd;'>Developed by interns at selfattention.in</div>
                </div>
                <div class='shine' style='padding:12px 20px; border-radius:999px; border:1px solid rgba(125,211,252,0.4); color:#e0f2fe; font-weight:700; background:rgba(14,116,144,0.25);'>Gemini free-tier ready</div>
            </div>
            <p style='margin-top:1rem; color:#cbd5e1; max-width:1000px; font-size:1.1rem;'>Craft immersive, budget-aware vacation plans with rich destination summaries, day-by-day itineraries, and live travel visuals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_missing_key_warning() -> None:
    st.warning(
        "Add your Gemini API key to Streamlit Cloud secrets as GEMINI_API_KEY or set it in your local environment before running the planner."
    )


def render_serper_results(category: str, query: str) -> None:
    st.markdown(f"### {category}")
    results = search_serper(query, num_results=5)

    if not results:
        st.info(f"No live {category.lower()} results were returned. Add a valid SERPER_API_KEY to enable live search.")
        return

    for item in results:
        title = item.get("title", "Result")
        snippet = item.get("snippet", "")
        link = item.get("link")

        st.markdown(f"**{title}**")
        if link:
            st.markdown(f"[Open result]({link})")
        if snippet:
            st.write(snippet)
        st.markdown("---")


def render_plan(plan: Dict[str, Any], origin: str = "London") -> None:
    st.subheader("Trip snapshot")

    metrics = st.columns(4)
    with metrics[0]:
        st.markdown(
            f"<div class='metric-card'><div style='color:#7dd3fc; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase;'>Destination</div><div style='font-size:1.9rem; font-weight:800; margin-top:8px;'>{plan.get('destination', 'N/A')}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics[1]:
        st.markdown(
            f"<div class='metric-card'><div style='color:#7dd3fc; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase;'>Trip length</div><div style='font-size:1.9rem; font-weight:800; margin-top:8px;'>{plan.get('trip_duration_days', 0)} days</div></div>",
            unsafe_allow_html=True,
        )
    with metrics[2]:
        st.markdown(
            f"<div class='metric-card'><div style='color:#7dd3fc; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase;'>Travelers</div><div style='font-size:1.9rem; font-weight:800; margin-top:8px;'>{plan.get('travelers', 0)}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics[3]:
        st.markdown(
            f"<div class='metric-card'><div style='color:#7dd3fc; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase;'>Budget</div><div style='font-size:1.9rem; font-weight:800; margin-top:8px;'>{plan.get('currency', 'USD')} {plan.get('budget_total', 0):,.0f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tab_overview, tab_itinerary, tab_budget, tab_pack = st.tabs([
        "Overview",
        "Day-by-day plan",
        "Budget analytics",
        "Packing & tips",
    ])

    with tab_overview:
        st.markdown(f"### {plan.get('destination', 'Destination')} overview")
        st.write(plan.get("summary", "No summary available."))

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"Best time to visit: {plan.get('best_time_to_visit', 'Flexible')}")
            st.info(f"Travel style: {plan.get('travel_style', 'Balanced')}")
        with col_b:
            st.success(f"Flight strategy: {plan.get('flight_strategy', 'Flexible route planning')}")
            st.success(f"Country focus: {plan.get('country', 'TBD')}")

        st.markdown("### Signature highlights")
        highlights = plan.get("highlights", [])
        if highlights:
            for item in highlights:
                st.markdown(f"- ⭐ {item}")

    with tab_itinerary:
        days = plan.get("daily_itinerary", [])
        if days:
            for day in days:
                with st.container():
                    st.markdown(
                        f"<div class='day-card'><h4 style='margin:0 0 0.3rem 0;'>Day {day.get('day', '')}: {day.get('title', '')}</h4><p style='color:#cbd5e1; margin:0.2rem 0 0.5rem 0;'>{day.get('details', '')}</p><strong>Estimated spend:</strong> {plan.get('currency', 'USD')} {day.get('cost', 0):,.0f}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.write("Itinerary data is not available yet.")

    with tab_budget:
        budget_breakdown = plan.get("budget_breakdown", [])
        if budget_breakdown:
            budget_df = pd.DataFrame(budget_breakdown)
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.bar_chart(budget_df.set_index("category")["amount"])
            with chart_cols[1]:
                fig = px.pie(
                    budget_df,
                    names="category",
                    values="amount",
                    title="Budget distribution",
                    color_discrete_sequence=px.colors.sequential.Sunset,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0"},
                    legend={"font": {"color": "#e2e8f0"}},
                )
                st.plotly_chart(fig, use_container_width=True)

            daily_costs = []
            for day in plan.get("daily_itinerary", []):
                daily_costs.append({"day": day.get("day", 0), "cost": day.get("cost", 0)})
            if daily_costs:
                daily_df = pd.DataFrame(daily_costs)
                line_fig = px.line(
                    daily_df,
                    x="day",
                    y="cost",
                    title="Daily spend rhythm",
                    markers=True,
                    color_discrete_sequence=["#38bdf8"],
                )
                line_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0"},
                    xaxis={"title": "Day"},
                    yaxis={"title": "Cost"},
                )
                st.plotly_chart(line_fig, use_container_width=True)

    with tab_pack:
        st.markdown("### Packing checklist")
        for item in plan.get("packing_list", []):
            st.markdown(f"- 📦 {item}")

        st.markdown("### Local tips")
        for tip in plan.get("local_tips", []):
            st.markdown(f"- 💡 {tip}")

    st.markdown("---")
    st.subheader("Live options from Serper")
    serper_key = get_serper_api_key()
    if not serper_key:
        st.info("Add a SERPER_API_KEY in Streamlit Cloud secrets to fetch live flight, hotel, and food suggestions.")
    else:
        destination = plan.get("destination", "")
        start_date = plan.get("best_time_to_visit") or "upcoming dates"
        currency = plan.get("currency", "USD")
        interests = plan.get("travel_style", "local experiences")

        tabs = st.tabs(["Flights", "Hotels", "Food"])
        with tabs[0]:
            render_serper_results(
                "Flights",
                f"best flight options from {origin} to {destination} {start_date} flights affordable",
            )
        with tabs[1]:
            render_serper_results(
                "Hotels",
                f"best hotels in {destination} {currency} travel accommodation {plan.get('travel_style', 'comfortable')}",
            )
        with tabs[2]:
            render_serper_results(
                "Food",
                f"best restaurants food options in {destination} local cuisine {interests}",
            )


def login_page() -> bool:
    st.markdown(
        """
        <div class='top-panel'>
            <div style='font-size:12px; letter-spacing:0.18em; text-transform:uppercase; color:#7dd3fc; margin-bottom:8px;'>selfattention.in</div>
            <h1 style='margin:0; font-size:2.5rem; color:#f8fafc;'>SelfAttention.in Travel Planner</h1>
            <p style='margin-top:0.8rem; color:#cbd5e1;'>Sign in to access the SelfAttention.in AI holiday planner.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    username, password = get_login_credentials()

    with st.form("login_form"):
        st.markdown("### Login")
        entered_username = st.text_input("Username")
        entered_password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)

    if submit:
        if entered_username == username and entered_password == password:
            st.session_state["authenticated"] = True
            st.rerun()
            return True
        st.error("Invalid username or password.")

    return False


def main() -> None:
    st.session_state.setdefault("authenticated", False)

    if not st.session_state.get("authenticated"):
        if login_page():
            return
        return

    render_hero()

    api_key = get_api_key()
    if not api_key:
        show_missing_key_warning()
        with st.sidebar:
            st.header("Trip builder")
            st.info("Set GEMINI_API_KEY in your .env or Streamlit Cloud secret to enable the planner.")
        return

    model = get_model()

    with st.sidebar:
        st.header("Session")
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("authenticated", None)
            st.session_state.pop("trip_plan", None)
            st.rerun()

        st.header("Plan a new trip")
        with st.form("travel_form"):
            destination = st.text_input("Destination", value="Paris")
            origin = st.text_input("Departure city / airport", value="London")
            start_date = st.date_input("Departure date")
            trip_length = st.slider("Trip length (days)", 2, 14, 5)
            travelers = st.slider("Travelers", 1, 6, 2)
            budget = st.number_input("Budget", min_value=300, value=2200, step=100)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "AED", "INR"])
            trip_style = st.selectbox("Travel style", ["Luxury", "Balanced", "Budget", "Adventure", "Family"])
            interests = st.text_area("Interests", value="food, culture, scenic walks, local markets")
            submitted = st.form_submit_button("Generate holiday plan", use_container_width=True)

    if submitted:
        form_data = {
            "destination": destination,
            "start_date": start_date.isoformat(),
            "trip_length": int(trip_length),
            "travelers": int(travelers),
            "budget": float(budget),
            "currency": currency,
            "trip_style": trip_style,
            "interests": interests,
        }

        prompt = build_travel_prompt(form_data)

        with st.spinner("Gemini is designing your custom itinerary and travel visuals..."):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "response_mime_type": "application/json",
                    },
                )
                plan = safe_json_parse(response.text)
            except Exception as exc:
                st.error(f"The AI planner could not generate a valid itinerary: {exc}")
                return

        st.session_state["trip_plan"] = plan

    if "trip_plan" in st.session_state:
        render_plan(st.session_state["trip_plan"], origin=origin)
    else:
        st.markdown(
            """
            <div style='margin-top:2rem; padding:1.5rem; background:rgba(15,23,42,0.7); border:1px solid rgba(148,163,184,0.15); border-radius:18px;'>
                <h3 style='margin-top:0;'>What this planner does</h3>
                <p>Generate a personalized holiday plan with balanced budgets, neighborhood guides, local insights, and rich visual breakdowns.</p>
                <ul>
                    <li>AI-powered itinerary generation</li>
                    <li>Budget and daily cost tracking</li>
                    <li>Animated visual summaries</li>
                    <li>Travel packing and route guidance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
