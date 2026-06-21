"""
Streamlit frontend for the EPL Match Outcome Predictor API.

PAGES
-----
  ⚽ Predict Match   — single match prediction with visual result card
  📅 Match Week      — predict a full set of fixtures (batch)
  📊 League Tracker  — Elo power rankings for all 20 teams
"""

import requests
import streamlit as st
from datetime import date, timedelta


# CONFIG

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title = "EPL Predictor",
    page_icon  = "⚽",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# DESIGN TOKENS  — pitch-green on near-black with gold accent

CSS = """
<style>
/* ── Base ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] h1 {
    color: #58a6ff;
}

/* ── Main header ──────────────────────────────────────── */
.main-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #e6edf3;
    line-height: 1.1;
}
.main-title span {
    color: #3fb950;   /* pitch green */
}
.subtitle {
    color: #8b949e;
    font-size: 1.0rem;
    margin-top: 0.3rem;
    margin-bottom: 1.8rem;
}

/* ── Section label ────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}

/* ── Team selector cards ──────────────────────────────── */
.team-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem 1rem;
}
.team-role {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.team-role.home { color: #3fb950; }
.team-role.away { color: #f78166; }

/* ── Result card ──────────────────────────────────────── */
.result-card {
    border-radius: 14px;
    padding: 2rem 2.4rem;
    margin: 1.2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card.home-win  { background: linear-gradient(135deg, #0d1f13 0%, #163222 100%); border: 1px solid #238636; }
.result-card.draw      { background: linear-gradient(135deg, #161b22 0%, #1c2128 100%); border: 1px solid #30363d; }
.result-card.away-win  { background: linear-gradient(135deg, #200d0d 0%, #2d1316 100%); border: 1px solid #f85149; }

.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.6rem;
}
.result-outcome {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.result-outcome.home-win { color: #3fb950; }
.result-outcome.draw     { color: #d29922; }
.result-outcome.away-win { color: #f78166; }
.result-conf {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 0.5rem;
}
.result-conf strong { color: #e6edf3; }

/* ── Probability bars ─────────────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.prob-label {
    width: 90px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #8b949e;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: #21262d;
    border-radius: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s ease;
}
.prob-val {
    width: 44px;
    font-size: 0.85rem;
    font-weight: 700;
    color: #e6edf3;
    text-align: right;
    flex-shrink: 0;
}

/* ── Elo comparison ───────────────────────────────────── */
.elo-block {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.elo-value {
    font-size: 1.7rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #58a6ff;
}
.elo-sublabel {
    font-size: 0.7rem;
    color: #8b949e;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Feature table ────────────────────────────────────── */
.feat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.feat-row {
    display: flex;
    justify-content: space-between;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 7px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
}
.feat-key { color: #8b949e; }
.feat-val { font-weight: 600; color: #e6edf3; }

/* ── Batch fixture rows ───────────────────────────────── */
.fixture-row {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.fixture-teams { font-weight: 700; font-size: 1rem; color: #e6edf3; }
.fixture-date  { font-size: 0.75rem; color: #8b949e; }
.badge {
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-H { background: #1a3928; color: #3fb950; border: 1px solid #238636; }
.badge-D { background: #2d2415; color: #d29922; border: 1px solid #9e6a03; }
.badge-A { background: #2d1316; color: #f78166; border: 1px solid #f85149; }

/* ── Power ranking rows ───────────────────────────────── */
.rank-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    background: #161b22;
    border: 1px solid #21262d;
}
.rank-pos   { width: 28px; font-weight: 700; font-size: 0.85rem; color: #8b949e; }
.rank-team  { flex: 1; font-weight: 600; font-size: 0.92rem; }
.rank-elo   { font-size: 0.82rem; font-weight: 700; color: #58a6ff; }
.rank-bar-bg { width: 90px; height: 7px; background: #21262d; border-radius: 4px; overflow: hidden; }
.rank-bar-fill { height: 100%; border-radius: 4px; background: #3fb950; }

/* ── Warning / info banners ───────────────────────────── */
.banner-warn {
    background: #2d2415;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #d29922;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
}
.banner-err {
    background: #2d1316;
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #f78166;
    font-size: 0.82rem;
}
.banner-ok {
    background: #0d1f13;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #3fb950;
    font-size: 0.82rem;
}

/* ── Divider ──────────────────────────────────────────── */
.divider { height: 1px; background: #21262d; margin: 1.4rem 0; }

/* ── Streamlit overrides ──────────────────────────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"] button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.6rem !important;
    font-size: 0.95rem !important;
    width: 100%;
    transition: background 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: #2ea043 !important;
}
.stSpinner > div { border-top-color: #3fb950 !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)



# API HELPERS  — all calls go through here

@st.cache_data(ttl=300)   # refresh team list every 5 minutes
def fetch_teams():
    try:
        r = requests.get(f"{API_BASE}/teams", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_predict(home: str, away: str, match_date: str):
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"home_team": home, "away_team": away, "match_date": match_date},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown API error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API — is `uvicorn app:app --port 8000` running?"
    except Exception as e:
        return None, str(e)


def call_batch_predict(matches: list):
    try:
        r = requests.post(
            f"{API_BASE}/predict/batch",
            json={"matches": matches},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown API error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API."
    except Exception as e:
        return None, str(e)



# RENDER HELPERS

def render_prob_bar(label: str, value: float, color: str):
    pct = int(value * 100)
    st.markdown(f"""
    <div class="prob-row">
      <div class="prob-label">{label}</div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
      <div class="prob-val">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(result: dict):
    code     = result['predicted_result']
    label    = result['predicted_label']
    conf     = result['confidence']
    home     = result['home_team']
    away     = result['away_team']
    probs    = result['probabilities']
    elo      = result['elo_ratings']

    cls_map  = {'H': 'home-win', 'D': 'draw', 'A': 'away-win'}
    cls      = cls_map[code]

    # Result hero card
    st.markdown(f"""
    <div class="result-card {cls}">
      <div class="result-label">Predicted outcome</div>
      <div class="result-outcome {cls}">{label}</div>
      <div class="result-conf">Confidence <strong>{conf:.0%}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Win probabilities</div>',
                unsafe_allow_html=True)
    render_prob_bar(f"🟢 {home}",  probs['home_win'], "#3fb950")
    render_prob_bar("⚪ Draw",     probs['draw'],     "#d29922")
    render_prob_bar(f"🔴 {away}",  probs['away_win'], "#f78166")

    # Elo comparison
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Elo ratings going into this match</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="elo-block">
          <div class="elo-sublabel">🏠 {home}</div>
          <div class="elo-value">{elo['home']:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        diff = elo['diff']
        sign = "+" if diff >= 0 else ""
        st.markdown(f"""
        <div class="elo-block">
          <div class="elo-sublabel">Elo edge</div>
          <div class="elo-value" style="color:{'#3fb950' if diff>0 else '#f78166' if diff<0 else '#d29922'}">{sign}{diff:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="elo-block">
          <div class="elo-sublabel">✈️ {away}</div>
          <div class="elo-value">{elo['away']:.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Feature breakdown (collapsible)
    with st.expander("🔬 Feature breakdown (all 18 model inputs)"):
        feats = result.get('features_used', {})
        labels = {
            'home_elo':'Home Elo','away_elo':'Away Elo','elo_diff':'Elo diff',
            'home_form_pts':'Home form pts (L5)','away_form_pts':'Away form pts (L5)',
            'home_form_gf':'Home goals scored (L5)','home_form_ga':'Home goals conceded (L5)',
            'away_form_gf':'Away goals scored (L5)','away_form_ga':'Away goals conceded (L5)',
            'home_home_pts':'Home form at home (L5)','away_away_pts':'Away form away (L5)',
            'home_home_gf':'Home goals (home, L5)','away_away_gf':'Away goals (away, L5)',
            'h2h_home_wins':'H2H home wins (L5)','h2h_away_wins':'H2H away wins (L5)',
            'h2h_draws':'H2H draws (L5)',
            'home_days_rest':'Home rest days','away_days_rest':'Away rest days',
        }
        rows_html = ""
        for k, lbl in labels.items():
            v = feats.get(k, "—")
            rows_html += f'<div class="feat-row"><span class="feat-key">{lbl}</span><span class="feat-val">{v}</span></div>'
        st.markdown(f'<div class="feat-grid">{rows_html}</div>', unsafe_allow_html=True)


def result_badge(code: str, label: str) -> str:
    return f'<span class="badge badge-{code}">{label}</span>'



# SIDEBAR


with st.sidebar:
    st.markdown("## ⚽ EPL Predictor")
    st.markdown('<div style="color:#8b949e;font-size:0.82rem;margin-bottom:1.2rem">Premier League match outcome prediction using machine learning</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["⚽  Predict Match", "📅  Match Week", "📊  Power Rankings"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Live API health check
    st.markdown('<div class="section-label">API status</div>', unsafe_allow_html=True)
    health = fetch_health()
    if health and health.get('status') == 'ok':
        st.markdown(f"""
        <div class="banner-ok">
          ● API online &nbsp;·&nbsp; {health.get('teams_available', 0)} teams loaded<br>
          <span style="color:#8b949e;font-size:0.72rem">States from {health.get('generated_at','—')}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="banner-err">
          ● API offline<br>
          <span style="font-size:0.75rem">Run: uvicorn app:app --port 8000</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8b949e;font-size:0.72rem">Model: 3-class classifier<br>Features: 18 pre-match only<br>Target: H / D / A</div>', unsafe_allow_html=True)



# PAGE 1 — PREDICT MATCH


if "Predict Match" in page:
    st.markdown('<div class="main-title">Match <span>Predictor</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select two teams and a date — the model predicts the outcome using pre-match stats only.</div>', unsafe_allow_html=True)

    teams = fetch_teams()
    if not teams:
        st.markdown('<div class="banner-warn">⚠️ Could not load team list. Check the API is running.</div>', unsafe_allow_html=True)
        st.stop()

    # Team selectors 
    col_home, col_vs, col_away = st.columns([5, 1, 5])

    with col_home:
        st.markdown('<div class="section-label">🏠 Home team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0, label_visibility="collapsed")

    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:1.2rem;font-weight:900;color:#8b949e;padding-top:0.3rem">vs</div>', unsafe_allow_html=True)

    with col_away:
        st.markdown('<div class="section-label">✈️ Away team</div>', unsafe_allow_html=True)
        default_away = "Chelsea" if "Chelsea" in teams else (teams[1] if len(teams) > 1 else teams[0])
        away_team = st.selectbox("Away team", teams, index=teams.index(default_away) if default_away in teams else 1, label_visibility="collapsed")

    # Date + predict button 
    col_date, col_btn = st.columns([3, 2])
    with col_date:
        st.markdown('<div class="section-label" style="margin-top:0.8rem">📅 Match date</div>', unsafe_allow_html=True)
        match_date = st.date_input("Match date", value=date.today() + timedelta(days=7), label_visibility="collapsed")

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚽ Predict outcome", use_container_width=True)

    # Validation 
    if home_team == away_team:
        st.markdown('<div class="banner-warn">⚠️ Home and away teams must be different.</div>', unsafe_allow_html=True)
        st.stop()


    # ── Result ────────────────────────────────────────────────────────────
    if predict_clicked:
        with st.spinner("Predicting…"):
            result, error = call_predict(home_team, away_team, str(match_date))

        if error:
            st.markdown(f'<div class="banner-err">❌ {error}</div>', unsafe_allow_html=True)
        else:
            if result.get('warnings'):
                for w in result['warnings']:
                    st.markdown(f'<div class="banner-warn">⚠️ {w}</div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            render_result_card(result)



# PAGE 2 — MATCH WEEK

elif "Match Week" in page:
    st.markdown('<div class="main-title">Match Week <span>Fixtures</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter up to 10 fixtures and predict the full match-week at once.</div>', unsafe_allow_html=True)

    teams = fetch_teams()
    if not teams:
        st.markdown('<div class="banner-warn">⚠️ Could not load team list. Check the API is running.</div>', unsafe_allow_html=True)
        st.stop()

    # Fixture builder ─
    if 'fixtures' not in st.session_state:
        st.session_state.fixtures = [
            {"home": teams[0], "away": teams[1] if len(teams)>1 else teams[0], "date": str(date.today() + timedelta(days=7))}
        ]

    col_add, col_clear = st.columns([2, 1])
    with col_add:
        if st.button("+ Add fixture", use_container_width=True):
            if len(st.session_state.fixtures) < 10:
                st.session_state.fixtures.append(
                    {"home": teams[0], "away": teams[2] if len(teams)>2 else teams[1], "date": str(date.today() + timedelta(days=7))}
                )
            else:
                st.warning("Maximum 10 fixtures per batch.")

    with col_clear:
        if st.button("✕ Clear all", use_container_width=True):
            st.session_state.fixtures = []
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Render each fixture input row
    to_delete = []
    for i, fix in enumerate(st.session_state.fixtures):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([4, 0.6, 4, 3, 1])
            with c1:
                st.session_state.fixtures[i]['home'] = st.selectbox(
                    f"Home {i}", teams,
                    index=teams.index(fix['home']) if fix['home'] in teams else 0,
                    label_visibility="collapsed", key=f"home_{i}"
                )
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div style="text-align:center;color:#8b949e;font-weight:700;padding-top:0.3rem">vs</div>', unsafe_allow_html=True)
            with c3:
                st.session_state.fixtures[i]['away'] = st.selectbox(
                    f"Away {i}", teams,
                    index=teams.index(fix['away']) if fix['away'] in teams else (1 if len(teams)>1 else 0),
                    label_visibility="collapsed", key=f"away_{i}"
                )
            with c4:
                d = st.date_input(f"Date {i}", value=date.today() + timedelta(days=7),
                                  label_visibility="collapsed", key=f"date_{i}")
                st.session_state.fixtures[i]['date'] = str(d)
            with c5:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_{i}"):
                    to_delete.append(i)

    for idx in sorted(to_delete, reverse=True):
        st.session_state.fixtures.pop(idx)
    if to_delete:
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("⚽ Predict all fixtures", use_container_width=True):
        valid = [f for f in st.session_state.fixtures if f['home'] != f['away']]
        invalid_count = len(st.session_state.fixtures) - len(valid)
        if invalid_count:
            st.markdown(f'<div class="banner-warn">⚠️ {invalid_count} fixture(s) skipped — home and away teams must differ.</div>', unsafe_allow_html=True)

        if not valid:
            st.markdown('<div class="banner-err">No valid fixtures to predict.</div>', unsafe_allow_html=True)
        else:
            payload = [{"home_team": f['home'], "away_team": f['away'], "match_date": f['date']} for f in valid]
            with st.spinner(f"Predicting {len(payload)} fixtures…"):
                data, error = call_batch_predict(payload)

            if error:
                st.markdown(f'<div class="banner-err">❌ {error}</div>', unsafe_allow_html=True)
            else:
                preds = data.get('predictions', [])
                st.markdown(f'<div class="section-label" style="margin-top:0.5rem">Results — {len(preds)} fixtures</div>', unsafe_allow_html=True)

                for p in preds:
                    if 'error' in p:
                        st.markdown(f'<div class="banner-err">❌ {p["home_team"]} vs {p["away_team"]}: {p["error"]}</div>', unsafe_allow_html=True)
                        continue

                    code  = p['predicted_result']
                    label = p['predicted_label']
                    probs = p['probabilities']
                    hw, dr, aw = int(probs['home_win']*100), int(probs['draw']*100), int(probs['away_win']*100)

                    badge_html = result_badge(code, label)
                    bar_home  = f'<div class="prob-bar-fill" style="width:{hw}%;background:#3fb950"></div>'
                    bar_draw  = f'<div class="prob-bar-fill" style="width:{dr}%;background:#d29922"></div>'
                    bar_away  = f'<div class="prob-bar-fill" style="width:{aw}%;background:#f78166"></div>'

                    st.markdown(f"""
                    <div class="fixture-row">
                      <div>
                        <div class="fixture-teams">{p['home_team']} <span style="color:#8b949e">vs</span> {p['away_team']}</div>
                        <div class="fixture-date">{p['match_date']}</div>
                      </div>
                      <div style="flex:1;padding:0 1.5rem">
                        <div class="prob-row" style="margin-bottom:0.3rem">
                          <div class="prob-label" style="width:70px;font-size:0.72rem">{p['home_team'][:10]}</div>
                          <div class="prob-bar-bg" style="height:7px">{bar_home}</div>
                          <div class="prob-val" style="width:35px;font-size:0.78rem">{hw}%</div>
                        </div>
                        <div class="prob-row" style="margin-bottom:0.3rem">
                          <div class="prob-label" style="width:70px;font-size:0.72rem">Draw</div>
                          <div class="prob-bar-bg" style="height:7px">{bar_draw}</div>
                          <div class="prob-val" style="width:35px;font-size:0.78rem">{dr}%</div>
                        </div>
                        <div class="prob-row">
                          <div class="prob-label" style="width:70px;font-size:0.72rem">{p['away_team'][:10]}</div>
                          <div class="prob-bar-bg" style="height:7px">{bar_away}</div>
                          <div class="prob-val" style="width:35px;font-size:0.78rem">{aw}%</div>
                        </div>
                      </div>
                      <div>{badge_html}</div>
                    </div>
                    """, unsafe_allow_html=True)



# PAGE 3 — POWER RANKINGS (Elo leaderboard)

elif "Power Rankings" in page:
    st.markdown('<div class="main-title">Elo <span>Power Rankings</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live team strength rankings based on Elo ratings — updated after every match in the dataset.</div>', unsafe_allow_html=True)

    # We build rankings by predicting each team vs a neutral reference via the health endpoint
    # and pulling Elo from team_states. Since we can't call a "get Elo" endpoint directly,
    # we build rankings from real predictions for a reference fixture.
    teams = fetch_teams()
    if not teams:
        st.markdown('<div class="banner-warn">⚠️ Could not load team list.</div>', unsafe_allow_html=True)
        st.stop()

    if not teams or len(teams) < 2:
        st.info("Not enough teams to build rankings.")
        st.stop()

    st.markdown('<div class="section-label">Collecting Elo ratings from API…</div>', unsafe_allow_html=True)

    # Call predict for every team vs the first team to extract Elo snapshots
    elo_data = {}
    ref_team = teams[0]

    with st.spinner(f"Fetching Elo ratings for {len(teams)} teams…"):
        for team in teams:
            if team == ref_team:
                continue
            result, _ = call_predict(team, ref_team, str(date.today() + timedelta(days=30)))
            if result:
                elo_data[team]     = result['elo_ratings']['home']
                elo_data[ref_team] = result['elo_ratings']['away']

    if not elo_data:
        st.markdown('<div class="banner-err">Could not load Elo ratings. Check the API is running.</div>', unsafe_allow_html=True)
        st.stop()

    # Sort by Elo descending
    ranked = sorted(elo_data.items(), key=lambda x: x[1], reverse=True)
    max_elo = ranked[0][1]
    min_elo = ranked[-1][1]
    elo_range = max(max_elo - min_elo, 1)

    # Zone colours
    def zone_color(pos, total):
        if pos <= 4:   return "#3fb950"   # UCL
        if pos <= 6:   return "#58a6ff"   # Europa
        if pos >= total - 2: return "#f85149"  # Relegation
        return "#8b949e"

    total = len(ranked)
    st.markdown("<br>", unsafe_allow_html=True)

    # Legend
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1: st.markdown('<div style="color:#3fb950;font-size:0.78rem;font-weight:700">🟢 Top 4 — Champions League</div>', unsafe_allow_html=True)
    with lcol2: st.markdown('<div style="color:#58a6ff;font-size:0.78rem;font-weight:700">🔵 5th-6th — Europa League</div>', unsafe_allow_html=True)
    with lcol3: st.markdown('<div style="color:#f85149;font-size:0.78rem;font-weight:700">🔴 Bottom 3 — Relegation zone</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for pos, (team, elo) in enumerate(ranked, start=1):
        bar_width = int(((elo - min_elo) / elo_range) * 100)
        color     = zone_color(pos, total)
        st.markdown(f"""
        <div class="rank-row">
          <div class="rank-pos" style="color:{color}">{pos}</div>
          <div class="rank-team" style="color:{color if pos<=4 or pos>total-3 else '#e6edf3'}">{team}</div>
          <div class="rank-bar-bg">
            <div class="rank-bar-fill" style="width:{bar_width}%;background:{color}"></div>
          </div>
          <div class="rank-elo" style="color:{color}">{elo:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#8b949e;font-size:0.75rem">Elo range: {min_elo:.0f} – {max_elo:.0f} · Baseline: 1500 · {len(ranked)} teams ranked</div>', unsafe_allow_html=True)