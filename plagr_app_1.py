import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plagiarism Detector // Swiss",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. THEME STATE ---
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# --- 3. SWISS DESIGN SYSTEM (V2) ---
# Classic Swiss Style: Helvetica/Inter, Grid systems, Asymmetry, High Contrast
THEMES = {
    "light": {
        "bg": "#FFFFFF",
        "text": "#000000",
        "accent": "#FF3300", # International Orange/Red
        "secondary": "#E5E5E5",
        "border": "#000000",
        "surface": "#FFFFFF",
        "success": "#00CC66",
        "warning": "#FFCC00",
        "danger": "#FF3300"
    },
    "dark": {
        "bg": "#121212",
        "text": "#FFFFFF",
        "accent": "#FF3300",
        "secondary": "#333333",
        "border": "#FFFFFF",
        "surface": "#000000",
        "success": "#00CC66",
        "warning": "#FFCC00",
        "danger": "#FF3300"
    }
}

C = THEMES[st.session_state.theme]

# --- 4. CSS INJECTION ---
st.markdown(f"""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* RESET & BASE */
    [data-testid="stAppViewContainer"] {{
        background-color: {C['bg']};
    }}
    [data-testid="stHeader"] {{
        background-color: transparent;
    }}
    
    html, body, p, div, label, span, button {{
        font-family: 'Inter', Helvetica, Arial, sans-serif !important;
        color: {C['text']} !important;
    }}

    /* TYPOGRAPHY */
    h1 {{
        font-weight: 900 !important;
        font-size: 4rem !important;
        letter-spacing: -0.05em !important;
        text-transform: uppercase;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0.9 !important;
    }}
    
    h2, h3 {{
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        text-transform: uppercase;
    }}

    /* CONTROLS (BUTTONS) */
    .stButton > button {{
        background-color: {C['bg']} !important;
        color: {C['text']} !important;
        border: 2px solid {C['text']} !important;
        border-radius: 0px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        padding: 1rem 2rem !important;
        transition: all 0.2s ease;
        box-shadow: 4px 4px 0px {C['text']} !important;
    }}
    .stButton > button:hover {{
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px {C['text']} !important;
    }}
    .stButton > button:active {{
        transform: translate(4px, 4px);
        box-shadow: none !important;
        background-color: {C['accent']} !important;
        color: #fff !important;
    }}

    /* INPUTS */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {C['bg']} !important;
        border: 1px solid {C['secondary']} !important;
        border-left: 4px solid {C['text']} !important;
        border-radius: 0px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {C['accent']} !important;
    }}

    /* CUSTOM CLASSES */
    .hero-container {{
        border-bottom: 4px solid {C['text']};
        padding-bottom: 2rem;
        margin-bottom: 3rem;
    }}
    
    .meta-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: {C['accent']} !important;
        margin-bottom: 0.5rem;
        display: block;
    }}

    .metric-card {{
        border: 1px solid {C['secondary']};
        padding: 1.5rem;
        height: 100%;
        position: relative;
    }}

    .metric-value {{
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.05em;
    }}

    .metric-label {{
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        opacity: 0.6;
    }}

    .comparison-text {{
        font-family: 'Georgia', serif;
        line-height: 1.6;
        padding: 20px;
        background: {C['secondary']}20; /* 20 is hex opacity */
        height: 400px;
        overflow-y: auto;
        border: 1px solid {C['secondary']};
    }}

    .highlight-red {{
        background-color: {C['accent']};
        color: #FFFFFF !important;
        padding: 2px 0;
    }}
    
    /* PROGRESS BAR OVERRIDE */
    div[data-testid="stProgressBar"] > div {{
        height: 1rem !important;
        background-color: {C['secondary']} !important;
    }}
    div[data-testid="stProgressBar"] > div > div {{
        background-color: {C['accent']} !important;
    }}
    
</style>
""", unsafe_allow_html=True)

# --- 5. LOGIC CORE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

def similarity(doc1, doc2):
    return cosine_similarity([doc1], [doc2])[0][0]

def get_common_sentences(text1, text2, threshold=0.65):
    # Split by simple punctuation
    s1 = [s.strip() for s in re.split(r'[.!?]+', text1) if len(s.strip()) > 20]
    s2 = [s.strip() for s in re.split(r'[.!?]+', text2) if len(s.strip()) > 20]
    
    if not s1 or not s2: return []
    
    try:
        vectorizer = TfidfVectorizer()
        all_s = s1 + s2
        vectors = vectorizer.fit_transform(all_s)
        
        v1 = vectors[:len(s1)]
        v2 = vectors[len(s1):]
        
        matches = []
        for i, sent1 in enumerate(s1):
            for j, sent2 in enumerate(s2):
                sim = cosine_similarity(v1[i:i+1], v2[j:j+1])[0][0]
                if sim >= threshold:
                    matches.append((sent1, sent2, round(sim, 3)))
        
        # Sort by similarity score descending
        return sorted(matches, key=lambda x: x[2], reverse=True)[:50]
    except:
        return []

def highlight_text(text, sentences, is_first=True):
    result = text
    idx = 0 if is_first else 1
    # Simple replacement strategy (Note: This is basic and might overlap in complex cases)
    for pair in sentences:
        sent = pair[idx]
        if sent in result:
            result = result.replace(sent, f'<span class="highlight-red">{sent}</span>')
    return result

# --- 6. UI LAYOUT ---

# Top Bar (Toggle)
c_top, c_toggle = st.columns([10, 1])
with c_toggle:
    if st.button("◑", help="Toggle Theme"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

# Header
st.markdown(f"""
<div class="hero-container">
    <span class="meta-tag">/// SYS.ANALYSIS.V2</span>
    <h1>Plagiarism<br><span style="color: {C['accent']}">Detector</span></h1>
</div>
""", unsafe_allow_html=True)


# Main Content
if not st.session_state.analyzed:
    # --- INPUT MODE ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 01 / SOURCE A")
        text_a = st.text_area("Input A", height=400, placeholder="/// PASTE TEXT HERE", label_visibility="collapsed")
    
    with col2:
        st.markdown("### 02 / SOURCE B")
        text_b = st.text_area("Input B", height=400, placeholder="/// PASTE TEXT HERE", label_visibility="collapsed")

    # Action Bar
    st.markdown("<br>", unsafe_allow_html=True)
    c_action, c_void = st.columns([2, 4])
    with c_action:
        if st.button("RUN ANALYSIS // EXECUTE", use_container_width=True):
            if text_a and text_b:
                with st.spinner("CALCULATING VECTORS..."):
                    # Basic Logic for 2 texts
                    vecs = vectorize([text_a, text_b])
                    sim_score = similarity(vecs[0], vecs[1])
                    risk = "CRITICAL" if sim_score > 0.6 else "MODERATE" if sim_score > 0.3 else "LOW"
                    
                    st.session_state.results = [{
                        "a": "Source A", "b": "Source B", 
                        "score": sim_score, 
                        "risk": risk, 
                        "text_a": text_a, 
                        "text_b": text_b
                    }]
                    st.session_state.analyzed = True
                    st.rerun()
            else:
                st.error("/// DATA MISSING: INPUT REQUIRED")

else:
    # --- REPORT MODE ---
    res = st.session_state.results[0] # Focusing on single pair for this design update
    
    # 1. Metrics Grid
    m1, m2, m3 = st.columns([2, 2, 1])
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Similarity Index</div>
            <div class="metric-value">{res['score']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        risk_color = C['danger'] if res['score'] > 0.5 else C['text']
        st.markdown(f"""
        <div class="metric-card" style="border-left: 10px solid {risk_color}">
            <div class="metric-label">Risk Assessment</div>
            <div class="metric-value" style="font-size: 3rem; margin-top: 1rem;">{res['risk']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        if st.button("NEW\nSCAN", use_container_width=True):
            st.session_state.analyzed = False
            st.rerun()
            
    # 2. Visual Comparison
    st.markdown("<br><h3>/// TEXT MATCHING</h3>", unsafe_allow_html=True)
    
    common_sents = get_common_sentences(res['text_a'], res['text_b'])
    
    comp1, comp2 = st.columns(2, gap="medium")
    
    with comp1:
        st.caption("DOCUMENT A")
        safe_html_a = highlight_text(res['text_a'], common_sents, True)
        st.markdown(f'<div class="comparison-text">{safe_html_a}</div>', unsafe_allow_html=True)
        
    with comp2:
        st.caption("DOCUMENT B")
        safe_html_b = highlight_text(res['text_b'], common_sents, False)
        st.markdown(f'<div class="comparison-text">{safe_html_b}</div>', unsafe_allow_html=True)

    # 3. Footer / Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "DOWNLOAD LOGS [.CSV]", 
        pd.DataFrame(st.session_state.results).to_csv().encode(), 
        "analysis_log.csv", 
        "text/csv",
        use_container_width=True
    )