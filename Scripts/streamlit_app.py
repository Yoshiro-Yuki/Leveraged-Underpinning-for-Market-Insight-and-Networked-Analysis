import streamlit as st
import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from meta_model import MetaModel

# --- 1. System Configuration & Theme ---
st.set_page_config(page_title="Mercari Intelligence", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #000000; }
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 15, 15, 0.8) !important;
        border: 2px solid rgba(0, 255, 200, 0.3) !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6) !important;
        padding: 10px !important;
    }
    
    .section-label { color: #00FFC8; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div { background-color: #0A0A0A; color: #00FFC8; border: 1px solid #333; }
    
    .stButton>button {
        border-radius: 8px; border: 2px solid #BC13FE; background: transparent;
        color: #FFFFFF; font-weight: bold; width: 100%; height: 45px; transition: 0.3s;
    }
    .stButton>button:hover { background: #BC13FE; box-shadow: 0 0 15px #BC13FE; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_system_model():
    return MetaModel.load_model() or MetaModel()

@st.cache_data(show_spinner=False)
def load_dropdown_data(filepath, id_key, name_key):
    """Loads JSON data and creates a mapping of Name -> ID for UI dropdowns."""
    if not os.path.exists(filepath):
        return {"Unknown": 0} # Fallback if file is missing
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    mapping = {}
    for entry in data.values():
        name = str(entry.get(name_key, "Unknown")).strip()
        item_id = entry.get(id_key, 0)
        # Keep the first occurrence of a name to avoid duplicates in the UI
        if name and name not in mapping:
            mapping[name] = item_id
            
    # Sort alphabetically, but force 'unknown' to the top
    sorted_keys = sorted(mapping.keys(), key=lambda x: (x.lower() != 'unknown', x))
    return {k: mapping[k] for k in sorted_keys}

def load_simple_list(filepath):
    """Use this if the JSON is just ['Item1', 'Item2', ...]"""
    if not os.path.exists(filepath):
        return ["Unknown"]
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Just return the list directly (sorted for the UI)
    return sorted([str(x).strip() for x in data if x])

def main():
    st.markdown("<h1 style='text-align: center; color: #00FFC8; margin-bottom: 30px;'>C2C Market Analyzer</h1>", unsafe_allow_html=True)

    # Load JSON lookups
    brand_mapping = load_dropdown_data("..\\Datasets\\brand_data.json", "brand_id", "brand_name")
    size_mapping = load_dropdown_data("..\\Datasets\\size_data.json", "size_id", "size_name")
    color_mapping = load_dropdown_data("..\\Datasets\\color_data.json", "color_id", "color")
    main_cat_mapping = load_simple_list("..\\Datasets\\main_categories.json")
    sub_cat_mapping = load_simple_list("..\\Datasets\\sub_categories.json")

    # --- SESSION STATE INITIALIZATION ---
    if 'dashboard_active' not in st.session_state:
        st.session_state.dashboard_active = False

    # --- ROW 1: INPUT SECTION ---
    with st.container(border=True):
        st.markdown("<div class='section-label'>⚙️ LISTING CONFIGURATION</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1.5])
        with c1:
            name = st.text_input("Item Name", placeholder="e.g. Apple iPhone 11", help="The exact title of your listing.")
            brand_name = st.selectbox("Brand", list(brand_mapping.keys()), help="Select the brand of the item.")
        with c2:
            cat_0 = st.selectbox("Category Main", list(main_cat_mapping), help="Select the closest matching size.")
            cat_2 = st.selectbox("Category Sub", list(sub_cat_mapping), help="Select the closest matching size.")
        with c3:
            cond = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor", "Very Poor"], index=2, help="1 is Mint/New, 5 is Poor/Damaged.")
            ship = st.selectbox("Shipping", ["Buyer Pays", "Seller Pays"], index=1, help="Items with free shipping (Seller Pays) often convert higher.")
        with c4:
            size_name = st.selectbox("Size", list(size_mapping.keys()), help="Select the closest matching size.")
            color_name = st.selectbox("Color", list(color_mapping.keys()), help="Select the primary item color.")

        # Transform inputs into Model-ready formats
        if cond == "Excellent": cond_val = 1
        elif cond == "Good": cond_val = 2 
        elif cond == "Fair": cond_val = 3
        elif cond == "Poor": cond_val = 4
        else: cond_val = 5
        
        shipper_val = 1 if ship == "Seller Pays" else 0
        
        # Extract IDs from our JSON dictionaries
        brand_id = brand_mapping[brand_name]
        size_id = size_mapping[size_name]
        color_id = color_mapping[color_name]

        predict_clicked = st.button("RUN SYSTEM INFERENCE")

    if predict_clicked:
        if not all([name, cat_0, cat_2]):
            st.error("SYSTEM ERROR: All text fields must be filled to run inference.")
            st.session_state.dashboard_active = False
        else:
            st.session_state.dashboard_active = True

    # --- RENDER DASHBOARD ---
    if st.session_state.dashboard_active:
        
        with st.spinner("INITIALIZING: Loading Neural Weights & Processing Market Variables..."):
            
            mm = load_system_model() 
            
            # Baseline Prediction
            price, (att, inter, conv) = mm.predict(
                name=name, brand_name=brand_name, brand_id=brand_id, item_condition=cond_val, 
                shipper=shipper_val, category_0=cat_0, category_2=cat_2,
                color_id=color_id, size_id=size_id
            )
            
            # Heuristic Calculations (calculating feature impact)
            p_no_brand, _ = mm.predict(name=name, brand_name="Unknown", brand_id=0, item_condition=cond_val, shipper=shipper_val, category_0=cat_0, category_2=cat_2, color_id=color_id, size_id=size_id)
            p_poor_cond, _ = mm.predict(name=name, brand_name=brand_name, brand_id=brand_id, item_condition=5, shipper=shipper_val, category_0=cat_0, category_2=cat_2, color_id=color_id, size_id=size_id)
            p_buyer_ship, _ = mm.predict(name=name, brand_name=brand_name, brand_id=brand_id, item_condition=cond_val, shipper=0, category_0=cat_0, category_2=cat_2, color_id=color_id, size_id=size_id)
            
            brand_impact = price - p_no_brand
            cond_impact = price - p_poor_cond
            ship_impact = price - p_buyer_ship

            # Generate Demand Curve Data (Simulating prices -30% to +30%)
            test_prices = np.linspace(max(1, price * 0.7), price * 1.3, 11)
            sim_conversions = []
            for p in test_prices:
                _, (_, _, c) = mm.predict(
                    name=name, brand_name=brand_name, brand_id=brand_id, item_condition=cond_val, 
                    shipper=shipper_val, category_0=cat_0, category_2=cat_2, 
                    color_id=color_id, size_id=size_id, price_override=p
                )
                sim_conversions.append(c)

        # --- ROW 2: PRICE & FLOW ---
        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            with st.container(border=True):
                st.markdown("<div class='section-label'>💰 PRICE ANALYSIS</div>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: white; margin:0;'>${price:,.2f}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #BC13FE;'>Optimal Range: ${max(0, price-32):,.2f} to ${price+32:,.2f}</p>", unsafe_allow_html=True)
                
                x = np.linspace(max(0, price - 100), price + 100, 100)
                y = norm.pdf(x, price, 32)
                fig_p = px.area(x=x, y=y, height=180)
                fig_p.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                fig_p.update_xaxes(showgrid=False, color="white")
                fig_p.update_yaxes(showgrid=False, showticklabels=False)
                fig_p.update_traces(fillcolor='rgba(188, 19, 254, 0.2)', line_color='#BC13FE')
                st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})

        with col_right:
            with st.container(border=True):
                st.markdown("<div class='section-label'>🌊 BEHAVIORAL FLOW</div>", unsafe_allow_html=True)
                fig_s = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15, thickness=20, line=dict(color="black", width=0.5),
                        label=["Audience", "Attracted", "Lost", "Interested", "Dropped", "Converted", "Abandoned"],
                        color=["#FFFFFF", "#00FFC8", "#333333", "#BC13FE", "#333333", "#00FFC8", "#333333"]
                    ),
                    link=dict(
                        source=[0, 0, 1, 1, 3, 3], target=[1, 2, 3, 4, 5, 6],
                        value=[att, 100-att, inter, att-inter, conv, inter-conv], color="rgba(255, 255, 255, 0.1)"
                    )
                )])
                fig_s.update_layout(font_color="#FFFFFF", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})

        # --- ROW 3: DONUTS & FEATURE IMPORTANCE ---
        col_d, col_f = st.columns([1.5, 1])
        with col_d:
            with st.container(border=True):
                st.markdown("<div class='section-label'>📊 PROBABILITY METRICS</div>", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                def make_donut(val, title, color):
                    fig = go.Figure(go.Pie(values=[val, 100-val], hole=.8, marker_colors=[color, "#1A1A1A"], textinfo='none', hoverinfo='none'))
                    fig.update_layout(showlegend=False, height=200, margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig.add_annotation(text=f"{val}%", showarrow=False, font=dict(size=26, color=color))
                    fig.add_annotation(text=title, y=1.15, showarrow=False, font=dict(size=14, color="#FFFFFF"))
                    return fig
                with d1: st.plotly_chart(make_donut(att, "ATTRACTION", "#00FFC8"), use_container_width=True)
                with d2: st.plotly_chart(make_donut(inter, "INTEREST", "#BC13FE"), use_container_width=True)
                with d3: st.plotly_chart(make_donut(conv, "CONVERSION", "#00FFC8"), use_container_width=True)
        
        with col_f:
            with st.container(border=True):
                st.markdown("<div class='section-label'>🔍 FEATURE IMPACT ($)</div>", unsafe_allow_html=True)
                fig_f = px.bar(
                    x=[brand_impact, cond_impact, ship_impact], 
                    y=["Brand", "Condition", "Free Ship"], 
                    orientation='h',
                    color=[brand_impact, cond_impact, ship_impact],
                    color_continuous_scale=["#BC13FE", "#00FFC8"]
                )
                fig_f.update_layout(height=200, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, coloraxis_showscale=False)
                fig_f.update_xaxes(title="", showgrid=False, color="white")
                fig_f.update_yaxes(title="", showgrid=False, color="white")
                st.plotly_chart(fig_f, use_container_width=True, config={'displayModeBar': False})

        # --- ROW 4: PRICE SENSITIVITY (DEMAND CURVE) ---
        with st.container(border=True):
            st.markdown("<div class='section-label'>📈 PRICE SENSITIVITY & DEMAND CURVE</div>", unsafe_allow_html=True)
            
            fig_demand = go.Figure()
            fig_demand.add_trace(go.Scatter(
                x=test_prices, y=sim_conversions, mode='lines+markers', 
                line=dict(color='#00FFC8', width=3, shape='spline'), 
                marker=dict(size=8, color='#BC13FE'),
                hovertemplate="Price: $%{x:.2f}<br>Conversion: %{y:.1f}%<extra></extra>"
            ))
            
            fig_demand.add_vline(x=price, line_width=2, line_dash="dash", line_color="#BC13FE")
            fig_demand.add_annotation(x=price, y=max(sim_conversions), text="Optimal Suggested Price", showarrow=False, yshift=15, font=dict(color="#BC13FE"))
            
            fig_demand.update_layout(
                height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Simulated Price ($)", color="white", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Conversion Prob (%)", color="white", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig_demand, use_container_width=True, config={'displayModeBar': False})

        # --- ROW 5: SUMMARY CARD ---
        with st.container(border=True):
            st.markdown("<div class='section-label'>📋 MARKET VIABILITY SUMMARY</div>", unsafe_allow_html=True)
            score = round((att + inter + conv) / 3, 1)
            c_score, c_comment = st.columns([1, 3])
            c_score.metric("SYSTEM SCORE", f"{score}%")
            
            if score >= 80:
                c_comment.success("🟢 Exceptional compatibility. The listing is highly optimized for immediate market conversion.")
            elif score >= 65:
                c_comment.success("🟢 High compatibility. The item features and suggested price point indicate strong potential for a successful sale.")
            elif score >= 50: 
                c_comment.info("🔵 Moderate viability. The listing should generate standard traction. Check the Demand Curve to see if a small price drop yields a big conversion jump.")
            elif score >= 35:
                c_comment.warning("🟡 Low traction detected. The current configuration may struggle to attract buyers. Optimize the title or lower the price.")
            else:
                c_comment.error("🔴 Critical visibility risk. Major revision of pricing, title, and categorization is strongly advised.")

if __name__ == "__main__":
    main()