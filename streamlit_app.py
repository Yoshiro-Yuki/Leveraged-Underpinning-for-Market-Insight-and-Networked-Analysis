import streamlit as st
import os
import json
import pandas as pd
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
    return MetaModel.load_model()

@st.cache_data(show_spinner=False)
def load_dropdown_data(filepath, id_key, name_key):
    if not os.path.exists(filepath): return {"Unknown": 0}
    with open(filepath, 'r') as f:
        data = json.load(f)
    mapping = {str(e.get(name_key, "Unknown")).strip(): e.get(id_key, 0) for e in data.values()}
    return {k: mapping[k] for k in sorted(mapping.keys(), key=lambda x: (x.lower() != 'unknown', x))}

@st.cache_data(show_spinner=False)
def load_simple_list(filepath):
    if not os.path.exists(filepath): return ["Unknown"]
    with open(filepath, 'r') as f:
        data = json.load(f)
    return sorted([str(x).strip() for x in data if x])

def main():
    override_price = False
    st.markdown("<h1 style='text-align: center; color: #00FFC8; margin-bottom: 5px;'>Price Estimation & Value-Added Customer Path (PREVAC)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFFFFF; margin-bottom: 30px;'>Prototype Model-I</h3>", unsafe_allow_html=True)

    # Load JSON lookups
    brand_mapping = load_dropdown_data("..\\Datasets\\brand_data.json", "brand_id", "brand_name")
    size_mapping = load_dropdown_data("..\\Datasets\\size_data.json", "size_id", "size_name")
    color_mapping = load_dropdown_data("..\\Datasets\\color_data.json", "color_id", "color")
    main_cat_mapping = load_simple_list("..\\Datasets\\main_categories.json")
    sub_cat_mapping = load_simple_list("..\\Datasets\\sub_categories.json")

    if 'dashboard_active' not in st.session_state:
        st.session_state.dashboard_active = False

    # --- ROW 1: INPUT SECTION ---
    with st.container(border=True):
        st.markdown("<div class='section-label'>⚙️ LISTING CONFIGURATION</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 2, 1.2, 1.5])
        with c1:
            name = st.text_input("Item Name", placeholder="e.g. Apple iPhone 13 Pro Max")
            brand_name = st.selectbox("Brand", list(brand_mapping.keys()))
        with c2:
            cat_0 = st.selectbox("Category Main", list(main_cat_mapping))
            cat_2 = st.selectbox("Category Sub", list(sub_cat_mapping))
        with c3:
            cond = st.selectbox("Current Condition", ["Excellent", "Good", "Fair", "Poor", "Very Poor"], index=0)
            ship = st.selectbox("Shipping", ["Buyer Pays", "Seller Pays"], index=1)
        with c4:
            sub1, sub2 = st.columns(2)
            with sub1: size_name = st.selectbox("Size", list(size_mapping.keys()))
            with sub2: color_name = st.selectbox("Color", list(color_mapping.keys()))
            use_research = st.toggle("Include Market Research", value=False)
            researched_price = None
            if use_research:
                override_price = True
                res_input = st.text_input("MSRP / Researched Price ($)", value="")
                if res_input.strip(): 
                    researched_price = float(res_input)

        cond_map = {"Excellent": 1, "Good": 2, "Fair": 3, "Poor": 4, "Very Poor": 5}
        cond_val, ship_val = cond_map.get(cond, 1), (1 if ship == "Seller Pays" else 0)
        brand_id, size_id, color_id = brand_mapping[brand_name], size_mapping[size_name], color_mapping[color_name]
        
        predict_clicked = st.button("RUN SYSTEM INFERENCE")

    if predict_clicked:
        st.session_state.dashboard_active = True

    if st.session_state.dashboard_active:
        # --- THE SPINNER ---
        with st.spinner("ANALYZING MARKET DATA & SIMULATING SCENARIOS..."):
            mm = load_system_model()
            price, (att, inter, conv) = mm.predict(
                                        name=name, 
                                        brand_name=brand_name, 
                                        brand_id=brand_id, 
                                        item_condition=cond_val, 
                                        shipper=ship_val,
                                        category_0=cat_0, 
                                        category_2=cat_2, 
                                        color_id=color_id, 
                                        size_id=size_id, 
                                        price_override=researched_price if use_research else None,
                                        researched_price=researched_price
                                    )
            
            # Heuristics for Feature Impact
            p_no_brand, _ = mm.predict(name=name, 
                                       brand_name="Unknown", 
                                       brand_id=0, 
                                       item_condition=cond_val, 
                                       shipper=ship_val, 
                                       category_0=cat_0, 
                                       category_2=cat_2, 
                                       color_id=color_id, 
                                       size_id=size_id
                                    )
            p_poor_cond, _ = mm.predict(name=name, 
                                        brand_name=brand_name, 
                                        brand_id=brand_id, 
                                        item_condition=5, 
                                        shipper=ship_val, 
                                        category_0=cat_0, 
                                        category_2=cat_2, 
                                        color_id=color_id, 
                                        size_id=size_id
                                    )
            p_buyer_ship, _ = mm.predict(name=name, 
                                         brand_name=brand_name, 
                                         brand_id=brand_id, 
                                         item_condition=cond_val, 
                                         shipper=0, category_0=cat_0, 
                                         category_2=cat_2, 
                                         color_id=color_id, 
                                         size_id=size_id
                                    )
            
            b_imp, c_imp, s_imp = (price - p_no_brand), (price - p_poor_cond), (price - p_buyer_ship)

            # --- ROW 2: PRIMARY ANALYSIS ---
            col_l, col_r = st.columns([1, 1.5])
            with col_l:
                with st.container(border=True):
                    st.markdown("<div class='section-label'>💰 PRICE ANALYSIS</div>", unsafe_allow_html=True)
                    

                    # Change price color if override is used
                    if override_price:
                        st.markdown(f"<h1 style='color: #696969; margin:0;'>${price:,.2f} (overridden)</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #696969;'>Optimal Range: ${max(0, price-32):,.2f} to ${price+32:,.2f}</p>", unsafe_allow_html=True)
                        price = researched_price  # Use the researched price for the distribution plot
                    else:
                        st.markdown(f"<h1 style='color: white; margin:0;'>${price:,.2f}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #BC13FE;'>Optimal Range: ${max(0, price-32):,.2f} to ${price+32:,.2f}</p>", unsafe_allow_html=True)

                    x = np.linspace(max(0, price - 100), price + 100, 100)
                    fig_p = px.area(x=x, y=norm.pdf(x, price, 32), height=180)
                    fig_p.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    fig_p.update_traces(fillcolor='rgba(188, 19, 254, 0.2)', line_color='#BC13FE')
                    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})

            with col_r:
                with st.container(border=True):
                    st.markdown("<div class='section-label'>BEHAVIORAL FLOW</div>", unsafe_allow_html=True)
                    fig_s = go.Figure(
                        data=[go.Sankey(node=dict(pad=15, thickness=20, label=["Audience", "Attracted", "Lost", "Interested", "Dropped", "Converted", "Abandoned"], 
                                                  color=["#FFFFFF", "#00FFC8", "#333333", "#BC13FE", "#333333", "#00FFC8", "#333333"]), 
                                                  link=dict(source=[0, 0, 1, 1, 3, 3], target=[1, 2, 3, 4, 5, 6], 
                                                  value=[att, 100-att, inter, att-inter, conv, inter-conv], color="rgba(255, 255, 255, 0.1)"))])
                    fig_s.update_layout(font_color="#FFFFFF", paper_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=0,r=0,t=10,b=10))
                    st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})

            # --- ROW 3: PROBABILITY & IMPACT ---
            col_met, col_imp = st.columns([1.5, 1])
            with col_met:
                with st.container(border=True):
                    st.markdown("<div class='section-label'>PROBABILITY METRICS</div>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    def donut(v, t, c):
                        f = go.Figure(go.Pie(values=[v, 100-v], hole=.8, marker_colors=[c, "#1A1A1A"], textinfo='none'))
                        f.update_layout(showlegend=False, height=180, margin=dict(t=30,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                        f.add_annotation(text=f"{v}%", showarrow=False, font=dict(size=22, color=c))
                        f.add_annotation(text=t, y=1.2, showarrow=False, font=dict(size=12, color="white"))
                        return f
                    with m1: st.plotly_chart(donut(att, "ATTRACTION", "#00FFC8"), use_container_width=True)
                    with m2: st.plotly_chart(donut(inter, "INTEREST", "#BC13FE"), use_container_width=True)
                    with m3: st.plotly_chart(donut(conv, "CONVERSION", "#00FFC8"), use_container_width=True)
            
            with col_imp:
                with st.container(border=True):
                    st.markdown("<div class='section-label'>🔍 FEATURE IMPACT ($)</div>", unsafe_allow_html=True)
                    fig_i = px.bar(x=[s_imp, c_imp, b_imp], y=["Free Ship", "Condition", "Brand"], orientation='h', color_discrete_sequence=['#BC13FE'])
                    fig_i.update_layout(height=180, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig_i.update_xaxes(showgrid=False, color="white")
                    st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar': False})

            # --- ROW 4: SCENARIOS & DEMAND ---
            with st.container(border=True):
                st.markdown("<div class='section-label'>🧪 CONDITION SCENARIO SIMULATION</div>", unsafe_allow_html=True)
                scenarios = []
                for i, lbl in enumerate(["Excellent", "Good", "Fair", "Poor", "Very Poor"], 1):
                    s_p, (s_a, s_i, s_c) = mm.predict(
                        name=name, 
                        brand_name=brand_name, 
                        brand_id=brand_id, 
                        item_condition=i, 
                        shipper=ship_val, 
                        category_0=cat_0, 
                        category_2=cat_2, 
                        color_id=color_id, 
                        size_id=size_id, 
                        # FIX: Pass the researched_price as the price_override 
                        # so the model evaluates probabilities at THAT specific price.
                        price_override=researched_price, 
                        researched_price=researched_price
                    )
                    scenarios.append(
                        {"Condition": f"{lbl} {'👈' if i==cond_val else ''}", "Suggested Price": f"${s_p:,.2f}", "Attraction": f"{s_a}%", "Interest": f"{s_i}%", "Conversion": f"{s_c}%"})
                st.table(pd.DataFrame(scenarios))

            with st.container(border=True):
                st.markdown("<div class='section-label'>PRICE SENSITIVITY</div>", unsafe_allow_html=True)
                t_prices = np.linspace(max(1, price * 0.5), price * 1.5, 15)
                sim_c = [mm.predict(name=name, 
                                    brand_name=brand_name, 
                                    brand_id=brand_id, 
                                    item_condition=cond_val, 
                                    shipper=ship_val, 
                                    category_0=cat_0, 
                                    category_2=cat_2, 
                                    color_id=color_id, 
                                    size_id=size_id, 
                                    price_override=p, 
                                    researched_price=researched_price)[1][2] for p in t_prices]
                fig_d = go.Figure(go.Scatter(x=t_prices, y=sim_c, mode='lines+markers', line=dict(color='#00FFC8', width=3, shape='spline')))
                fig_d.add_vline(x=price, line_dash="dash", line_color="#BC13FE")
                fig_d.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_d, use_container_width=True)

            # --- ROW 5: MARKET VIABILITY SUMMARY (RESTORED) ---
            with st.container(border=True):
                st.markdown("<div class='section-label'>📋 MARKET VIABILITY SUMMARY</div>", unsafe_allow_html=True)
                score = round((att * 0.6 + inter * 0.3 + conv * 0.1), 2)
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