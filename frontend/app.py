import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

API_URL = "https://real-estate-price-prediction-api-e03q.onrender.com"

LOCATIONS = [
    "Whitefield", "Sarjapur  Road", "Electronic City", "Marathahalli", 
    "Raja Rajeshwari Nagar", "Haralur Road", "Hennur Road", "Bannerghatta Road",
    "Uttarahalli", "Thanisandra", "Electronic City Phase II", "Hebbal",
    "7th Phase JP Nagar", "Yelahanka", "Kanakpura Road", "KR Puram", "Sarjapur",
    "Rajaji Nagar", "Kasavanhalli", "Bellandur", "Begur Road", "Banashankari",
    "Kothanur", "Hormavu", "Harlur", "Akshaya Nagar", "Jakkur",
    "Electronics City Phase 1", "Varthur", "Chandapura", "HSR Layout", "Hennur",
    "Ramamurthy Nagar", "Ramagondanahalli", "Kaggadasapura", "Kundalahalli",
    "Koramangala", "Hulimavu", "Budigere", "Hoodi", "Malleshwaram", "Hegde Nagar",
    "8th Phase JP Nagar", "Gottigere", "JP Nagar", "Yeshwanthpur", "Channasandra",
    "Bisuvanahalli", "Vittasandra", "Indira Nagar", "Vijayanagar", "Kengeri",
    "Brookefield", "Sahakara Nagar", "Hosa Road", "Old Airport Road", "Bommasandra",
    "Balagere", "Green Glen Layout", "Old Madras Road", "Rachenahalli", "Panathur",
    "Kudlu Gate", "Thigalarapalya", "Ambedkar Nagar", "Jigani", "Yelahanka New Town",
    "Talaghattapura", "Mysore Road", "Kadugodi", "Frazer Town", "Dodda Nekkundi",
    "Devanahalli", "Kanakapura", "Attibele", "Anekal", "Lakshminarayana Pura",
    "Nagarbhavi", "Ananth Nagar", "5th Phase JP Nagar", "TC Palaya", "CV Raman Nagar",
    "Kengeri Satellite Town", "Kudlu", "Jalahalli", "Subramanyapura", "Bhoganhalli",
    "Doddathoguru", "Kalena Agrahara", "Horamavu Agara", "Vidyaranyapura",
    "BTM 2nd Stage", "Hebbal Kempapura", "Hosur Road", "Horamavu Banaswadi",
    "Domlur", "Mahadevpura", "Tumkur Road"
]

AREA_TYPES = ["Super built-up  Area", "Built-up  Area", "Plot  Area"]

MODELS = {
    "🤖 All Models (Compare)": "all",
    "📈 Linear Regression": "linear_regression",
    "🌳 Decision Tree": "decision_tree",
    "🌲 Random Forest": "random_forest",
    "🚀 Gradient Boosting (Best)": "gradient_boosting"
}

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 700;
    }
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 0.5rem 0;
    }
    .lr-card { border-color: #3498db; }
    .dt-card { border-color: #27ae60; }
    .rf-card { border-color: #9b59b6; }
    .gb-card { border-color: #e67e22; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 Bangalore Real Estate Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict house prices using Machine Learning models trained on 6,000+ properties</p>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Property Details")
    
    location = st.selectbox("📍 Location", sorted(LOCATIONS), index=sorted(LOCATIONS).index("Whitefield"))
    area_type = st.selectbox("🏗️ Area Type", AREA_TYPES)
    
    col_a, col_b = st.columns(2)
    with col_a:
        total_sqft = st.number_input("📐 Total Sqft", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
    
    with col_b:
        bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        balcony = st.selectbox("🌅 Balconies", [0, 1, 2, 3], index=1)
    
    ready_to_move = st.checkbox("✅ Ready to Move", value=True)

with col2:
    st.subheader("🤖 Model Selection")
    
    selected_model = st.selectbox("Select Prediction Model", list(MODELS.keys()), index=0)
    model_value = MODELS[selected_model]
    
    st.info("""
    **Model Guide:**
    - **All Models**: Compare predictions from all 4 models
    - **Linear Regression**: Simple, interpretable baseline
    - **Decision Tree**: Captures non-linear patterns
    - **Random Forest**: Ensemble for better accuracy
    - **Gradient Boosting**: Best performing model (95% R²)
    """)

st.markdown("---")

if st.button(" Predict Price", type="primary", use_container_width=True):
    
    property_data = {
        "bath": float(bath),
        "balcony": float(balcony),
        "total_sqft": float(total_sqft),
        "bhk": int(bhk),
        "location": location,
        "area_type": area_type,
        "ready_to_move": ready_to_move
    }
    
    with st.spinner("🔄 Getting predictions..."):
        try:
            if model_value == "all":
                results = []
                model_endpoints = {
                    "Linear Regression": "linear_regression",
                    "Decision Tree": "decision_tree",
                    "Random Forest": "random_forest",
                    "Gradient Boosting": "gradient_boosting"
                }
                
                for name, endpoint in model_endpoints.items():
                    response = requests.post(f"{API_URL}/predict/{endpoint}", json=property_data)
                    if response.status_code == 200:
                        data = response.json()
                        results.append({
                            "Model": name,
                            "Predicted Price": f"₹ {data['predicted_price']:.2f} Lakhs",
                            "Price/Sqft": f"₹ {data['price_per_sqft']:.0f}",
                            "raw_price": data['predicted_price']
                        })
                
                if results:
                    st.success("✅ Predictions from all models!")
                    
                    avg_price = sum(r['raw_price'] for r in results) / len(results)
                    best_model = max(results, key=lambda x: x['raw_price'])
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Average Predicted Price</div>
                        <div class="price-value">₹ {avg_price:.2f} Lakhs</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">Range: ₹ {min(r['raw_price'] for r in results):.2f} - ₹ {max(r['raw_price'] for r in results):.2f} Lakhs</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📊 Model Comparison")
                    
                    col_lr, col_dt, col_rf, col_gb = st.columns(4)
                    
                    cards = [
                        (col_lr, results[0], "lr-card", "#3498db", "📈"),
                        (col_dt, results[1], "dt-card", "#27ae60", "🌳"),
                        (col_rf, results[2], "rf-card", "#9b59b6", "🌲"),
                        (col_gb, results[3], "gb-card", "#e67e22", "🚀")
                    ]
                    
                    for col, result, card_class, color, icon in cards:
                        with col:
                            st.markdown(f"""
                            <div class="model-card {card_class}">
                                <div style="font-size: 1.5rem;">{icon}</div>
                                <div style="font-weight: 600; color: {color};">{result['Model']}</div>
                                <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">{result['Predicted Price']}</div>
                                <div style="color: #666;">{result['Price/Sqft']}/sqft</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    df = pd.DataFrame([{
                        "Model": r['Model'],
                        "Price (Lakhs)": r['raw_price'],
                        "Price/Sqft (₹)": float(r['Price/Sqft'].replace('₹ ', '').replace(',', ''))
                    } for r in results])
                    
                    st.subheader("📈 Price Comparison Chart")
                    st.bar_chart(df.set_index('Model')['Price (Lakhs)'])
                    
            else:
                response = requests.post(f"{API_URL}/predict/{model_value}", json=property_data)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"✅ Prediction using {data['model_used'].replace('_', ' ').title()}")
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted Price</div>
                        <div class="price-value">₹ {data['predicted_price']:.2f} Lakhs</div>
                        <div style="font-size: 1.1rem; margin-top: 0.5rem;">₹ {data['price_per_sqft']:.0f} per sqft</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("🏷️ Total Price", f"₹ {data['predicted_price'] * 100000:,.0f}")
                    col_b.metric("📐 Price/Sqft", f"₹ {data['price_per_sqft']:,.0f}")
                    col_c.metric("🤖 Model", data['model_used'].replace('_', ' ').title())
                else:
                    st.error(f"❌ API Error: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the backend is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Built with ❤️ using Streamlit & FastAPI by <a href="https://www.google.com/search?q=omkar+chebale" target="_blank">Omkar Chebale</a> | Models trained on Bangalore House Price Dataset</p>
    <p>🔗 <a href="https://github.com/Chebaleomkar/real-estate-price-prediction" target="_blank">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
