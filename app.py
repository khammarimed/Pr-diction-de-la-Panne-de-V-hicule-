import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import plotly.graph_objects as go

# ========================================================
# -------------------- CONFIG PAGE ------------------------
# ========================================================
st.set_page_config(
    page_title="AI Vehicle Health Monitor 🚗", 
    layout="wide",
    page_icon="🚗"
)

# ========================================================
# --------------------- STYLE CSS ------------------------
# ========================================================

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    background-attachment: fixed;
}

/* Header avec effet glassmorphism */
.glass-header {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Cards modernes */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

/* Boutons modernes */
.stButton > button {
    background: linear-gradient(45deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(45deg, #8b5cf6, #3b82f6);
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(59, 130, 246, 0.4);
}

/* Inputs stylisés */
.stSelectbox, .stNumberInput {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Titres */
h1, h2, h3 {
    background: linear-gradient(45deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Container 3D */
.model-container {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Badges de statut */
.status-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 14px;
}

.safe-badge {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.danger-badge {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ========================================================
# ----------------------- HEADER --------------------------
# ========================================================
col_logo, col_title = st.columns([1, 3])

with col_logo:
    st.markdown('<div class="model-container">', unsafe_allow_html=True)
    # Placeholder pour le modèle 3D - à remplacer par une vraie intégration
    # Vous pouvez utiliser: 
    # 1. pyvista pour les modèles 3D locaux
    # 2. Plotly pour les visualisations 3D
    # 3. Ou intégrer un viewer externe
    
    # Exemple de visualisation 3D simple avec Plotly
    fig = go.Figure()
    
    # Créer une voiture simplifiée en 3D
    # Corps de la voiture
    fig.add_trace(go.Mesh3d(
        x=[-2, 2, 2, -2, -2, 2, 2, -2],
        y=[-1, -1, 1, 1, -1, -1, 1, 1],
        z=[0, 0, 0, 0, 1, 1, 1, 1],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color='#3b82f6',
        opacity=0.8,
        flatshading=True
    ))
    
    # Roues
    fig.add_trace(go.Scatter3d(
        x=[-1.5, 1.5, -1.5, 1.5],
        y=[-1.2, -1.2, 1.2, 1.2],
        z=[0.2, 0.2, 0.2, 0.2],
        mode='markers',
        marker=dict(
            size=12,
            color='#1f2937',
            symbol='circle'
        )
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        width=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_title:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown("<h1 style='margin: 0;'>🚗 AI Vehicle Health Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #cbd5e1; font-size: 18px;'>Prédiction intelligente des risques de panne dans les 30 prochains jours</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# -------------------- CHARGEMENT DATA --------------------
# ========================================================
df = pd.read_csv("vehicle_failure_dataset.csv")

# Nettoyage de la colonne cible
df["panne_dans_30_jours"] = pd.to_numeric(df["panne_dans_30_jours"], errors="coerce")
df["panne_dans_30_jours"] = df["panne_dans_30_jours"].fillna(df["panne_dans_30_jours"].mode()[0]).astype(int)

numeric_features = [
    "kilometrage_total",
    "km_moyen_jour",
    "nombre_vidanges",
    "nb_freins_changes",
    "jours_dernier_entretien",
    "temperature_moyenne"
]

categorical_features = [
    "type_vehicule",
    "type_route",
    "bruit_moteur",
    "vibration",
    "temoin_moteur_allume"
]

X = df[numeric_features + categorical_features]
y = df["panne_dans_30_jours"]

# ========================================================
# -------------------- PIPELINE ML ------------------------
# ========================================================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "pipeline_rf.pkl")

# ========================================================
# -------------------- FORMULAIRE INPUT -------------------
# ========================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📊 Saisie des paramètres du véhicule")

# Création de 3 colonnes pour le formulaire
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Informations générales**")
    type_vehicule = st.selectbox("Type de véhicule", df["type_vehicule"].dropna().unique(), key="type_vehicule")
    kilometrage_total = st.number_input("Kilométrage total (km)", 0, 500000, 50000, step=1000, key="km_total")
    km_moyen_jour = st.number_input("Km moyen / jour", 0, 1000, 50, step=10, key="km_jour")

with col2:
    st.markdown("**Historique d'entretien**")
    nombre_vidanges = st.number_input("Nombre de vidanges", 0, 50, 5, key="vidanges")
    nb_freins_changes = st.number_input("Freins changés", 0, 20, 1, key="freins")
    jours_dernier_entretien = st.number_input("Jours depuis dernier entretien", 0, 2000, 100, step=10, key="entretien")

with col3:
    st.markdown("**État actuel**")
    temperature_moyenne = st.slider("Température moyenne (°C)", -20, 60, 25, key="temp")
    type_route = st.selectbox("Type de route", df["type_route"].dropna().unique(), key="route")
    bruit_moteur = st.selectbox("Bruit moteur", df["bruit_moteur"].dropna().unique(), key="bruit")
    vibration = st.selectbox("Vibration", df["vibration"].dropna().unique(), key="vibration")
    temoin_moteur_allume = st.selectbox("Témoin moteur", df["temoin_moteur_allume"].dropna().unique(), key="temoin")

st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# ----------------------- PREDICTION ----------------------
# ========================================================
if st.button("🔍 Analyser le risque de panne", use_container_width=True):
    
    input_df = pd.DataFrame([{
        "type_vehicule": type_vehicule,
        "kilometrage_total": kilometrage_total,
        "km_moyen_jour": km_moyen_jour,
        "nombre_vidanges": nombre_vidanges,
        "nb_freins_changes": nb_freins_changes,
        "jours_dernier_entretien": jours_dernier_entretien,
        "temperature_moyenne": temperature_moyenne,
        "type_route": type_route,
        "bruit_moteur": bruit_moteur,
        "vibration": vibration,
        "temoin_moteur_allume": temoin_moteur_allume
    }])

    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    
    # Création des résultats
    col_result, col_stats = st.columns([2, 1])
    
    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if pred == 1:
            st.markdown('<div class="danger-badge status-badge">🚨 RISQUE ÉLEVÉ</div>', unsafe_allow_html=True)
            st.error("### ⚠️ Alerte Critique")
            st.markdown("**Le système détecte un risque élevé de panne dans les 30 prochains jours.**")
            st.progress(0.85)
            st.markdown("**Probabilité de panne:** {:.1f}%".format(proba[1]*100))
            
            # Recommandations
            with st.expander("🔧 Recommandations d'urgence"):
                st.markdown("""
                - ⚠️ **Arrêt immédiat recommandé** pour diagnostic complet
                - 🔧 **Contrôle moteur urgent** nécessaire
                - 📞 **Contacter un mécanicien** dans les 48h
                - 🚫 **Éviter les longs trajets** jusqu'à inspection
                """)
                
        else:
            st.markdown('<div class="safe-badge status-badge">✅ SÉCURISÉ</div>', unsafe_allow_html=True)
            st.success("### ✅ État Optimal")
            st.markdown("**Aucun risque de panne détecté — véhicule en bon état de fonctionnement.**")
            st.progress(0.15)
            st.markdown("**Probabilité de panne:** {:.1f}%".format(proba[1]*100))
            
            # Conseils d'entretien
            with st.expander("💡 Conseils de maintenance"):
                st.markdown("""
                - ✅ **Prochain entretien recommandé:** Dans {} jours
                - 🛢️ **Prochaine vidange:** À {} km
                - 🔍 **Contrôle périodique:** Recommandé dans 3 mois
                - 📊 **Surveillance continue:** Maintenir les bonnes pratiques de conduite
                """.format(max(0, 180 - jours_dernier_entretien), max(0, 10000 - (kilometrage_total % 10000))))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Statistiques")
        
        # Création d'un graphique radar simple
        fig_stats = go.Figure()
        
        categories = ['Kilométrage', 'Entretien', 'Moteur', 'Freins', 'Température']
        values = [
            min(kilometrage_total / 500000, 1),
            min(jours_dernier_entretien / 500, 1),
            0.8 if bruit_moteur == "normal" else 0.3,
            min(nb_freins_changes / 10, 1),
            abs(temperature_moyenne - 25) / 45
        ]
        
        fig_stats.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='#3b82f6', width=2),
            name='État actuel'
        ))
        
        fig_stats.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color='white')
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Détails des données
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Détails des paramètres analysés")
    
    col_details1, col_details2, col_details3 = st.columns(3)
    
    with col_details1:
        st.metric("Kilométrage total", f"{kilometrage_total:,} km", 
                 delta=None if kilometrage_total < 150000 else "Élevé")
        st.metric("Km journalier", f"{km_moyen_jour} km/jour")
    
    with col_details2:
        st.metric("Dernier entretien", f"{jours_dernier_entretien} jours", 
                 delta=None if jours_dernier_entretien < 180 else "Attention")
        st.metric("Nombre de vidanges", nombre_vidanges)
    
    with col_details3:
        st.metric("Température", f"{temperature_moyenne}°C",
                 delta=None if 15 <= temperature_moyenne <= 35 else "Extrême")
        st.metric("Changements freins", nb_freins_changes)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# -------------------- FOOTER -----------------------------
# ========================================================
st.markdown("---")
st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 14px;">', unsafe_allow_html=True)
st.markdown("**AI Vehicle Health Monitor** v2.0 | Système de prédiction de maintenance préventive")
st.markdown("© 2024 - Tous droits réservés")
st.markdown("</div>", unsafe_allow_html=True)