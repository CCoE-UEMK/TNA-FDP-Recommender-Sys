import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FDP TNA Recommender", page_icon="🎯", layout="wide")

# Cached loading of model
@st.cache_resource
def load_model():
    with open("tna_model.pkl", "rb") as f:
        return pickle.load(f)

classifier = load_model()

# FDP topic mapping
fdp_topic_map = {
    "A11:Subject Knowledge": ["Advanced Subject Knowledge", "Emerging Trends", "Deep Dive Seminars"],
    "A12:Teaching Methods – Theoretical Knowledge": ["Innovative Teaching Methods", "Case-Based Learning", "Flipped Classrooms"],
    "A13:Teaching Methods – Practical Application": ["Practical Lab Pedagogy", "Hands-on Workshops", "Industry Practices"],
    "A14:Information Literacy and Management": ["Data Mining Tools", "AI in Info Management", "Reference Managers"],
    "A15:Languages ": ["Technical English", "Scholarly Communication", "Domain-specific Language"],
    "A16:Academic Literacy and Numeracy": ["Academic Writing", "Quant Techniques", "Research Methods"],

    "A21:Analyzing": ["Critical Analysis Methods", "Problem Structuring", "Evidence-based Discussions"],
    "A22:Synthesizing": ["Synthesis Techniques", "Cross-disciplinary Ideas", "Thematic Integration"],
    "A23:Critical Thinking": ["Debate & Argumentation", "Logical Reasoning", "Reflective Thinking"],
    "A24:Evaluating": ["Outcome Assessment", "Evaluation Metrics", "Rubrics Design"],
    "A25:Problem Solving": ["Problem Solving Frameworks", "Creative Ideation", "Solution Modelling"],

    "A31:Inquiring Mind": ["Curiosity Workshops", "Inquiry Learning", "Research Questions"],
    "A32:Intellectual Insight": ["Advanced Conceptual Thinking", "Abstract Models", "Strategic Insight"],
    "A33:Innovation": ["Innovation Labs", "Design Bootcamps", "Startup Mindsets"],
    "A34:Argument Construction": ["Argument Building", "Position Papers", "Ethical Dilemmas"],

    "B11:Enthusiasm": ["Motivation Strategies", "Teaching Passion", "Gamified Learning"],
    "B12:Perseverance": ["Goal Setting", "Resilience", "Professional Perseverance"],
    "B13:Integrity": ["Integrity in Research & Teaching", "Plagiarism Awareness", "Academic Honesty"],
    "B14:Responsibility": ["Personal Responsibility", "Ownership Projects", "Self-Directed Learning"],

    "B21:Preparation and Prioritization": ["Effective Lesson Planning", "Teaching Blueprints", "Prioritization"],
    "B22:Commitment to Teaching": ["Commitment Workshops", "Professional Accountability", "Goal Mapping"],
    "B23:Time Management": ["Time Management Tools", "Scheduling", "Deadline Adherence"],
    "B24:Responsiveness to Change": ["Change Management", "Adaptive Teaching", "Risk Awareness"],

    "B31:Continuing Professional Development": ["Career Advancement", "Certifications", "Digital Portfolios"],
    "B32:Student Feedback": ["Student Feedback", "Continuous Improvement", "Feedback Loops"],
    "B33:Networking": ["Networking Forums", "Peer Learning", "Professional Societies"],
    "B34:Reputation and Esteem": ["Thought Leadership", "Public Speaking", "Reputation Building"],

    "C11:Ethics, Principles, and Sustainability": ["Ethics in Teaching", "Sustainability Practices", "Responsible Research"],
    "C12:Intellectual Property Rights and Copyright": ["IPR & Patents", "Copyright for Faculty", "Data Sharing Norms"],
    "C21:Research Strategy": ["Institutional Strategies", "Aligning NEP 2020", "Strategic Planning"],
    "C31:Income and Funding Generation": ["Funding Proposals", "Grant Management", "Budget Allocations"],

    "D11:Team Working": ["Collaborative Research", "Team Dynamics", "Cross-functional Teams"],
    "D12:People Management": ["Delegation Skills", "Negotiating Deadlines", "People Coordination"],
    "D13:Supervision": ["Effective Supervision", "UG Project Management", "Mentored Assessments"],
    "D14:Mentoring": ["Mentoring Juniors", "Skill Transfer", "Guided Learning"],
    "D15:Influence and Leadership": ["Influence Building", "Institutional Leadership", "Policy Influence"],
    "D16:Collaboration": ["Research Collaborations", "Consortium Projects", "Joint Publications"],
    "D17:Equality and Diversity": ["Inclusive Teaching", "Diversity Sensitization", "Equity Frameworks"],

    "D21:Communication Methods": ["Communication Skills", "Academic Presentations", "Stakeholder Reports"],
    "D22:Communication Media": ["Digital Media Outreach", "Webinars & MOOCs", "Online Engagement"],
    "D23:Publication": ["Publishing High Impact", "Grant Writing", "Conference Papers"],

    "D31:Teaching": ["Innovative Teaching Aids", "UG Research Projects", "Seminars"],
    "D32:Policy": ["Policy Understanding", "NBA/NAAC Prep", "Education Regulations"]
}

# Main Title
st.title("🎯 Comprehensive TNA FDP Recommender")
st.write(
    "Predicts high FDP need, highlights top subdomains, and applies smart rules to suggest FDPs."
)

# Sidebar inputs
st.sidebar.header("📝 Enter TNA Scores (1-10)")
scores = {k: st.sidebar.slider(k, 1.0, 10.0, 5.0, step=0.1) for k in fdp_topic_map.keys()}

# Prepare feature vector
X = np.array(list(scores.values())).reshape(1, -1)

# Predictions
prediction = classifier.predict(X)[0]
probability = classifier.predict_proba(X)[0][1]

# Top 3 subdomains
top_subdomains = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

# Rule engine
rule_based_fdps = []
triggered_rules = []

if scores["A11:Subject Knowledge"] > 8 and scores["A21:Analyzing"] > 7:
    rule_based_fdps.append("Advanced interdisciplinary FDPs combining subject expertise and cognitive challenges")
    triggered_rules.append("A11:Subject Knowledge > 8 & A21:Analyzing > 7")

if scores["A31:Inquiring Mind"] > 8:
    rule_based_fdps.append("Creative pedagogy, design thinking, gamification workshops")
    triggered_rules.append("A31:Inquiring Mind > 8")

if scores["B21:Preparation and Prioritization"] > 7 and scores["B31:Continuing Professional Development"] > 7:
    rule_based_fdps.append("Time management, career progression, leadership skills")
    triggered_rules.append("B21:Preparation and Prioritization > 7 & B31:Continuing Professional Development > 7")

if scores["C21:Research Strategy"] > 8 or scores["C31:Income and Funding Generation"] > 8:
    rule_based_fdps.append("Research proposal writing, grants & funding management")
    triggered_rules.append("C21:Research Strategy > 8 or C31:Income and Funding Generation > 8")

if scores["D11:Team Working"] > 8 and scores["D21:Communication Methods"] > 7:
    rule_based_fdps.append("Collaboration, communication, stakeholder negotiation")
    triggered_rules.append("D11:Team Working > 8 & D21:Communication Methods > 7")

if scores["D31:Teaching"] > 7:
    rule_based_fdps.append("Public engagement, impact creation, industry partnerships")
    triggered_rules.append("D31:Teaching > 7")

if scores["B11:Enthusiasm"] > 8 and scores["A31:Inquiring Mind"] > 7:
    rule_based_fdps.append("Motivation, resilience, innovative teaching FDPs")
    triggered_rules.append("B11:Enthusiasm > 8 & A31:Inquiring Mind > 7")

if scores["C11:Ethics, Principles, and Sustainability"] > 8 and scores["B11:Enthusiasm"] > 7:
    rule_based_fdps.append("Ethics, professional integrity, mentoring workshops")
    triggered_rules.append("C11:Ethics, Principles, and Sustainability > 8 & B11:Enthusiasm > 7")

# Any two domains > 8
domains_over_8 = sum(1 for v in scores.values() if v > 8)
if domains_over_8 >= 2:
    rule_based_fdps.append("Integrated FDPs covering teaching + research + engagement")
    triggered_rules.append("Any two domains > 8")

# Display prediction
st.subheader("🔍 Prediction Results")
if prediction == 1:
    st.success(f"High FDP Need: ✅ YES (probability: {probability:.2%})")
else:
    st.info(f"High FDP Need: 🚫 NO (probability: {probability:.2%})")

# Top 3 focus areas
st.subheader("🏆 Top 3 Focus Areas with Suggested FDP Topics")
for subdomain, score in top_subdomains:
    st.markdown(f"### 📌 {subdomain} (Score: {score})")
    st.write("Recommended FDP Topics:")
    for topic in fdp_topic_map[subdomain]:
        st.markdown(f"- {topic}")

# Rule-based FDPs
if rule_based_fdps:
    st.subheader("🧠 Smart Rule-based FDP Recommendations")
    for fdp, rule in zip(rule_based_fdps, triggered_rules):
        st.markdown(f"✅ **{fdp}** _(triggered by: {rule})_")
else:
    st.info("No special rules triggered. Adjust sliders to explore tailored FDPs.")

# Feature importances
st.subheader("🚀 Feature Importances in the Model")
if hasattr(classifier, "feature_importances_"):
    feature_names = list(fdp_topic_map.keys())
    importances = classifier.feature_importances_
    fig, ax = plt.subplots(figsize=(8,6))
    pd.Series(importances, index=feature_names).sort_values().plot(kind="barh", ax=ax, color="teal")
    st.pyplot(fig)
else:
    st.warning("This model does not provide feature importances.")
