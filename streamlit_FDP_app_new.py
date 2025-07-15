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
    "A11:Subject Knowledge": [
        "Advanced Subject Masterclasses", "Emerging Interdisciplinary Trends", 
        "AI Applications in Discipline", "Future Skills in Domain", 
        "Deep Dive Conceptual Workshops", "Cutting-edge Innovations"
    ],
    "A12:Teaching Methods – Theoretical Knowledge": [
        "AI-enhanced Teaching Strategies", "Flipped & Hybrid Classrooms", 
        "Socratic & Case-based Learning", "Interactive Lecture Design", 
        "Learning Analytics for Theory", "Digital Pedagogy Essentials"
    ],
    "A13:Teaching Methods – Practical Application": [
        "Project-based & Experiential Learning", "AI Labs & Virtual Simulations", 
        "Industry-aligned Practical Pedagogy", "Blended Hands-on Approaches", 
        "Design Thinking in Curriculum", "Immersive Tech for Practical Learning"
    ],
    "A14:Information Literacy and Management": [
        "Data Mining for Faculty", "AI Tools for Information Management", 
        "Reference Managers & Literature Maps", "Smart Digital Libraries", 
        "Evidence-based Information Use", "Scholarly Database Training"
    ],
    "A15:Languages": [
        "Scholarly Writing & Technical English", "AI-based Language Tools", 
        "Academic Presentation Skills", "Discipline-specific Communication", 
        "Language Models for Research", "Multilingual Digital Tools"
    ],
    "A16:Academic Literacy and Numeracy": [
        "Quantitative Reasoning in Academia", "AI for Research Methods", 
        "Data Interpretation Skills", "Academic Integrity & Writing", 
        "Survey Design & Analysis", "Numeracy in Social Sciences"
    ],

    "A21:Analyzing": [
        "AI for Critical Analysis", "Root Cause & Data-driven Analysis", 
        "Analytical Thinking Labs", "Systems Thinking Approaches", 
        "Data Visualization & Interpretation", "Strategic Problem Dissection"
    ],
    "A22:Synthesizing": [
        "Synthesis & Concept Mapping", "Interdisciplinary Integration", 
        "AI to Discover Connections", "Thematic Reviews & Meta-analysis", 
        "Big Picture Thinking", "Synthesizing Evidence & Policy"
    ],
    "A23:Critical Thinking": [
        "Debate & Argumentation Workshops", "Logic & Reasoning Bootcamps", 
        "Reflective Inquiry with AI", "Case & Scenario Analysis", 
        "Bias & Fallacy Awareness", "Building Intellectual Autonomy"
    ],
    "A24:Evaluating": [
        "Outcome Assessment Tools", "AI-driven Rubric Design", 
        "Evaluating Impact & ROI", "Peer Review & Feedback Loops", 
        "Digital Assessment Platforms", "Evaluation in Accreditation Contexts"
    ],
    "A25:Problem Solving": [
        "AI for Decision Support", "Creative Problem Solving Frameworks", 
        "Hackathons & Solution Labs", "Scenario Planning", 
        "Collaborative Problem Solving", "Complex Systems Solutions"
    ],

    "A31:Inquiring Mind": [
        "Cultivating Curiosity", "Research Question Design", 
        "AI Tools to Explore Ideas", "Inquiry-based Teaching", 
        "Creative Thinking Labs", "Exploratory Learning Pathways"
    ],
    "A32:Intellectual Insight": [
        "Advanced Conceptual Frameworks", "Strategic Scenario Building", 
        "Abstract Modelling with AI", "Futures & Foresight", 
        "Analytical Depth Workshops", "Strategic Research Visioning"
    ],
    "A33:Innovation": [
        "Innovation & Design Sprints", "AI-driven Creativity", 
        "Startup Ecosystems for Faculty", "Patents & Prototyping", 
        "Entrepreneurial Mindset", "EdTech Innovations"
    ],
    "A34:Argument Construction": [
        "Evidence-based Argumentation", "Position Papers with AI Support", 
        "Ethics in Debates", "Structuring Research Arguments", 
        "Critical Dialogues", "Policy Argument Labs"
    ],

    "B11:Enthusiasm": [
        "Gamification & Motivation", "Fostering Passion in Teaching", 
        "AI Tools to Engage Learners", "Positive Pedagogy Practices", 
        "Energy Management", "Joyful Learning Approaches"
    ],
    "B12:Perseverance": [
        "Building Academic Resilience", "Overcoming Teaching Challenges", 
        "Goal Mapping for Long-term Impact", "Grit & Growth Mindset", 
        "Handling Failures in Research", "Sustaining Motivation"
    ],
    "B13:Integrity": [
        "Academic & Research Ethics", "Plagiarism Tools & AI Checkers", 
        "Responsible Data Use", "Integrity in Publications", 
        "Moral Reasoning in Teaching", "AI Bias & Ethics"
    ],
    "B14:Responsibility": [
        "Owning the Learning Process", "Self-directed Faculty Development", 
        "Portfolio-driven Growth", "Accountability in Projects", 
        "Ethical Leadership", "Service Commitments"
    ],

    "B21:Preparation and Prioritization": [
        "Data-informed Lesson Planning", "Timeboxing for Faculty", 
        "AI Tools for Planning", "Strategic Prioritization", 
        "Curriculum Blueprints", "Outcome-aligned Planning"
    ],
    "B22:Commitment to Teaching": [
        "Professional Accountability", "Aligning Personal & Institutional Goals", 
        "Reflective Teaching Practices", "Long-term Teaching Strategies", 
        "Continuous Engagement Models", "Leveraging AI for Improvement"
    ],
    "B23:Time Management": [
        "Digital Time Management Tools", "Efficient Academic Workflows", 
        "AI-based Scheduling", "Deadline Management Strategies", 
        "Balanced Research & Teaching", "Overcoming Procrastination"
    ],
    "B24:Responsiveness to Change": [
        "Change Management Frameworks", "Adapting to EdTech & AI", 
        "Risk-taking in Pedagogy", "Flexible Curriculum Approaches", 
        "Navigating Policy Shifts", "Scenario-based Adaptability"
    ],

    "B31:Continuing Professional Development": [
        "Career Progression Paths", "Certifications in AI & EdTech", 
        "Global Fellowship Opportunities", "Showcasing in Digital Portfolios", 
        "Professional Learning Networks", "Research Leadership"
    ],
    "B32:Student Feedback": [
        "Collecting & Acting on Feedback", "AI Sentiment Analysis", 
        "Closing the Feedback Loop", "Designing Effective Surveys", 
        "Feedback for Curriculum Tuning", "Reflective Student Dialogues"
    ],
    "B33:Networking": [
        "Building Inter-institutional Networks", "Collaborative Platforms", 
        "AI-driven Professional Connects", "Conference Ecosystems", 
        "Online Academic Communities", "Global Partnerships"
    ],
    "B34:Reputation and Esteem": [
        "Thought Leadership via Digital Media", "Public Speaking Excellence", 
        "AI-assisted Profile Building", "Awards & Recognition Prep", 
        "Research Visibility", "Media Engagement Strategies"
    ],

    "C11:Ethics, Principles, and Sustainability": [
        "Ethics in AI & Research", "Green Campuses & Teaching", 
        "Sustainable Development Goals in Curriculum", "Responsible Innovations", 
        "Equity-focused Pedagogy", "AI for Social Good"
    ],
    "C12:Intellectual Property Rights and Copyright": [
        "IPR & Patents Filing", "Copyright Compliance", 
        "AI-generated Content Ethics", "Creative Commons Licenses", 
        "Fair Use in Academia", "Data Sharing Agreements"
    ],
    "C21:Research Strategy": [
        "Aligning with Institutional Missions", "Data-driven Strategic Planning", 
        "AI to Spot Research Gaps", "Collaborative Strategy Labs", 
        "Foresight-driven Research", "NEP 2020 & Beyond"
    ],
    "C31:Income and Funding Generation": [
        "Grant Proposal Writing", "AI Tools for Funding Match", 
        "Budget Planning Workshops", "CSR & Industry Funding", 
        "International Grants", "Revenue Diversification Strategies"
    ],

    "D11:Team Working": [
        "High-performing Academic Teams", "Collaborative Research Tools", 
        "AI for Team Dynamics", "Shared Vision Development", 
        "Cross-functional Synergies", "Joint Faculty Development"
    ],
    "D12:People Management": [
        "Delegation & Empowerment", "Negotiating & Influencing", 
        "Conflict Resolution", "Mentoring Diverse Teams", 
        "AI Tools for HR & Planning", "Building Psychological Safety"
    ],
    "D13:Supervision": [
        "Effective Research Supervision", "AI for Plagiarism & Review", 
        "Mentored Assessments", "Outcome-driven Project Management", 
        "Guided Inquiry Techniques", "Tracking Progress Digitally"
    ],
    "D14:Mentoring": [
        "Structured Mentorship Programs", "Skill Transfer Models", 
        "Reverse Mentoring", "Inclusive Mentoring Approaches", 
        "Mentoring for Innovation", "Longitudinal Faculty Mentoring"
    ],
    "D15:Influence and Leadership": [
        "Institutional Leadership Labs", "Strategic Influence Models", 
        "Policy Advocacy & AI", "Community Engagement", 
        "Ethical Leadership in AI Era", "Vision & Legacy Building"
    ],
    "D16:Collaboration": [
        "Joint Research Ventures", "Industry-academia Connects", 
        "Digital Collaboration Tools", "Global Research Consortia", 
        "Virtual International Teams", "Collaborative Publishing"
    ],
    "D17:Equality and Diversity": [
        "Inclusive Pedagogies", "Diversity Sensitization Labs", 
        "Equity Audits", "Policy for Inclusion", 
        "AI & Bias Awareness", "Universal Design for Learning"
    ],

    "D21:Communication Methods": [
        "AI-powered Communication Platforms", "Public Engagement", 
        "Academic Storytelling", "Policy Brief Writing", 
        "Science Communication", "Stakeholder Reporting"
    ],
    "D22:Communication Media": [
        "Digital Outreach Strategies", "Webinars & MOOCs Design", 
        "Social Media for Academia", "Video Lecturing Best Practices", 
        "Podcasting Academic Content", "AI in Media Production"
    ],
    "D23:Publication": [
        "Publishing in High Impact Journals", "Open Access Strategies", 
        "AI for Manuscript Editing", "Conference Paper Excellence", 
        "Ethics in Publication", "Boosting Research Visibility"
    ],

    "D31:Teaching": [
        "AI-enhanced Teaching Aids", "Capstone & UG Research Projects", 
        "Industry-driven Seminars", "Outcome-based Education", 
        "Student-centred Learning", "Hybrid Teaching Environments"
    ],
    "D32:Policy": [
        "NBA/NAAC & Global Benchmarks", "Policy Impact on Teaching", 
        "Digital Policies in Education", "NEP 2020 Implementation", 
        "AI in Education Policies", "Accreditation-readiness"
    ]
}
# Main Title
st.title("🎯 Comprehensive TNA FDP Recommender")
st.write(
    "Highlights top subdomains, and applies smart rules to suggest FDPs."
)

# Sidebar inputs
#st.sidebar.header("📝 Enter TNA Scores (1-10)")
#scores = {k: st.sidebar.slider(k, 1.0, 10.0, 1.0, step=0.01) for k in fdp_topic_map.keys()}

use_slider = st.sidebar.radio("Select input mode:", ("Slider", "Manual Entry"))

scores = {}
for k in fdp_topic_map.keys():
    if use_slider == "Slider":
        scores[k] = st.sidebar.slider(k, 1.0, 10.0, 1.0, step=0.01)
    else:
        scores[k] = st.sidebar.number_input(k, min_value=1.0, max_value=10.0, value=1.0, step=0.01)

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
#st.subheader("🔍 Prediction Results")
#if prediction == 1:
    #st.success(f"High FDP Need: ✅ YES (probability: {probability:.2%})")
#else:
    #st.info(f"High FDP Need: 🚫 NO (probability: {probability:.2%})")

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
#st.subheader("🚀 Feature Importances in the Model")
#if hasattr(classifier, "feature_importances_"):
    #feature_names = list(fdp_topic_map.keys())
    #importances = classifier.feature_importances_
    #fig, ax = plt.subplots(figsize=(8,6))
    #pd.Series(importances, index=feature_names).sort_values().plot(kind="barh", ax=ax, color="teal")
    #st.pyplot(fig)
#else:
    #st.warning("This model does not provide feature importances.")
