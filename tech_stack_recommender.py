# Project 3: AI Recommendation Logic
# DecodeLabs - Industrial Training Kit
# Capstone: Tech Stack Recommender
# Goal: User apni skills de, system usko best matching job roles suggest kare

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# STEP 1: INPUT - Dataset load karo (job roles + unki skills)
# ---------------------------------------------------------
df = pd.read_csv("raw_skills.csv")
print("Available Job Roles:")
print(df["job_role"].tolist())
print("-" * 60)

# ---------------------------------------------------------
# STEP 2: Ingestion - User apni skills de (kam se kam 3)
# ---------------------------------------------------------
user_skills = ["Python", "Cloud", "Automation"]   # Yahan apni skills likh sakte ho
user_profile = " ".join(user_skills)

print("User Skills:", user_skills)
print("-" * 60)

# ---------------------------------------------------------
# STEP 3: PROCESS - Vector Mapping using TF-IDF
# ---------------------------------------------------------
# Har job role ki skills + user ki skills ko ek hi vocabulary mein daalte hain
all_texts = df["skills"].tolist() + [user_profile]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Aakhri row user ka vector hai, baaki sab job roles ke vectors hain
job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

# ---------------------------------------------------------
# STEP 4: Scoring - Cosine Similarity nikaalo
# ---------------------------------------------------------
similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

# ---------------------------------------------------------
# STEP 5: Sorting & Filtering - Top 3 matches nikaalo
# ---------------------------------------------------------
df["similarity_score"] = similarity_scores
top_matches = df.sort_values(by="similarity_score", ascending=False).head(3)

# ---------------------------------------------------------
# OUTPUT - Result dikhana
# ---------------------------------------------------------
print("Top 3 Recommended Career Paths:\n")
for rank, (_, row) in enumerate(top_matches.iterrows(), start=1):
    match_percent = round(row["similarity_score"] * 100, 2)
    print(f"{rank}. {row['job_role']} — Match: {match_percent}%")
    print(f"   Required Skills: {row['skills']}\n")
