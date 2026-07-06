import streamlit as st
from pymongo import MongoClient
from config import mongo_settings

client = MongoClient(mongo_settings.URI)
db = client[mongo_settings.DATABASE]

st.title("Mavan LLM Metrics")

# 1. Total calls
st.metric("Total LLM calls", db.llm_calls.count_documents({}))

# 2. Calls by capability
st.subheader("Calls by capability")
pipeline_cap = [
    {"$group": {"_id": "$capability", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]
st.bar_chart({r["_id"]: r["count"] for r in db.llm_calls.aggregate(pipeline_cap)})

# 3. Avg latency by capability
st.subheader("Avg latency (ms)")
pipeline_lat = [
    {"$group": {"_id": "$capability", "avg_ms": {"$avg": "$latency_ms"}}},
]
st.bar_chart({r["_id"]: round(r["avg_ms"], 1) for r in db.llm_calls.aggregate(pipeline_lat)})

# 4. Tokens per day
st.subheader("Output tokens per day")
pipeline_tok = [
    {"$group": {
        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
        "tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
    }},
    {"$sort": {"_id": 1}},
]
st.line_chart({r["_id"]: r["tokens"] for r in db.llm_calls.aggregate(pipeline_tok)})

# 5. Feedback ratio (like / dislike)
st.subheader("Message feedback")
likes = db.messages.count_documents({"reaction": "like"})
dislikes = db.messages.count_documents({"reaction": "dislike"})
st.bar_chart({"like": likes, "dislike": dislikes})
