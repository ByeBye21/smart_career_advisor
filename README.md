# smart_career_advisor

Yapay zekâ destekli bu uygulama, kariyerle ilgili sorularınıza önceki konuşmaları da dikkate alarak kişisel tavsiyeler sunar. LangGraph, Chroma ve Streamlit ile geliştirilmiştir. HuggingFace gömlemeleri ve GPT-3.5-Turbo kullanır. 

Veri seti: [CareerVillage @ Kaggle](https://www.kaggle.com/competitions/data-science-for-good-careervillage)

# Not

OpenAI API anahtarınızı kod içindeki api_key alanına girmeniz gerekir.

# Kurulum

```bash
git clone https://github.com/kullanici-adi/smart-career-advisor.git
cd smart-career-advisor
pyhton rebuild_chroma.py
python run.py
