# LexiQ – AI-Powered Legal Document Intelligence System

**Built by Syed Sajjad Hussain**

LexiQ helps you **understand legal documents faster, safer, and with clarity**.  
Upload contracts, ask questions in plain English, get answers with risk flags and citations.

---

## 🎯 The Problem

- Contracts are long, dense, and full of hidden risks  
- Manual review is slow and expensive  
- Missing a critical clause can cost you

---

## 💡 The Solution

Upload a PDF → Ask questions → Get clear, explainable answers with:
- 📌 Source citations and page references
- 🚨 Risk flags (High / Medium / Low)
- 🔐 Local-first processing – your data stays private

---

## ⚡ Quick Start

```bash
# Setup
git clone <repo-url>
cd lexiq
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .

# Configure
echo "OPENAI_API_KEY=sk-your-key" > .env

# Run
streamlit run app.py
```

---

## 💬 Try Asking

- *"What are the main risks?"*
- *"How can I terminate this contract?"*
- *"Any liability caps?"*

---

## 🛠️ Tech Stack

**Streamlit** • **LangGraph** • **ChromaDB** • **OpenAI GPT-4o** • **PyMuPDF**

---

## ⚠️ Disclaimer

For educational purposes only. Not legal advice. Always consult an attorney.

---

## 📄 More Info

- [Project Structure](STRUCTURE.md)
- [License](LICENSE) (MIT)

---

<p align="center">
  <strong>LexiQ</strong> • Local-first • Privacy-conscious • Risk-aware
</p>
