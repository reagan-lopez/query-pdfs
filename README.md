# Query PDFs using Google GeminiPro
## Steps:
- Clone the repo and `cd` into it.
- Create a virtual environment. `conda create -p venv python=3.10`
- Activate the environment. `conda activate ./venv`
- Install required packages. `python install -r requirements.txt`
- Get a GOOGLE_API_KEY for Gemini Pro from https://ai.google.dev/.
  - Save it in a `.env` file as `GOOGLE_API_KEY = '<api key>'`
- Execute the app and follow the instructions. `streamlit run app.py`

