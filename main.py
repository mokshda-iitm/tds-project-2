import os
import asyncio
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List, Dict
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API",
    description="API for generating HTML and JSON data analysis reports.",
    version="3.4"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataAnalystAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    
    async def _call_llm(self, prompt: str) -> str:
        """Calls the Gemini API with a given prompt."""
        retries = 3
        delay = 2
        for attempt in range(retries):
            try:
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}", json=payload, timeout=90
                )
                response.raise_for_status()
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text'].strip().replace('**', '')
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return "Error: Could not get a response from the LLM."

    async def _summarize_fact(self, question: str, fact: str) -> str:
        """Uses the LLM to turn a calculated fact into a natural language sentence."""
        prompt = f"""
        Based on the user's question and the provided data fact, formulate a clear and direct one-sentence answer.

        **User's Question:**
        {question}

        **Data Fact:**
        {fact}

        **Answer:**
        """
        return await self._call_llm(prompt)

    def _create_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Creates a base64 encoded scatter plot image."""
        try:
            plt.figure(figsize=(10, 7))
            sns.regplot(x=df[x_col], y=df[y_col], line_kws={"color": "red", "linestyle": "--"})
            plt.title(f'Relationship between {x_col} and {y_col}', fontsize=16)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{base64_img}"
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
            return "Could not generate plot."

    async def analyze_wikipedia(self, url: str, questions: List[str]) -> List[Dict]:
        """
        Uses a hybrid approach: pandas for calculations and LLM for summarization.
        """
        try:
            # 1. Scrape and clean data
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_html(io.StringIO(response.text), flavor='html5lib')[0]
            
            # More robust data cleaning
            df.columns = [str(col) for col in df.columns]
            if 'Worldwide gross' in df.columns:
                 df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.extract(r'(\d[\d,.]*)')[0].str.replace(',', '').astype(float)
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
            
            # Improved numerical column cleaning for a more precise correlation
            for col in ['Rank', 'Peak', 'Title']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Handle non-numeric values like '—' or '–' and convert to numeric
            for col in ['Rank', 'Peak']:
                if col in df.columns:
                    df[col] = df[col].str.replace('–', '').str.replace('—', '').str.extract(r'(\d+)')[0]
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            results = []

            # 2. Process each question with the correct tool
            for question in questions:
                answer = ""
                answer_type = "text"

                if any(q in question.lower() for q in ['plot', 'graph', 'chart', 'scatter']):
                    if 'Rank' in df.columns and 'Peak' in df.columns:
                        answer = self._create_plot(df, 'Rank', 'Peak')
                        answer_type = "image"
                    else:
                        answer = "Could not generate plot: 'Rank' or 'Peak' columns not found."

                elif 'correlation' in question.lower():
                    if 'Rank' in df.columns and 'Peak' in df.columns:
                        # Drop rows with NaN in these columns for a clean calculation
                        correlation_df = df.dropna(subset=['Rank', 'Peak'])
                        if not correlation_df.empty:
                            correlation = correlation_df['Rank'].corr(correlation_df['Peak'])
                            fact = f"The Pearson correlation is {correlation:.4f}."
                            answer = await self._summarize_fact(question, fact)
                        else:
                            answer = "Could not calculate correlation due to insufficient data."
                    else:
                        answer = "Could not calculate correlation: 'Rank' or 'Peak' columns not found."
                
                elif 'earliest' in question.lower() and '1.5 billion' in question.lower():
                    if 'Worldwide gross' in df.columns and 'Year' in df.columns and 'Title' in df.columns:
                        subset = df[df['Worldwide gross'] >= 1_500_000_000].dropna(subset=['Year'])
                        if not subset.empty:
                            earliest = subset.loc[subset['Year'].idxmin()]
                            fact = f"The film is '{earliest['Title']}' released in {int(earliest['Year'])}."
                            answer = await self._summarize_fact(question, fact)
                        else:
                            answer = "No films matching the criteria were found."
                    else:
                        answer = "Required columns (Worldwide gross, Year, Title) not found."

                elif 'how many' in question.lower() and '2 billion' in question.lower() and 'before 2000' in question.lower():
                    if 'Worldwide gross' in df.columns and 'Year' in df.columns:
                        count = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0]
                        fact = f"There is {count} film." if count == 1 else f"There are {count} films."
                        answer = await self._summarize_fact(question, fact)
                    else:
                        answer = "Required columns (Worldwide gross, Year) not found."

                else:
                    prompt = f"Based on the provided data snippet, answer the user's question: {question}\n\nData:\n{df.head().to_string()}"
                    answer = await self._call_llm(prompt)

                results.append({"question": question, "answer": answer, "type": answer_type})
            return results

        except Exception as e:
            logger.error(f"Wikipedia analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data analysis error: {str(e)}")


def extract_questions(text: str) -> List[str]:
    return [line.strip() for line in text.split('\n') if line.strip() and 'wikipedia.org' not in line]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Data Analyst Agent API! The API endpoints are /api/ for JSON and /api/html for HTML."}

# Endpoint for JSON output
@app.post("/api/", response_class=JSONResponse)
async def analyze_data_json(
    questions_file: UploadFile = File(..., alias="questions.txt", description="Text file with URL and questions")
):
    try:
        content_bytes = await questions_file.read()
        content_text = content_bytes.decode('utf-8').strip()
        
        url_match = re.search(r'https?://[^\s]+', content_text)
        if not url_match:
            raise HTTPException(status_code=400, detail="No URL found in the questions file.")
        url = url_match.group(0)
        
        questions = extract_questions(content_text)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")
        
        agent = DataAnalystAgent()
        analysis_results = await agent.analyze_wikipedia(url, questions)
        
        return JSONResponse(content=analysis_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# Endpoint for HTML output
@app.post("/api/html", response_class=HTMLResponse)
async def analyze_data_html(
    questions_file: UploadFile = File(..., alias="questions.txt", description="Text file with URL and questions")
):
    try:
        content_bytes = await questions_file.read()
        content_text = content_bytes.decode('utf-8').strip()
        
        url_match = re.search(r'https?://[^\s]+', content_text)
        if not url_match:
            raise HTTPException(status_code=400, detail="No URL found in the questions file.")
        url = url_match.group(0)
        
        questions = extract_questions(content_text)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")
        
        agent = DataAnalystAgent()
        analysis_results = await agent.analyze_wikipedia(url, questions)
        
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                h1 { color: #1a1a1a; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                .qa-pair { margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
                .question { font-weight: bold; color: #0056b3; font-size: 1.1em; }
                .answer { margin-top: 10px; }
                .answer img { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }
                pre { background-color: #f7f7f7; padding: 15px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; }
            </style>
        </head>
        <body>
            <h1>Data Analysis Report 📊</h1>
        """

        for item in analysis_results:
            html_content += '<div class="qa-pair">'
            html_content += f'<div class="question">❓ {item["question"]}</div>'
            if item["type"] == "image":
                html_content += f'<div class="answer"><img src="{item["answer"]}" alt="Generated Plot"></div>'
            else:
                html_content += f'<div class="answer"><pre>{item["answer"]}</pre></div>'
            html_content += '</div>'

        html_content += "</body></html>"
        
        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

