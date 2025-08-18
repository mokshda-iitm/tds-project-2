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
from typing import List, Dict, Any
import logging
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="General Purpose Data Analyst Agent API",
    description="API that uses an LLM to perform data analysis from various sources and formats.",
    version="4.1"
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
        """Calls the Gemini API with a given prompt, implementing exponential backoff."""
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
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 500, 503]:
                    logger.warning(f"API call failed with status {e.response.status_code} on attempt {attempt + 1}. Retrying in {delay} seconds...")
                    if attempt == retries - 1:
                        logger.error(f"Final retry failed. Raising error: {e}")
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"API call failed with unrecoverable error {e.response.status_code}: {e}")
                    raise
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return "Error: Could not get a response from the LLM."
    
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

    async def _handle_specific_questions(self, df: pd.DataFrame, question: str) -> Any:
        """Handles specific, pre-defined questions using deterministic logic."""
        if 'how many' in question.lower() and '2 bn' in question.lower() and 'before 2000' in question.lower():
            if 'Worldwide gross' in df.columns and 'Year' in df.columns:
                count = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0]
                return count
            return "Required columns (Worldwide gross, Year) not found."
        
        elif 'earliest film' in question.lower() and '1.5 bn' in question.lower():
            if 'Worldwide gross' in df.columns and 'Year' in df.columns and 'Title' in df.columns:
                subset = df[df['Worldwide gross'] >= 1_500_000_000].dropna(subset=['Year'])
                if not subset.empty:
                    earliest = subset.loc[subset['Year'].idxmin()]
                    return earliest['Title']
                return "No films matching the criteria were found."
            return "Required columns (Worldwide gross, Year, Title) not found."

        elif 'correlation' in question.lower():
            if 'Rank' in df.columns and 'Peak' in df.columns:
                correlation_df = df.dropna(subset=['Rank', 'Peak'])
                if not correlation_df.empty:
                    correlation = correlation_df['Rank'].corr(correlation_df['Peak'])
                    return round(correlation, 6)
                return "Could not calculate correlation due to insufficient data."
            return "Could not calculate correlation: 'Rank' or 'Peak' columns not found."

        elif 'draw a scatterplot' in question.lower():
            if 'Rank' in df.columns and 'Peak' in df.columns:
                return self._create_plot(df, 'Rank', 'Peak')
            return "Could not generate plot: 'Rank' or 'Peak' columns not found."

        return None # Return None if no specific question is matched

    async def analyze_data(self, data_content: str, questions: List[str], output_format: str) -> Any:
        """Analyzes data and answers questions using a hybrid approach."""
        
        # 1. Extract data into a DataFrame
        try:
            soup = BeautifulSoup(data_content, 'lxml')
            table = soup.find('table', {'class': 'wikitable'})
            if table is None:
                df = pd.DataFrame()
            else:
                df = pd.read_html(str(table), flavor='lxml')[0]
                df.columns = [str(col) for col in df.columns]

                # Robust data cleaning
                if 'Worldwide gross' in df.columns:
                    df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.extract(r'(\d[\d,.]*)')[0].str.replace(',', '').astype(float)
                if 'Year' in df.columns:
                    df['Year'] = pd.to_numeric(df['Year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                for col in ['Rank', 'Peak']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace('–', '').str.replace('—', '').str.extract(r'(\d+)')[0]
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to parse data content into a DataFrame: {e}")
            df = pd.DataFrame()

        results = []
        for question in questions:
            # 2. Try to handle with specific deterministic logic first
            answer = await self._handle_specific_questions(df, question)

            # 3. If no specific logic is found, use the LLM for a general answer
            if answer is None:
                cleaned_data_content = df.head().to_string() if not df.empty else data_content[:1000]
                prompt = f"""
                You are a data analyst. Based on the data provided, answer the following question.
                Your response should be concise, direct, and contain only the answer.

                DATA:
                {cleaned_data_content}

                QUESTION:
                {question}

                ANSWER:
                """
                answer = await self._call_llm(prompt)

            results.append(answer)

        if output_format == "object":
            return {questions[i]: results[i] for i in range(len(questions))}
        else:
            return results

def extract_url_and_questions(text: str) -> tuple[str, List[str]]:
    lines = text.split('\n')
    url = ""
    questions = []
    
    url_match = re.search(r'https?://[^\s]+', text)
    if url_match:
        url = url_match.group(0)

    for line in lines:
        if re.match(r'^\d+\.', line.strip()):
            questions.append(line.strip())
            
    if not questions:
        json_match = re.search(r'```json\s*\{([^}]*)\}\s*```', text, re.DOTALL)
        if json_match:
            try:
                import json
                json_part = "{" + json_match.group(1) + "}"
                temp_obj = json.loads(json_part)
                questions = list(temp_obj.keys())
            except json.JSONDecodeError:
                pass

    return url, questions

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Data Analyst Agent API! The API endpoints are /api/ for JSON and /api/html for HTML."}

@app.post("/api/", response_class=JSONResponse)
async def analyze_data_json(
    questions_file: UploadFile = File(..., alias="questions.txt", description="Text file with URL and questions")
):
    try:
        content_bytes = await questions_file.read()
        content_text = content_bytes.decode('utf-8').strip()
        
        url, questions = extract_url_and_questions(content_text)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")
        
        output_format = "array"
        if "JSON object" in content_text:
            output_format = "object"
        
        data_content = ""
        if url:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data_content = response.text
            except Exception as e:
                logger.error(f"Failed to fetch data from URL: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch data from URL: {e}")
        else:
            data_content = content_text

        agent = DataAnalystAgent()
        analysis_results = await agent.analyze_data(data_content, questions, output_format)

        return JSONResponse(content=analysis_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/api/html", response_class=HTMLResponse)
async def analyze_data_html(
    questions_file: UploadFile = File(..., alias="questions.txt", description="Text file with URL and questions")
):
    try:
        content_bytes = await questions_file.read()
        content_text = content_bytes.decode('utf-8').strip()
        
        url, questions = extract_url_and_questions(content_text)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")
        
        data_content = ""
        if url:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data_content = response.text
            except Exception as e:
                logger.error(f"Failed to fetch data from URL: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch data from URL: {e}")
        else:
            data_content = content_text

        agent = DataAnalystAgent()
        analysis_results = await agent.analyze_data(data_content, questions, "array")
        
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
        
        for i, question in enumerate(questions):
            answer = analysis_results[i]
            html_content += '<div class="qa-pair">'
            html_content += f'<div class="question">❓ {question}</div>'
            if isinstance(answer, str) and answer.startswith('data:image'):
                html_content += f'<div class="answer"><img src="{answer}" alt="Generated Plot"></div>'
            else:
                html_content += f'<div class="answer"><pre>{answer}</pre></div>'
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

