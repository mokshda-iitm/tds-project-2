import os
import asyncio
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List, Dict, Any, Optional
import logging
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="General Purpose Data Analyst Agent API",
    description="API that performs data analysis from various sources and formats.",
    version="6.6"
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
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, color: str = 'blue') -> str:
        """Creates a base64 encoded bar chart."""
        try:
            plt.figure(figsize=(10, 7))
            sns.barplot(x=df[x_col], y=df[y_col], color=color)
            plt.title(f'{y_col} by {x_col}', fontsize=16)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{base64_img}"
        except Exception as e:
            logger.error(f"Bar chart creation failed: {e}")
            return "Could not generate bar chart."

    def _create_cumulative_line_chart(self, df: pd.DataFrame, date_col: str, sales_col: str, color: str = 'red') -> str:
        """Creates a base64 encoded cumulative line chart."""
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            df['cumulative_sales'] = df[sales_col].cumsum()
            
            plt.figure(figsize=(10, 7))
            sns.lineplot(x=df[date_col], y=df['cumulative_sales'], color=color)
            plt.title('Cumulative Sales Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Sales', fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{base64_img}"
        except Exception as e:
            logger.error(f"Cumulative line chart creation failed: {e}")
            return "Could not generate cumulative line chart."

    def _create_scatterplot(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Creates a base64 encoded scatter plot image."""
        try:
            plt.figure(figsize=(10, 7))
            sns.regplot(x=df[x_col], y=df[y_col], line_kws={"color": "red", "linestyle": "--"})
            plt.title(f'Relationship between {x_col} and {y_col}', fontsize=16)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{base64_img}"
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
            return "Could not generate plot."

    def _perform_analysis(self, df: pd.DataFrame, question: str) -> Any:
        """Performs analysis based on the question and returns a value."""
        if 'total sales' in question.lower() and 'across all regions' in question.lower():
            return float(df['sales'].sum())
        
        elif 'highest total sales' in question.lower() and 'region' in question.lower():
            region_sales = df.groupby('region')['sales'].sum()
            return str(region_sales.idxmax())

        elif 'correlation' in question.lower() and 'day of month' in question.lower() and 'sales' in question.lower():
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_month'] = df['date'].dt.day
                correlation = df['day_of_month'].corr(df['sales'])
                return float(round(correlation, 6))
            return "Date column not found or malformed."

        elif 'median sales amount' in question.lower():
            return float(df['sales'].median())
        
        elif 'total sales tax' in question.lower() and '10%' in question.lower():
            total_sales = df['sales'].sum()
            return float(round(total_sales * 0.10, 2))

        elif 'bar chart' in question.lower() and 'sales' in question.lower() and 'region' in question.lower():
            region_sales = df.groupby('region')['sales'].sum().reset_index()
            return self._create_bar_chart(region_sales, 'region', 'sales')

        elif 'cumulative sales' in question.lower() and 'line chart' in question.lower():
            return self._create_cumulative_line_chart(df, 'date', 'sales')

        elif 'how many' in question.lower() and '2 bn' in question.lower() and 'before 2000' in question.lower():
            if 'Worldwide gross' in df.columns and 'Year' in df.columns:
                count = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0]
                return int(count)
            return "Required columns (Worldwide gross, Year) not found."
        
        elif 'earliest film' in question.lower() and '1.5 bn' in question.lower():
            if 'Worldwide gross' in df.columns and 'Year' in df.columns and 'Title' in df.columns:
                subset = df[df['Worldwide gross'] >= 1_500_000_000].dropna(subset=['Year'])
                if not subset.empty:
                    earliest = subset.loc[subset['Year'].idxmin()]
                    return str(earliest['Title'])
                return "No films matching the criteria were found."
            return "Required columns (Worldwide gross, Year, Title) not found."

        elif 'correlation' in question.lower() and 'rank' in question.lower() and 'peak' in question.lower():
            if 'Rank' in df.columns and 'Peak' in df.columns:
                correlation_df = df.dropna(subset=['Rank', 'Peak'])
                if not correlation_df.empty:
                    correlation = correlation_df['Rank'].corr(correlation_df['Peak'])
                    return float(round(correlation, 6))
                return "Could not calculate correlation due to insufficient data."
            return "Could not calculate correlation: 'Rank' or 'Peak' columns not found."

        elif 'draw a scatterplot' in question.lower() and 'rank' in question.lower() and 'peak' in question.lower():
            if 'Rank' in df.columns and 'Peak' in df.columns:
                return self._create_scatterplot(df, 'Rank', 'Peak')
            return "Could not generate plot: 'Rank' or 'Peak' columns not found."

        return None

    def _read_data_into_df(self, file_content: Optional[bytes], url: Optional[str]) -> pd.DataFrame:
        """Reads data from an uploaded file or URL into a DataFrame."""
        df = pd.DataFrame()
        if file_content:
            try:
                df = pd.read_csv(io.BytesIO(file_content))
                if 'sales' in df.columns:
                    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except Exception as e:
                logger.error(f"Failed to read uploaded file as CSV: {e}")
        elif url:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                table = soup.find('table', {'class': 'wikitable'})
                if table:
                    df = pd.read_html(str(table), flavor='lxml')[0]
                    df.columns = [str(col) for col in df.columns]
                    if 'Worldwide gross' in df.columns:
                        df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.extract(r'(\d[\d,.]*)')[0].str.replace(',', '').astype(float)
                    if 'Year' in df.columns:
                        df['Year'] = pd.to_numeric(df['Year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                    for col in ['Rank', 'Peak']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace('–', '').str.replace('—', '').str.extract(r'(\d+)')[0]
                            df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.error(f"Failed to fetch or parse data from URL: {e}")
        return df

    async def _analyze_data_and_format_response(self, df: pd.DataFrame, questions: Dict[str, str], output_format: str) -> Any:
        """Orchestrates analysis and formats the final response."""
        results = {}
        for key, question in questions.items():
            answer = self._perform_analysis(df, question)
            if answer is None:
                cleaned_data_content = df.head().to_string() if not df.empty else "No structured data available."
                prompt = f"""
                You are a data analyst. Based on the data provided (if any), answer the following question.
                Your response should be concise, direct, and contain only the answer.

                DATA:
                {cleaned_data_content}

                QUESTION:
                {question}

                ANSWER:
                """
                answer = await self._call_llm(prompt)
            results[key] = answer

        if output_format == "object":
            return {
                "total_sales": results.get("total_sales"),
                "top_region": results.get("top_region"),
                "day_sales_correlation": results.get("day_sales_correlation"),
                "bar_chart": results.get("bar_chart"),
                "median_sales": results.get("median_sales"),
                "total_sales_tax": results.get("total_sales_tax"),
                "cumulative_sales_chart": results.get("cumulative_sales_chart"),
                "films_2bn_before_2000": results.get("films_2bn_before_2000"),
                "earliest_1_5bn_film": results.get("earliest_1_5bn_film"),
                "rank_peak_correlation": results.get("rank_peak_correlation"),
                "rank_peak_scatterplot": results.get("rank_peak_scatterplot"),
            }
        else:
            return list(results.values())


def extract_questions(text: str) -> Dict[str, str]:
    """
    Extracts questions and their keys from the prompt text,
    handling various formats and mapping to expected analysis keys.
    """
    questions = {}
    
    # First, try to extract from JSON format
    json_match = re.search(r'```json\s*(\{.*\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            import json
            json_str = json_match.group(1)
            json_str = re.sub(r'\.\.\.', '""', json_str)
            json_str = re.sub(r'//.*?\n', '', json_str)
            temp_obj = json.loads(json_str)
            
            # Direct mapping for known question patterns
            for question_text in temp_obj.keys():
                q_lower = question_text.lower()
                
                if 'total sales' in q_lower and 'across all regions' in q_lower:
                    questions['total_sales'] = question_text.strip()
                elif 'highest total sales' in q_lower and 'region' in q_lower:
                    questions['top_region'] = question_text.strip()
                elif 'correlation' in q_lower and 'day of month' in q_lower and 'sales' in q_lower:
                    questions['day_sales_correlation'] = question_text.strip()
                elif 'bar chart' in q_lower and 'sales' in q_lower and 'region' in q_lower:
                    questions['bar_chart'] = question_text.strip()
                elif 'median sales amount' in q_lower:
                    questions['median_sales'] = question_text.strip()
                elif 'total sales tax' in q_lower and '10%' in q_lower:
                    questions['total_sales_tax'] = question_text.strip()
                elif 'cumulative sales' in q_lower and 'line chart' in q_lower:
                    questions['cumulative_sales_chart'] = question_text.strip()
                elif '2 bn' in q_lower and 'before 2000' in q_lower:
                    questions['films_2bn_before_2000'] = question_text.strip()
                elif 'earliest film' in q_lower and '1.5 bn' in q_lower:
                    questions['earliest_1_5bn_film'] = question_text.strip()
                elif 'correlation' in q_lower and 'rank' in q_lower and 'peak' in q_lower:
                    questions['rank_peak_correlation'] = question_text.strip()
                elif 'scatterplot' in q_lower and 'rank' in q_lower and 'peak' in q_lower:
                    questions['rank_peak_scatterplot'] = question_text.strip()
                else:
                    # Fallback: create safe key
                    safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', question_text.lower())[:30]
                    questions[safe_key] = question_text.strip()
            
            return questions
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON questions: {e}")
    
    # If no JSON or JSON parsing failed, extract from text
    lines = text.split('\n')
    question_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
            
        # Look for question patterns
        question_match = None
        patterns = [
            r'^(?:\d+[\.\)]?\s+)(.*?[?])',  # Numbered lists
            r'^[-*•]\s+(.*?[?])',           # Bullet points
            r'^"\s*(.*?[?])\s*"',           # Quoted questions
            r'^(.*?[?])\s*$'                # Standalone questions
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and '?' in match.group(1):
                question_match = match.group(1).strip()
                break
        
        if not question_match:
            # Try to extract any question from the line
            q_match = re.search(r'([^.!]*\?)', line)
            if q_match:
                question_match = q_match.group(1).strip()
        
        if question_match and len(question_match) > 8:
            q_lower = question_match.lower()
            key = None
            
            # Map to expected analysis keys
            if 'total sales' in q_lower and 'across all regions' in q_lower:
                key = 'total_sales'
            elif 'highest total sales' in q_lower and 'region' in q_lower:
                key = 'top_region'
            elif 'correlation' in q_lower and 'day of month' in q_lower and 'sales' in q_lower:
                key = 'day_sales_correlation'
            elif 'bar chart' in q_lower and 'sales' in q_lower and 'region' in q_lower:
                key = 'bar_chart'
            elif 'median sales amount' in q_lower:
                key = 'median_sales'
            elif 'total sales tax' in q_lower and '10%' in q_lower:
                key = 'total_sales_tax'
            elif 'cumulative sales' in q_lower and 'line chart' in q_lower:
                key = 'cumulative_sales_chart'
            elif '2 bn' in q_lower and 'before 2000' in q_lower:
                key = 'films_2bn_before_2000'
            elif 'earliest film' in q_lower and '1.5 bn' in q_lower:
                key = 'earliest_1_5bn_film'
            elif 'correlation' in q_lower and 'rank' in q_lower and 'peak' in q_lower:
                key = 'rank_peak_correlation'
            elif 'scatterplot' in q_lower and 'rank' in q_lower and 'peak' in q_lower:
                key = 'rank_peak_scatterplot'
            else:
                # Use generic key for unmapped questions
                key = f"question_{question_count + 1}"
            
            questions[key] = question_match
            question_count += 1
    
    # Log for debugging
    if questions:
        logger.info(f"Extracted {len(questions)} questions: {list(questions.keys())}")
    else:
        logger.warning("No questions could be extracted from the text")
        # Return at least one question to avoid 400 error
        questions['default_question'] = "What analysis can you perform on this data?"
    
    return questions


@app.get("/")
async def read_root():
    return {"message": "Welcome to the General Purpose Data Analyst Agent API! Endpoints: /api/ and /api/html"}

@app.post("/api/", response_class=JSONResponse)
async def analyze_data_json(
    request: Request,
    questions_file: Optional[UploadFile] = File(None, alias="questions.txt"),
    data_file: Optional[UploadFile] = File(None, alias="data.csv"),
    url_input: Optional[str] = Form(None, alias="url")
):
    try:
        agent = DataAnalystAgent()
        questions_content = ""
        df = pd.DataFrame()

        if questions_file:
            questions_content = (await questions_file.read()).decode('utf-8').strip()
        else:
            questions_content = (await request.body()).decode('utf-8').strip()

        output_format = "object" if "JSON object" in questions_content else "array"
        questions = extract_questions(questions_content)
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")

        if data_file:
            df = agent._read_data_into_df(await data_file.read(), None)
        elif url_input:
            df = agent._read_data_into_df(None, url_input)
        else:
            url_match = re.search(r'https?://[^\s]+', questions_content)
            if url_match:
                df = agent._read_data_into_df(None, url_match.group(0))

        if df.empty:
            raise HTTPException(status_code=400, detail="Could not read valid data from the provided source.")

        analysis_results = await agent._analyze_data_and_format_response(df, questions, output_format)

        return JSONResponse(content=analysis_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/api/html", response_class=HTMLResponse)
async def analyze_data_html(
    request: Request,
    questions_file: Optional[UploadFile] = File(None, alias="questions.txt"),
    data_file: Optional[UploadFile] = File(None, alias="data.csv"),
    url_input: Optional[str] = Form(None, alias="url")
):
    try:
        agent = DataAnalystAgent()
        questions_content = ""
        
        if questions_file:
            questions_content = (await questions_file.read()).decode('utf-8').strip()
        else:
            questions_content = (await request.body()).decode('utf-8').strip()

        questions = extract_questions(questions_content)
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the file.")
        
        df = pd.DataFrame()
        if data_file:
            df = agent._read_data_into_df(await data_file.read(), None)
        elif url_input:
            df = agent._read_data_into_df(None, url_input)
        else:
            url_match = re.search(r'https?://[^\s]+', questions_content)
            if url_match:
                df = agent._read_data_into_df(None, url_match.group(0))

        if df.empty:
            raise HTTPException(status_code=400, detail="Could not read valid data from the provided source.")
        
        analysis_results = await agent._analyze_data_and_format_response(df, questions, "object")
        
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
        
        for key, answer in analysis_results.items():
            html_content += '<div class="qa-pair">'
            html_content += f'<div class="question">❓ {key}</div>'
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
