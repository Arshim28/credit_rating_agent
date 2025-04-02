import os
import gc
import json
import argparse
import logging
from typing import Dict, List, Any, Optional, TypedDict, Literal

from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph_reflection import create_reflection_graph
from langchain.chat_models import init_chat_model

# Import RAG components from the provided repository structure
from config import GOOGLE_API_KEY, MISTRAL_API_KEY, DEFAULT_CONFIG
from ocr_processor import OCR
from embedding import GeminiEmbeddingProvider
from document_processor import DocumentProcessor
from ocr_vector_store import OCRVectorStore
from vector_store import FaissVectorStore
from text_chunk import TextChunk
from utils import log_memory_usage, logger

# Set up logging to use the existing logger from utils
logger = logging.getLogger(__name__)

# Define state class for trend analysis
class TrendAnalysisState(MessagesState):
    reports_data: List[Dict[str, Any]]
    current_report_index: int
    extracted_ratings: List[Dict[str, Any]]
    trend_analysis: Optional[Dict[str, Any]] = None
    company_name: str
    vector_stores: Dict[str, OCRVectorStore]

# Define the main agent model that will analyze each report
def analyze_report(state: TrendAnalysisState) -> Dict:
    """
    Process the current report with the main analysis agent.
    This agent extracts key information from the current report and adds it to the analysis.
    """
    log_memory_usage("before_analyze_report")
    model = init_chat_model(model="claude-3-7-sonnet-latest")
    
    # Get current report
    current_index = state["current_report_index"]
    if current_index >= len(state["reports_data"]):
        # Finalize analysis if we've processed all reports
        return finalize_analysis(state)
    
    current_report = state["reports_data"][current_index]
    report_date = current_report["date"]
    report_path = current_report["file_path"]
    
    logger.warning(f"ANALYZING REPORT: {report_path}")
    
    # Get vector store for current report
    vector_store = state["vector_stores"].get(report_path)
    if not vector_store:
        logger.error(f"Vector store not found for {report_path}")
        return {
            "messages": state["messages"] + [{
                "role": "assistant", 
                "content": f"Error: Could not find vector store for report dated {report_date}"
            }],
            "current_report_index": current_index + 1  # Skip to next report
        }
    
    # Extract key information using specific queries
    logger.warning("EXTRACTING KEY INFORMATION FROM REPORT")
    ratings_info = query_report_for_ratings(vector_store)
    key_factors = query_report_for_factors(vector_store)
    outlook_info = query_report_for_outlook(vector_store)
    financial_highlights = query_report_for_financials(vector_store)
    
    # Combine into report analysis
    report_analysis = {
        "date": report_date,
        "ratings": ratings_info,
        "key_factors": key_factors,
        "outlook": outlook_info,
        "financials": financial_highlights,
        "path": report_path
    }
    
    # Add to extracted ratings list
    extracted_ratings = state.get("extracted_ratings", [])
    extracted_ratings.append(report_analysis)
    
    # Compare with previous report if available
    comparison = ""
    if current_index > 0 and len(extracted_ratings) > 1:
        previous_report = extracted_ratings[-2]
        comparison = generate_comparison(previous_report, report_analysis)
    
    # Create prompt for analysis
    if current_index == 0:
        # First report
        analysis_prompt = (
            f"I'm analyzing the rating report for {state['company_name']} dated {report_date}. "
            f"Here's what I found:\n\n{format_report_findings(report_analysis)}\n\n"
            f"This is the first report in our timeline. Please help me identify key aspects to track "
            f"for trend analysis in future reports."
        )
    else:
        # Subsequent report
        analysis_prompt = (
            f"I'm continuing my analysis of {state['company_name']} with the report dated {report_date}. "
            f"Here's what I found in this report:\n\n{format_report_findings(report_analysis)}\n\n"
            f"Comparison with previous report ({extracted_ratings[-2]['date']}):\n{comparison}\n\n"
            f"Please help me understand the key changes and their implications."
        )
    
    # Get analysis from the model
    logger.warning("GETTING ANALYSIS FROM MODEL")
    response = model.invoke([{"role": "user", "content": analysis_prompt}])
    
    log_memory_usage("after_analyze_report")
    gc.collect()
    log_memory_usage("after_gc_analyze_report")
    
    # Return updated state
    return {
        "messages": state["messages"] + [
            {"role": "user", "content": f"Analyzing report from {report_date}..."},
            {"role": "assistant", "content": response.content}
        ],
        "current_report_index": current_index + 1,
        "extracted_ratings": extracted_ratings
    }

def query_report_for_ratings(vector_store: OCRVectorStore) -> Dict[str, Any]:
    """Extract rating information from the report."""
    queries = [
        "What are the current ratings assigned to the company?",
        "What were the previous ratings?",
        "What rating instruments or facilities are being rated?",
        "What is the rating outlook?",
        "Has there been any rating change or reaffirmation?"
    ]
    
    results = {}
    for query in queries:
        logger.info(f"Querying: {query}")
        search_results = vector_store.answer_question(query, k=5)
        text_results = [result["text"] for result in search_results]
        results[query] = text_results
    
    return results

def query_report_for_factors(vector_store: OCRVectorStore) -> Dict[str, Any]:
    """Extract key rating factors from the report."""
    queries = [
        "What are the key strengths or positive factors mentioned?",
        "What are the key concerns, weaknesses, or risk factors mentioned?",
        "What mitigating factors are discussed in the report?"
    ]
    
    results = {}
    for query in queries:
        logger.info(f"Querying: {query}")
        search_results = vector_store.answer_question(query, k=5)
        text_results = [result["text"] for result in search_results]
        results[query] = text_results
    
    return results

def query_report_for_outlook(vector_store: OCRVectorStore) -> Dict[str, Any]:
    """Extract outlook information from the report."""
    queries = [
        "What is the rating outlook and what factors could lead to rating upgrade?",
        "What factors could lead to rating downgrade?",
        "What is the company's business outlook according to the report?"
    ]
    
    results = {}
    for query in queries:
        logger.info(f"Querying: {query}")
        search_results = vector_store.answer_question(query, k=5)
        text_results = [result["text"] for result in search_results]
        results[query] = text_results
    
    return results

def query_report_for_financials(vector_store: OCRVectorStore) -> Dict[str, Any]:
    """Extract financial information from the report."""
    queries = [
        "What are the key financial metrics and their values?",
        "What is the revenue or income reported?",
        "What is the profit or profitability reported?",
        "What is the debt level or gearing ratio?",
        "What is the working capital position?"
    ]
    
    results = {}
    for query in queries:
        logger.info(f"Querying: {query}")
        search_results = vector_store.answer_question(query, k=5)
        text_results = [result["text"] for result in search_results]
        results[query] = text_results
    
    return results

def format_report_findings(report_analysis: Dict[str, Any]) -> str:
    """Format the findings from a report into a readable summary."""
    formatted = f"Report Date: {report_analysis['date']}\n\n"
    
    formatted += "RATINGS:\n"
    for query, results in report_analysis["ratings"].items():
        formatted += f"- {query}\n"
        for i, result in enumerate(results[:2]):  # Limit to first 2 results
            formatted += f"  {result[:300]}...\n"
        formatted += "\n"
    
    formatted += "KEY FACTORS:\n"
    for query, results in report_analysis["key_factors"].items():
        formatted += f"- {query}\n"
        for i, result in enumerate(results[:2]):  # Limit to first 2 results
            formatted += f"  {result[:300]}...\n"
        formatted += "\n"
    
    formatted += "OUTLOOK:\n"
    for query, results in report_analysis["outlook"].items():
        formatted += f"- {query}\n"
        for i, result in enumerate(results[:2]):  # Limit to first 2 results
            formatted += f"  {result[:300]}...\n"
        formatted += "\n"
    
    formatted += "FINANCIAL HIGHLIGHTS:\n"
    for query, results in report_analysis["financials"].items():
        formatted += f"- {query}\n"
        for i, result in enumerate(results[:2]):  # Limit to first 2 results
            formatted += f"  {result[:300]}...\n"
        formatted += "\n"
    
    return formatted

def generate_comparison(previous_report: Dict[str, Any], current_report: Dict[str, Any]) -> str:
    """Generate a comparison between the current and previous report."""
    prev_date = previous_report["date"]
    curr_date = current_report["date"]
    
    comparison = f"Comparison between reports dated {prev_date} and {curr_date}:\n\n"
    
    # Compare ratings
    comparison += "RATING CHANGES:\n"
    prev_ratings = extract_key_text(previous_report["ratings"])
    curr_ratings = extract_key_text(current_report["ratings"])
    comparison += f"Previous: {prev_ratings[:300]}...\n"
    comparison += f"Current: {curr_ratings[:300]}...\n\n"
    
    # Compare key factors
    comparison += "CHANGES IN KEY FACTORS:\n"
    prev_factors = extract_key_text(previous_report["key_factors"])
    curr_factors = extract_key_text(current_report["key_factors"])
    comparison += f"Previous: {prev_factors[:300]}...\n"
    comparison += f"Current: {curr_factors[:300]}...\n\n"
    
    # Compare outlook
    comparison += "OUTLOOK CHANGES:\n"
    prev_outlook = extract_key_text(previous_report["outlook"])
    curr_outlook = extract_key_text(current_report["outlook"])
    comparison += f"Previous: {prev_outlook[:300]}...\n"
    comparison += f"Current: {curr_outlook[:300]}...\n\n"
    
    # Compare financials
    comparison += "FINANCIAL CHANGES:\n"
    prev_financials = extract_key_text(previous_report["financials"])
    curr_financials = extract_key_text(current_report["financials"])
    comparison += f"Previous: {prev_financials[:300]}...\n"
    comparison += f"Current: {curr_financials[:300]}...\n\n"
    
    return comparison

def extract_key_text(section: Dict[str, List[str]]) -> str:
    """Extract the most relevant text from a section."""
    result = ""
    for query, texts in section.items():
        if texts and len(texts) > 0:
            result += texts[0] + " "
    return result

def finalize_analysis(state: TrendAnalysisState) -> Dict:
    """Generate the final trend analysis after all reports have been processed."""
    log_memory_usage("before_finalize_analysis")
    model = init_chat_model(model="claude-3-7-sonnet-latest")
    
    extracted_ratings = state["extracted_ratings"]
    if not extracted_ratings:
        return {
            "messages": state["messages"] + [{
                "role": "assistant", 
                "content": "Unable to generate trend analysis. No reports were successfully processed."
            }],
            "current_report_index": state["current_report_index"]
        }
    
    # Sort by date (oldest to newest)
    extracted_ratings.sort(key=lambda x: x["date"])
    
    # Create a summary of all reports
    report_summary = ""
    for i, report in enumerate(extracted_ratings):
        report_summary += f"Report {i+1} ({report['date']}):\n"
        report_summary += f"- Key ratings: {extract_key_text(report['ratings'])[:300]}...\n"
        report_summary += f"- Key factors: {extract_key_text(report['key_factors'])[:300]}...\n"
        report_summary += f"- Outlook: {extract_key_text(report['outlook'])[:300]}...\n"
        report_summary += f"- Financials: {extract_key_text(report['financials'])[:300]}...\n\n"
    
    logger.warning("GENERATING FINAL TREND ANALYSIS")
    
    # Create prompt for final analysis
    final_prompt = (
        f"I've analyzed {len(extracted_ratings)} rating reports for {state['company_name']} "
        f"from {extracted_ratings[0]['date']} to {extracted_ratings[-1]['date']}.\n\n"
        f"Here's a summary of each report:\n\n{report_summary}\n\n"
        f"Please provide a comprehensive trend analysis covering:\n"
        f"1. Overall rating trend over time\n"
        f"2. Key factors that have influenced rating changes\n"
        f"3. Evolution of financial performance\n"
        f"4. Changes in business outlook\n"
        f"5. Recurring strengths and weaknesses\n"
        f"6. Recommendations for the company based on the rating history\n\n"
        f"Format the analysis with clear sections and highlight the most significant trends."
    )
    
    # Get final analysis from the model
    response = model.invoke([{"role": "user", "content": final_prompt}])
    
    trend_analysis = {
        "company_name": state["company_name"],
        "report_count": len(extracted_ratings),
        "date_range": f"{extracted_ratings[0]['date']} to {extracted_ratings[-1]['date']}",
        "analysis": response.content
    }
    
    log_memory_usage("after_finalize_analysis")
    gc.collect()
    log_memory_usage("after_gc_finalize_analysis")
    
    return {
        "messages": state["messages"] + [
            {"role": "user", "content": "Please provide the final trend analysis based on all reports."},
            {"role": "assistant", "content": response.content}
        ],
        "current_report_index": state["current_report_index"],
        "extracted_ratings": extracted_ratings,
        "trend_analysis": trend_analysis
    }

# Define the critique function to validate analysis
def critique_analysis(state: TrendAnalysisState) -> Optional[Dict]:
    """
    Review the current analysis for completeness and accuracy.
    """
    log_memory_usage("before_critique_analysis")
    model = init_chat_model(model="openai:o3-mini")
    
    if state["current_report_index"] >= len(state["reports_data"]):
        # No more reports to process, skip critique
        return None
    
    current_index = state["current_report_index"] - 1  # We're critiquing the report we just analyzed
    if current_index < 0:
        return None
    
    extracted_ratings = state.get("extracted_ratings", [])
    if not extracted_ratings or current_index >= len(extracted_ratings):
        return None
    
    current_analysis = extracted_ratings[current_index]
    
    logger.warning(f"CRITIQUING ANALYSIS FOR REPORT: {current_analysis['date']}")
    
    # Prepare critique prompt
    critique_prompt = (
        f"Review this analysis of a rating report for {state['company_name']} dated {current_analysis['date']}:\n\n"
        f"{format_report_findings(current_analysis)}\n\n"
        f"Check for these issues:\n"
        f"1. Missing information - Are any key rating details missing?\n"
        f"2. Unclear trends - Is the relationship with previous ratings clear?\n"
        f"3. Incomplete factors - Are important rating drivers overlooked?\n"
        f"4. Financial gaps - Are key financial metrics missing?\n"
        f"5. Outlook clarity - Is the future outlook clearly presented?\n\n"
        f"If you find any issues, explain what needs improvement. If everything looks good, respond with 'COMPLETE'."
    )
    
    # Get critique from the model
    response = model.invoke([{"role": "system", "content": "You are a critical reviewer of credit rating analyses."}, 
                            {"role": "user", "content": critique_prompt}])
    
    critique = response.content
    
    log_memory_usage("after_critique_analysis")
    gc.collect()
    log_memory_usage("after_gc_critique_analysis")
    
    if "COMPLETE" in critique.upper():
        # No issues found
        logger.info("Critique complete: No issues found")
        return None
    else:
        # Issues found, return the critique as a new user message
        logger.warning("Critique identified issues - requesting improvements")
        return {
            "messages": state["messages"] + [
                {"role": "user", "content": f"Please improve the analysis of the report dated {current_analysis['date']} based on this feedback: {critique}"}
            ]
        }

# Create the main assistant graph
def create_assistant_graph():
    logger.info("Creating assistant graph")
    assistant_graph = (
        StateGraph(TrendAnalysisState)
        .add_node("analyze_report", analyze_report)
        .add_edge(START, "analyze_report")
        .add_edge("analyze_report", END)
        .compile()
    )
    return assistant_graph

# Create the critique graph
def create_critique_graph():
    logger.info("Creating critique graph")
    critique_graph = (
        StateGraph(TrendAnalysisState)
        .add_node("critique_analysis", critique_analysis)
        .add_edge(START, "critique_analysis")
        .add_edge("critique_analysis", END)
        .compile()
    )
    return critique_graph

# Process PDFs and create vector stores
def process_reports(reports_data: List[Dict[str, Any]], vector_store_dir: str) -> Dict[str, OCRVectorStore]:
    """Process each PDF report and create vector stores."""
    vector_stores = {}
    
    for report in reports_data:
        report_path = report["file_path"]
        report_date = report["date"]
        
        logger.warning(f"PROCESSING REPORT: {report_path}")
        log_memory_usage(f"before_process_{os.path.basename(report_path)}")
        
        # Create a unique directory for each report's vector store
        report_store_dir = os.path.join(
            vector_store_dir,
            f"{os.path.basename(os.path.dirname(report_path))}_{report_date}"
        )
        
        # Check if vector store already exists
        if os.path.exists(report_store_dir) and os.listdir(report_store_dir):
            logger.warning(f"LOADING EXISTING VECTOR STORE for {report_path}")
            vector_store = OCRVectorStore()
            vector_store.load(report_store_dir)
        else:
            logger.warning(f"CREATING NEW VECTOR STORE for {report_path}")
            vector_store = OCRVectorStore(
                index_type="HNSW",
                chunk_size=5000,
                chunk_overlap=500
            )
            os.makedirs(report_store_dir, exist_ok=True)
            
            # Process the PDF
            try:
                vector_store.add_document(report_path)
                vector_store.save(report_store_dir)
                logger.warning(f"VECTOR STORE SAVED to {report_store_dir}")
            except Exception as e:
                logger.error(f"ERROR PROCESSING {report_path}: {e}")
                continue
        
        vector_stores[report_path] = vector_store
        
        log_memory_usage(f"after_process_{os.path.basename(report_path)}")
        gc.collect()
        log_memory_usage(f"after_gc_process_{os.path.basename(report_path)}")
    
    return vector_stores

def main():
    parser = argparse.ArgumentParser(description="Rating Trend Analyzer")
    parser.add_argument("--input", required=True, help="Input JSON file with company reports data")
    parser.add_argument("--output", default="trend_analysis_report.json", help="Output file for trend analysis")
    parser.add_argument("--vector-store-dir", default="vector_stores", help="Directory for storing vector stores")
    
    args = parser.parse_args()
    
    logger.warning("STARTING RATING TREND ANALYSIS")
    log_memory_usage("program_start")
    
    # Load company data
    with open(args.input, 'r') as f:
        companies_data = json.load(f)
    
    all_analyses = []
    
    for company in companies_data["companies"]:
        company_name = company["name"]
        reports_data = company["reports"]
        
        logger.warning(f"PROCESSING COMPANY: {company_name} with {len(reports_data)} reports")
        
        # Process reports and create vector stores
        vector_stores = process_reports(reports_data, args.vector_store_dir)
        
        if not vector_stores:
            logger.error(f"NO VECTOR STORES CREATED for {company_name}. Skipping.")
            continue
        
        # Create the reflection system
        logger.warning("CREATING REFLECTION SYSTEM")
        assistant_graph = create_assistant_graph()
        critique_graph = create_critique_graph()
        reflection_app = create_reflection_graph(assistant_graph, critique_graph)
        reflection_app = reflection_app.compile()
        
        # Initialize the state
        initial_state = {
            "messages": [{"role": "system", "content": f"You are analyzing rating reports for {company_name} to identify trends over time."}],
            "reports_data": reports_data,
            "current_report_index": 0,
            "extracted_ratings": [],
            "company_name": company_name,
            "vector_stores": vector_stores
        }
        
        # Run the reflection system
        logger.warning(f"STARTING ANALYSIS for {company_name}")
        log_memory_usage("before_reflection_run")
        
        result = reflection_app.invoke(initial_state)
        
        log_memory_usage("after_reflection_run")
        gc.collect()
        log_memory_usage("after_gc_reflection_run")
        
        # Save the analysis
        if result and "trend_analysis" in result and result["trend_analysis"]:
            all_analyses.append(result["trend_analysis"])
            logger.warning(f"ANALYSIS COMPLETED for {company_name}")
        else:
            logger.error(f"ANALYSIS FAILED for {company_name}")
    
    # Save all analyses to file
    with open(args.output, 'w') as f:
        json.dump({"analyses": all_analyses}, f, indent=2)
    
    logger.warning(f"ALL ANALYSES SAVED to {args.output}")
    log_memory_usage("program_end")

if __name__ == "__main__":
    main()