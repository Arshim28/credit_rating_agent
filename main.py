import os
import argparse
from src.rating_agent import RatingTrendAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Rating Trend Analyzer")
    parser.add_argument("--input", required=True, help="Input JSON file with company reports data")
    parser.add_argument("--output", default="trend_analysis_report", help="Output file for trend analysis (without extension)")
    parser.add_argument("--output-format", choices=["json", "markdown"], default="markdown", help="Output format (json or markdown)")
    parser.add_argument("--vector-store-dir", default="vector_stores", help="Directory for storing vector stores")
    parser.add_argument("--max-stores", type=int, default=3, help="Maximum number of vector stores to keep in memory")
    
    parser.add_argument("--report-analysis-model", default="gemini-2.5-pro-exp-03-25", help="Model to use for report analysis")
    parser.add_argument("--report-analysis-provider", default="google", help="Provider for report analysis model")
    parser.add_argument("--critique-model", default="gemini-2.0-flash", help="Model to use for critique")
    parser.add_argument("--critique-provider", default="google", help="Provider for critique model")
    parser.add_argument("--final-analysis-model", default="gpt-4o-mini", help="Model to use for final analysis")
    parser.add_argument("--final-analysis-provider", default="openai", help="Provider for final analysis model")
    
    args = parser.parse_args()
    
    models = {
        "report_analysis": {"name": args.report_analysis_model, "provider": args.report_analysis_provider},
        "critique": {"name": args.critique_model, "provider": args.critique_provider},
        "final_analysis": {"name": args.final_analysis_model, "provider": args.final_analysis_provider}
    }
    
    # Initialize and run the analyzer with yearly ratings extraction
    analyzer = RatingTrendAnalyzer(
        input_file=args.input,
        output_file=args.output,
        output_format=args.output_format,
        vector_store_dir=args.vector_store_dir,
        max_stores=args.max_stores,
        models=models
    )
    
    analyzer.run()
    
    # Inform user about additional ratings JSON output
    ratings_json_path = f"{args.output}_ratings.json"
    if os.path.exists(ratings_json_path):
        print(f"Analysis complete. Main report saved to {args.output}.{'md' if args.output_format == 'markdown' else 'json'}")
        print(f"Yearly ratings data saved to {ratings_json_path}")
    else:
        print(f"Analysis complete. Report saved to {args.output}.{'md' if args.output_format == 'markdown' else 'json'}")

if __name__ == "__main__":
    main()