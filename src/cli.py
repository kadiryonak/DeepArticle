"""
Multi-Agent Academic Paper Analysis System

This system uses multiple AI agents to:
1. Search academic papers from ArXiv, Semantic Scholar, PubMed, and Google Scholar
2. Collect metadata (journal metrics, citations, Q quartile)
3. Analyze and rank papers
4. Generate summaries using Groq LLM
5. Create an optimized reading order

Usage:
    python main.py "your research query"
    python main.py --interactive
"""

import sys
import argparse
from datetime import datetime

from config import GROQ_API_KEY
from graph.workflow import run_analysis
from utils.formatters import format_reading_list, export_to_json, export_to_markdown


def print_banner():
    """Print the application banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Multi-Agent Academic Paper Analysis System               ║
║     Powered by LangChain, LangGraph, and Groq               ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """Check if required environment variables are set."""
    warnings = []
    
    if not GROQ_API_KEY:
        warnings.append("⚠ GROQ_API_KEY not set - LLM summarization will be disabled")
    
    if warnings:
        print("\nEnvironment Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()
    
    return len(warnings) == 0


def run_interactive():
    """Run in interactive mode."""
    print_banner()
    check_environment()
    
    print("Enter your research queries (type 'quit' to exit):\n")
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not query:
                print("Please enter a query.\n")
                continue
            
            # Run analysis
            result = run_analysis(query)
            
            # Display results
            reading_order = result.get("reading_order", [])
            
            if reading_order:
                print("\n" + format_reading_list(reading_order))
                
                # Offer to export
                export = input("\nExport results? (json/md/no): ").strip().lower()
                
                if export == 'json':
                    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    if export_to_json(reading_order, filename):
                        print(f"✓ Exported to {filename}")
                    else:
                        print("✗ Export failed")
                        
                elif export == 'md':
                    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    if export_to_markdown(reading_order, filename):
                        print(f"✓ Exported to {filename}")
                    else:
                        print("✗ Export failed")
            else:
                print("\n⚠ No papers found for this query.\n")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break


def run_single_query(query: str, output_format: str = None, output_file: str = None):
    """Run a single query and optionally export results."""
    print_banner()
    check_environment()
    
    # Run analysis
    result = run_analysis(query)
    
    # Get results
    reading_order = result.get("reading_order", [])
    
    if reading_order:
        print("\n" + format_reading_list(reading_order))
        
        # Export if requested
        if output_format and output_file:
            if output_format == 'json':
                if export_to_json(reading_order, output_file):
                    print(f"\n✓ Exported to {output_file}")
                else:
                    print(f"\n✗ Failed to export to {output_file}")
            elif output_format == 'md':
                if export_to_markdown(reading_order, output_file):
                    print(f"\n✓ Exported to {output_file}")
                else:
                    print(f"\n✗ Failed to export to {output_file}")
    else:
        print("\n⚠ No papers found for this query.")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Academic Paper Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "machine learning for drug discovery"
  python main.py --interactive
  python main.py "transformer models" --output results.json --format json
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to analyze"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["json", "md"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive()
    elif args.query:
        run_single_query(args.query, args.format, args.output)
    else:
        # No arguments - show help and run interactive
        parser.print_help()
        print("\n" + "-" * 50)
        run_interactive()


if __name__ == "__main__":
    main()
