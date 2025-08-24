#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def create_line_graph(filename):
    """
    Read CSV file and create a line graph with n on x-axis and time on y-axis
    
    Args:
        filename (str): Path to the CSV file
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Check if required columns exist
        if 'n' not in df.columns or 'time' not in df.columns:
            print("Error: CSV file must contain 'n' and 'time' columns.")
            sys.exit(1)
        
        print(df['n'])

        # Create the line graph
        plt.figure(figsize=(10, 6))
        plt.plot(df['n'], df['time'], marker='o', linewidth=2, markersize=8)
        
        # Customize the graph
        plt.xlabel('n', fontsize=12)
        plt.ylabel('Time', fontsize=12)
        plt.title('Time vs n', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add some styling
        plt.tight_layout()

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        print(base_name)
        output_filename = os.path.join("..", "..","EspositoGrassiVenezia" ,"text", "report", f"img\{base_name}_graph.png")
        
        # Save the graph as PNG
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Graph saved as: {output_filename}")
        
        
        # Show the graph
        plt.show()
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create a line graph from CSV data')
    parser.add_argument('filename', help='Path to the CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create the graph
    create_line_graph(args.filename)

if __name__ == "__main__":
    main()