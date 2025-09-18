#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np

def create_line_graph(filename):
    """
    Read CSV file and create both logarithmic and linear scale graphs
    
    Args:
        filename (str): Path to the CSV file
    """
    try:
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        
        df = pd.read_csv(filename)
        
        if 'n' not in df.columns or 'time' not in df.columns:
            print("Error: CSV file must contain 'n' and 'time' columns.")
            sys.exit(1)

        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        if "solve_time.csv" in filename:
            base_name = "Solve"
            mult=1e3
        elif "setup_time.csv" in filename:
            base_name = "Set-up"
            mult=1
        elif "total_time.csv" in filename:
            base_name = "Total"
            mult=1e3
        else:
            base_name = "Time"
            mult=1e3

        # Parametri di leggibilità
        title_fontsize = 24
        label_fontsize = 18
        tick_fontsize = 14
        legend_fontsize = 16

        # Logarithmic scale graph
        plt.figure(figsize=(12, 8))
        plt.loglog(df['n'], df['time'], marker='o', label=f"{base_name} Time", color='blue')
        plt.loglog(df['n'], 1e3/df['n'], linestyle='--', label='O(n)', color='red')
        plt.xlabel('n', fontsize=label_fontsize)
        plt.ylabel('Time', fontsize=label_fontsize)
        plt.title(f"{base_name} Time vs n (Logarithmic Scale)", fontsize=title_fontsize, fontweight='bold')
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        
        log_output = os.path.join("..", "..","EspositoGrassiVenezia" ,"text", "report", f"img\{base_name}_log_graph.png")
        plt.savefig(log_output, dpi=300, bbox_inches='tight')
        print(f"Logarithmic graph saved as: {log_output}")
        plt.show()
        
        # Linear scale graph
        plt.figure(figsize=(12, 8))
        plt.plot(df['n'], df['time'], marker='o', label=f"{base_name} Time", color='blue')
        plt.xlabel('n', fontsize=label_fontsize)
        plt.ylabel('Time', fontsize=label_fontsize)
        plt.title(f"{base_name} Time vs n (Linear Scale)", fontsize=title_fontsize, fontweight='bold')
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        
        linear_output = os.path.join("..", "..","EspositoGrassiVenezia" ,"text", "report", f"img\{base_name}_linear_graph.png")
        plt.savefig(linear_output, dpi=300, bbox_inches='tight')
        print(f"Linear graph saved as: {linear_output}")
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
    parser = argparse.ArgumentParser(description='Create line graphs from CSV data')
    parser.add_argument('filename', help='Path to the CSV file')
    args = parser.parse_args()
    create_line_graph(args.filename)

if __name__ == "__main__":
    main()
