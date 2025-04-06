import sqlite3
import pandas as pd
from pathlib import Path

def view_table(table_name: str, limit: int = 10) -> pd.DataFrame:
    """View contents of a specific table."""
    db_path = Path("databases/recruitment.db")
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    return pd.read_sql_query(query, conn)

def list_tables() -> list:
    """List all tables in the database."""
    db_path = Path("databases/recruitment.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [table[0] for table in cursor.fetchall()]

def main():
    # List all tables
    tables = list_tables()
    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table}")
    
    # Let user select a table
    while True:
        try:
            choice = input("\nEnter table number to view (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            table_idx = int(choice) - 1
            if 0 <= table_idx < len(tables):
                table_name = tables[table_idx]
                print(f"\nShowing first 10 rows of {table_name}:")
                df = view_table(table_name)
                print(df.to_string())
                print("\nDataFrame shape:", df.shape)
            else:
                print("Invalid table number!")
        except ValueError:
            print("Please enter a valid number!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 