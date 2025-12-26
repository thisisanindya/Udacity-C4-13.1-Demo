
### GitHub Repo
### https://github.com/thisisanindya/Udacity-C4-13.1-Demo/blob/main/test-new.py
###  https://github.com/thisisanindya/Udacity-C4-13.1-Demo/blob/main/test-new-output.txt

import re
import os
import dotenv
import ast
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Engine
from sqlalchemy.sql import text
from dotenv import load_dotenv

# Import pydantic and related packages
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel
from typing import Dict, Literal
from pydantic_ai import Agent
# from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openai import OpenAIChatModel

# from pydantic_ai.models.openai import OpenAIResponsesModel
################################

# put a note install - numpy, pandas and pydantic-ai 
# packages using below commandspi 
################# RUN BELOW COMMANDS 

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},  # per cup
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]


# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """

    # Debug log, comment if not required
    # print(f"---> Function (generate_sample_inventory): Populate date for sample inventory.")

    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """

    # Debug log, comment if not required
    # print(f"---> Function (init_database): Initialize the database with transactions, quotes and inventory data.")

    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],  # Quantity involved
            "price": [],  # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """

    # Debug log, comment if not required
    # print(f"---> Function (create_transaction): Creating transaction for '{item_name}' of type '{transaction_type}'.")

    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        ################ Commenting for time being miving this to QuotingAgent
        # Calculate discount based on the quantity of items 
        '''
        print("---> Check for quantity = ", quantity)
        discount = 0
        if quantity < 100:
            discount = 0.05
            print("---> small quantity")
        elif quantity < 500:
            discount = 0.10
            print("---> medium quantity")
        elif quantity < 1000:
            discount = 0.15   
            print("---> large quantity") 
        price -= price*discount
        print("---> discounted price = ", price)
        # call a simple discount function apply_discount(item_name, transaction_type, quantity, price, date)        
        '''
        ################

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # Debug log, comment if not required
    print(f"---> Function (get_all_inventory): Get inventory data as of '{as_of_date}'")

    # SQL query to compute stock levels per item as of the given date
    query = """
            SELECT item_name,
                   SUM(CASE
                           WHEN transaction_type = 'stock_orders' THEN units
                           WHEN transaction_type = 'sales' THEN -units
                           ELSE 0
                       END) as stock
            FROM transactions
            WHERE item_name IS NOT NULL
              AND transaction_date <= :as_of_date
            GROUP BY item_name
            HAVING stock > 0 \
            """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

#     Code commented and added below updated method
#     def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
#     """
#     Retrieve the stock level of a specific item as of a given date.
# 
#     This function calculates the net stock by summing all 'stock_orders' and 
#     subtracting all 'sales' transactions for the specified item up to the given date.
# 
#     Args:
#         item_name (str): The name of the item to look up.
#         as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.
# 
#     Returns:
#         pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
#     """

# INFO: Changed the return type to dict with attribute 'item_name' and 'current_stock'
# to handle pydantic_core._pydantic_core.PydanticSerializationError: 
# Unable to serialize unknown type: <class 'pandas.core.frame.DataFrame'>

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> dict:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        dict with attribute 'item_name' and 'current_stock'.
    """

    # Debug log (comment out in production if needed)
    print(f"---> Function (get_stock_level): Get stock for '{item_name}' as of '{as_of_date}'")
    
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
                SELECT item_name,
                        COALESCE(SUM(CASE
                                        WHEN transaction_type = 'stock_orders' THEN units
                                        WHEN transaction_type = 'sales' THEN -units
                                        ELSE 0
                            END), 0) AS current_stock
                FROM transactions
                WHERE item_name = :item_name
                AND transaction_date <= :as_of_date \
            """

    df = pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

    if df.empty:
        return {"item_name": item_name, "current_stock": 0}

    return df.iloc[0].to_dict()


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"---> Function (get_supplier_delivery_date): Get estimated delivery date in as of '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    # Debug log (comment out in production if needed)
    print(f"---> Function (get_cash_balance): Get cash balance as of '{as_of_date}'")

    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """

    # Debug log (comment out in production if needed)
    print(f"---> Function (generate_financial_report): Get financial report as of '{as_of_date}'")

    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
                      SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
                      FROM transactions
                      WHERE transaction_type = 'sales'
                        AND transaction_date <= :date
                      GROUP BY item_name
                      ORDER BY total_revenue DESC
                      LIMIT 5 \
                      """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    
    # Debug log (comment out in production if needed)
    print(f"---> Function (search_quote_history): Search quote history.")

    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """
    '''
    Code commented as the return statement is not correct
    try:
        # Execute parameterized query
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row._mapping) for row in result]
    except Exception as e:
        print(f"Error searching quote history: {e}")
        return []
    '''

    # Execute parameterized query
    # Updated as per suggestion from mentor
    # Reference - https://knowledge.udacity.com/questions/1076722 
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(r) for r in result.mappings()]

def date_check(supplier_delivery_date: str, requested_delivery_date: str) -> bool:
    """
    Checks if supplier date is greater than delivery date. 

    Args:
        date_1 (str): List of terms to match against customer requests and explanations.
        date_2 (str): Maximum number of quote records to return. Default is 5.

    Returns:
        bool: True or False
     
    """

    # Convert the string dates to a datetime object
    supplier_delivery_date = datetime.fromisoformat(supplier_delivery_date_str)
    requested_delivery_date = datetime.fromisoformat(requested_delivery_date_str)
    
    # Comparison
    if supplier_delivery_date >= requested_delivery_date:
        return True
    else:
        return False
 

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# Set up and load your env parameters and instantiate your model.
#load_dotenv()

print("Loading openai_api_key... ")
dotenv.load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
print("Checking OPENAI_API_KEY = ", openai_api_key)

#######################
# Initialize the OpenAI model (ensure your OPENAI_API_KEY is set in your environment)

openai_model = OpenAIChatModel(model_name="gpt-4o-mini")
print("Model ===> ", openai_model)

# Define tools for the agents
create_transaction_tool = Tool(
    name="create_transaction",
    description="""
        This function records a transaction of type 'stock_orders' or 'sales' with a specified
        item name, quantity, total price, and transaction date into the 'transactions' table of 
        the database.
    """,
    function=create_transaction,
    require_parameter_descriptions=True
)

get_all_inventory_tool = Tool(
    name="get_all_inventory",
    description="""
        Retrieve a snapshot of available inventory as of a specific date.

        This function calculates the net quantity of each item by summing 
        all stock orders and subtracting all sales up to and including the given date.

        Only items with positive stock are included in the result.
    """,
    function=get_all_inventory,
    require_parameter_descriptions=True
)

get_stock_level_tool = Tool(
    name="get_stock_level",
    description="""
        Retrieve the stock level of a specific item as of a given date.

        This function calculates the net stock by summing all 'stock_orders' 
        and subtracting all 'sales' transactions for the specified item up 
        to the given date.
    """,
    function=get_stock_level,
    require_parameter_descriptions=True
)

get_supplier_delivery_date_tool = Tool(
    name="get_supplier_delivery_date",
    description="""
        Estimate the supplier delivery date based on the requested order 
        quantity and a starting date.
    
        Delivery lead time increases with order size:
            - ≤10 units: same day
            - 11–100 units: 1 day
            - 101–1000 units: 4 days
            - >1000 units: 7 days    
    """,
    function=get_supplier_delivery_date,
    require_parameter_descriptions=True
)

get_cash_balance_tool = Tool(
    name="get_cash_balance",
    description="""
        Calculate the current cash balance as of a specified date.

        The balance is computed by subtracting total stock purchase costs 
        ('stock_orders') from total revenue ('sales') recorded in the 
        transactions table up to the given date.
    """,
    function=get_cash_balance,
    require_parameter_descriptions=True
)

generate_financial_report_tool = Tool(
    name="generate_financial_report",
    description="""
        Generate a complete financial report for the company as of a specific date.
        
        This includes:
        - Cash balance
        - Inventory valuation
        - Combined asset total
        - Itemized inventory breakdown
        - Top 5 best-selling products
    """,
    function=generate_financial_report,
    require_parameter_descriptions=True
)

search_quote_history_tool = Tool(
    name="search_quote_history",
    description="""
        Retrieve a list of historical quotes that match any of the provided search terms.
        
        The function searches both the original customer request (from `quote_requests`) and
        the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
        most recent order date and limited by the `limit` parameter.
    """,
    function=search_quote_history,
    require_parameter_descriptions=True
)

date_check_tool = Tool(
    name="date_check",
    description ="""
    Checks if supplier date is greater than delivery date. 
    """,
    function=search_quote_history,
    require_parameter_descriptions=True
)


# Set up your agents and create an orchestration agent that will manage them.
extraction_tools = [get_all_inventory_tool]

inventory_tools = [get_all_inventory_tool, get_stock_level_tool, get_supplier_delivery_date_tool,
                          create_transaction_tool, get_cash_balance_tool]
quoting_tools = [search_quote_history_tool, generate_financial_report_tool]
# generate_quote_tool, 
# discounting_tools = [calculate_discount_tool]
sales_tools = [create_transaction_tool, get_supplier_delivery_date_tool]
# reporting_tools = [generate_financial_report_tool, get_cash_balance_tool, search_quote_history_tool]

# Define output model for the orchestration agent
class BeaverOrchestrator(BaseModel):
    classification: Literal["QUERY", "ORDER"]

class ExtractionAgentResponse(BaseModel):
    answer: str
    ok_to_proceed: bool

class InventoryAgentResponse(BaseModel):
    answer: str
    ok_to_proceed: bool

class QuotingAgentResponse(BaseModel):
    answer: str
    items: list
    total_price: float
    ok_to_proceed: bool   

class SalesAgentResponse(BaseModel):
    answer: str
    items: list
    total_price: float
    ok_to_proceed: bool    

class ReceiptAgentResponse(BaseModel):
    answer: str
    ok_to_proceed: bool  

from pydantic_ai.usage import UsageLimits

shared_model_settings = ModelSettings(
    temperature=0.0,
    max_tokens=1024,
)

#shared_model_settings={"temperature": 0.0, "max_tokens": 1024}

UNLIMITED_USAGE = UsageLimits(
    request_limit=None,
    total_tokens_limit=None
)

INEVNTORY_DATA = [
					"Paper plates",
					"100 lb cover stock",
					"Glossy paper",
					"Rolls of banner paper (36-inch width)",
					"Photo paper",
					"Cardstock",
					"Colored paper",
					"80 lb text paper",
					"Large poster paper (24x36 inches)",
					"Table covers",
					"Butcher paper",
					"Kraft paper",
					"Banner paper",
					"Presentation folders",
					"Patterned paper",
					"A4 paper",
					"Invitation cards",
				]

ORCHESTRATOR_PROMPT = """
                        You are a request classifier for Beaver's Choice Paper Supply.

                        Classify the customer request into exactly one of:

                        - QUERY: asking for information only (availability, stock, delivery dates, reports, pricing history)
                        - ORDER: intent to buy, order, purchase, need items, quantities, or deadlines

                        Rules:
                        - If intent to buy is implied → ORDER
                        - If only information is requested → QUERY
                        - If unclear → QUERY
                        - Do not explain, Output JSON only
                        
						- Output Format:
						Return ONLY valid JSON matching this schema:
                        {
                            "classification": "QUERY" | "ORDER"
                        }
						
						Pass these details to {ExtractionAgent}
                    """

EXTRACTION_PROMPT = """
                        This is a technical work for Beaver's Choice Paper Supply. 
                        You are expert in extracting information like 
                        - item_name
                        - quantity
                        - unit
                        from a set of customer request. Example customer request 
                        is given below: 

                        Example -  
                            [
                                {
                                    'item_name': 'a4 glossy paper\n- 100 sheets of heavy cardstock (white)\n- 100 sheets of colored paper (assorted colors)\n\n i need these supplies delivered by april 15', 
                                    'quantity': 200, 
                                    'unit': 'sheets'
                                }, 
                                {
                                    'item_name': 'Photo paper\n- 200 sheets \n i need these supplies delivered by april 12', 
                                    'quantity': 202, 
                                    'unit': 'units'
                                }, 
                                {
                                    'item_name': 'colorful poster paper', 
                                    'quantity': 500, 
                                    'unit': 'sheets'
                                },
                                {
                                    'item_name': 'balloons for the parade', 
                                    'quantity': 200, 
                                    'unit': 'units'
                                }
                            ]

                        The customer request data is stored in dictionary with keys: 
                        - item_name, quantity and unit
                        There can be multiple customer requests. These are stored in
                        a list element as logical grouping. 

                        For each customer request, please follow below steps for extraction : 
							- use the tool `get_all_inventory` to get distinct list of
							item_name and their stock from the inventory
							- Checkpoint: Please note that the item_name should be available 
							in the data structure {INEVNTORY_DATA}
							- read the custoner request and scan the above inventory data to find the best fit 
							{item_name} from the inventory and find available {stock}. Use features like cosine similarity
							to find the best match. 
							- identify the {quantity} and {unit} requested {requested_delivery_date} and from the customer request
							- if {stock} is less than {quantity} 
								- update
									- {ok_to proceed}: False
									- {answer}: - {item_name} of {quantity} is not available. Delivery can't be met. 
							- Else 
								- update 
									- {ok_to proceed}: True
									- {answer}: {item_name} of {quantity} is available. Proceeding for next steps. 
							- Move to the next item_name
							- Prepare list / array of item_name's and list of corresponding quantity's
						
						- Tools:
						- `get_all_inventory`: Retrieve item name and available stock from inventory.

						Prepare all the data required to pass to inventory agent as per below output format. 
						Always be empathetic and helpful to the customer.
                        
                        - Output Format:
                            - For a string with details of - answer, item_name's, quantity's, stock's, requested_delivery_date's, ok_to_proceed in the below format 
                            - "answer: {answer}, items: {item_name}, quantity: {quantity}, , stock: {stock}, {requested_delivery_date}: requested_delivery_date, ok_to_proceed: {ok_to_proceed}"
							
						Pass these details to {ExtractionAgent}	
                    """

INVENTORY_PROMPT = """
                        You are the Inventory Agent for the Beaver's Choice Paper supply company.

                        You will receive the request which are identified as "QUERY" or "ORDER" 
                        Use ITEMS_JSON as the prime source for items and quantities.
                        Always follow below instructions for decision making: 
						
						For each {item_name}
							If classification is QUERY: 
							- Check stock using the tool `get_stock_level` for the item_name or item_names
							- {ok_to_proceeed}: True
								- if {stock} > 0, update 
									- {answer}: {item_name} is available with {stock} units. 
								- else, update
									- {answer}: {item_name} is not available presently. 
							- Do not modify inventory 

							If classification is ORDER:
							- Assume {minimum_cash_balance} is $ 1000. 
							- Assume {minimum_item_in_stock} is 100. 
							- Call `get_cash_balance` before any other action.
							- if it is < {minimum_cash_balance} update
								- {answer}: Internal Issues. Please come back later. 
								- {ok_to_proceeed}: False
							- else
								- if {quantity} < {minimum_item_in_stock} update
									- {ok_to_proceeed}: True
									- Move to {QuotingAgent} for further quoting steps. 
								- else
									- call `get_supplier_delivery_date` only for the customer requested item_name's
									- Use `date_check` tool to check if this date (supplier_delivery_date) greater than {requested_delivery_date} 
									from {ExtractionAgent} and update
										- {answer}: {item_name} can be delivered by {requested_delivery_date}. 
										- {ok_to_proceed}: True
										- Move to {QuotingAgent} for quoting. 
									- else update : 
										- {ok_to_proceed}: True
										- {answer}: {item_name} can be delivered by {supplier_delivery_date}. 
										- Move to {QuotingAgent} for order re-stocking with transaction_type as 'stock_orders'
							- Move to next item_name 
						
                        - Tools:
                        - `get_cash_balance`: Inventory of all items
                        - `get_stock_level`: Provide available quantity of a specific item
                        - `get_supplier_delivery_date`: Estimated delivery date
						- `date_check`: Compare supplier and delivery dates

						Prepare all the data required to pass to quoting agent as per below output format. 
						Always be empathetic and helpful to the customer.
                        
                        - Output Format:
                            - Form a string with details of - amswer, item_name's, quantity's, unit, stock, requested_delivery_date, ok_to_proceed in the below format 
                            - "answer: {answer}, items: {item_name}, quantity: {quantity}, {unit}: unit, stock: {stock}, {requested_delivery_date}: requested_delivery_date, {ok_to_proceed}: ok_to_proceed"
							
						Pass these details to {QuotingAgent}	

                    """

QUOTING_PROMPT = """
						You are the Quote Agent for the Beaver's Choice Paper supply company. 

						Your task is to generate competitive and strategic sales quote based on:
						- customer order request
						- the current stock and requested delivery date provided by the Inventory Agent
						- historical quote and sales data


						- Always follow below instructions for decision making: 

						1. Analyze the customer's request 
						- Identify the required item(s), quantity and delivery dates

						2. Use inventory context
						- Determine whether the items are available or when they will be deliverable (this information is provided to you as input).
						- This shouid be provided to you as input. No need to check from database.

						3. Analyze pricing history
						- Use `search_quote_history` to find related past quotes.
						- Use `generate_financial_report` if needed to fine sales and pricing trends.

						4. Calculate a competitive quote
						- Apply volume discounts if required
						- Consider priority and availability
						- Ensure profitability while being attractive to the customer.

						5. Prepare the output
						- Provide a clear price per unit and total price.
                        - Generate quote and pass pricing details to Sales Agent
						- Include any relevant remarks (e.g., “discount applied due to high volume”).
                        - Populate the fields 
                            - `answer` with expected time of delivery the items
                            - `items` with list of items
                            - `total_price` with the total price of the item
                            - `ok_to_proceed` with True if answer is valid or False otherwise

						- Tools:
						- `search_quote_history`: Retrieve previous similar offers for reference.
						- `generate_financial_report`: Analyze pricing and sales trends.

						You are not responsible to check stock. Focus only on creating an optimized offer. 
						Always be empathetic and helpful to the customer.
                        
						- Output Format:
                            - Form a string with details of - amswer, item_name's, quantity's, unit, stock, requested_delivery_date, ok_to_proceed in the below format 
                            - "answer: {answer}, items: {item_name}, quantity: {quantity}, {unit}: unit, stock: {stock}, {requested_delivery_date}: requested_delivery_date, total_price: {total_price}, {ok_to_proceed}: ok_to_proceed"
							
						Pass these details to {SalesAgent}	
                        """

SALES_PROMPT = """
						You are the Sales Agent for the Beaver's Choice Paper supply company.
						The main responsibility is to complete the customer's order based 
						on the quote provided by the Quote Agent and current inventory status.

						1. Assume the customer wants to proceed the order.

						2. Store the sale
							- For each item_name Call `create_transaction` with {transaction_type} 'sales'
                                - quantity = units ordered
                                - price = unit multiplied by quantity

						3. Reply to customer
							- Confirm if order is successful
							- Include estimated delivery date.
							
						4. Please note
							- No need generate new quote – this is done by Quoting Agent.
							- This role isto verify feasibility and execute the transaction.
                        
                        5. Populate the fields 
                            - `answer` with expected time of delivery the items
                            - `item_name` with list of items
                            - `total_price` with the total price of the item
                            - `ok_to_proceed` with True if answer id valid or False otherwise
						
                        Always be empathetic and helpful to the customer.

						- Tools:
						- `create_transaction`: Store the sale record

                        - Output Format:
                            - Form a string with details of - amswer, item_name's, quantity's, unit, stock, requested_delivery_date, ok_to_proceed in the below format 
                            - "answer: {answer}, items: {item_name}, quantity: {quantity}, {unit}: unit, stock: {stock}, {requested_delivery_date}: requested_delivery_date, total_price: {total_price}, {ok_to_proceed}: ok_to_proceed"
							
						Pass these details to {Receipt}
                        """

RECEIPT_PROMPT = """    
						You are the Receipt Agent for the Beaver's Choice Paper supply company.

						Main idea is to Generate a short summary of **customer Receipt** for finalized order 
						as a token of acknowledgement. 
						
						Your job is to generate a complete and professional **customer receipt** based on the finalized order.
						You receive data input that includes:
						- Customer name and optional address
						- item(s) purchased, quantities, unit prices and total price
						- Discounts applied, if any
						- Delivery date, if any

						- The receipt contains the following:
						- Follow a simple ASCII format. 
							- Receipt No: format it as RCPT-YYYY-MM-DD 
							- Receipt Date- use present date
							- Customer name, address and email — use `<placeholder>` if missing
							- List of items (name, quantity, unit price, line total)
							- Total Amount (net)
							- Discount, if applicable
							- Grand Total (after discount)
							- Delivery date
							- Thank-you note at the bottom

						- Formatting Notes:
						- Use a monospaced layout for the receipt block.
						- Align columns using spaces (not tabs).
						- Keep the width readable (max ~80 characters).
						- Separate sections with dashed lines or whitespace.

                        - Populate the fields 
                            - `answer` should contain - `Receipt No`, `Total Amount`, `Discount` and `Grand Total` 
                            - `ok_to_proceed` with True if answer id valid or False otherwise
                            of the return JSON object as mentioned below

						- Example of the receipt format (shortened):
						Receipt No: RCPT-2025-MM-DD
						Date: 2025-12-06

						Name: Jay Kumar
						Address: <placeholder>
						Email: jay@example.com

						Items:
						Qty Description Unit Price Line Total

						500 A4 Paper	 0.50 		50.00

						Subtotal: 50.00
						Discount (10%): -5.00
						Total Amount Due: 45.00

						Expected Delivery Date: 2025-12-07

						Thank you for your business with Beaver's !!!
						
						Always be empathetic and helpful to the customer.

                        - Output Format:
                            - Form a string with details of - amswer, item_name's, quantity's, unit, stock, requested_delivery_date, ok_to_proceed in the below format 
                            - "answer: {answer}, items: {item_name}, quantity: {quantity}, {unit}: unit, stock: {stock}, {requested_delivery_date}: requested_delivery_date, total_price: {total_price},{ok_to_proceed}: ok_to_proceed"
                        """

class WorkflowContext(BaseModel):
    """Shared context between agents"""
    request_id: str
    original_request: str

class MultiAgentWorkflow:
    def __init__(self):
        self.agents = {
            "orchestrator": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model,
                name="BeaverOrchestratorAgent",
                model_settings=shared_model_settings,
                system_prompt=ORCHESTRATOR_PROMPT,
                output_type=BeaverOrchestrator,
            ),

            "extractor": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model,
                name="ExtractionAgent",
                model_settings=shared_model_settings,
                system_prompt=EXTRACTION_PROMPT,
                output_type=str,
                tools=extraction_tools,
            ),

            "inventory": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model, 
                name="InventoryAgent",
                model_settings=shared_model_settings,
                system_prompt=INVENTORY_PROMPT,
                output_type=InventoryAgentResponse,
                tools=inventory_tools,
            ),

            "quoting": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model,
                name="QuotingAgent",
                model_settings=shared_model_settings,
                system_prompt=QUOTING_PROMPT,
                # output_type=QuotingAgentResponse,
                output_type=str,
                tools=quoting_tools,
                retries=2
            ),

            "sales": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model, 
                name="SalesAgent",
                model_settings=shared_model_settings,
                system_prompt=SALES_PROMPT,
                # output_type=SalesAgentResponse,
                output_type=str,
                tools=sales_tools,
            ),
            
            "receipt": Agent(
                #model=OpenAIChatModel(model_name="gpt-4o-mini"),
                model=openai_model,
                name="ReceiptAgent",
                model_settings=shared_model_settings,
                system_prompt=RECEIPT_PROMPT,
                output_type=ReceiptAgentResponse,
            ),
        }

        self.agent_usage_count = {
            "extractor": 0,
            "inventory": 0,
            "quoting": 0,
            "sales": 0,
            "receipt": 0,
        }

    def process_query(self, context: WorkflowContext) -> str:
        """
        Handle customer inquiries by using the inventory agent to check 
        the stock of the item.
        """
        # = self.extract_items_and_quantities(context.original_request)
        #("Items ===> ", items)

        #if not items:
        #    return "Sorry, requested items are not available."

        #("Matched items ---> ", items)
        
        ### Extraction Agent
        # Call Extraction agent to find item_name, quantity and unit 
        extraction_prompt = f"""
            Customer Request: {context.original_request}
        """
        extraction_response = self.agents["extractor"].run_sync(
            extraction_prompt,
        )
        self.agent_usage_count["extractor"] += 1

        if not extraction_response: #.output.ok_to_proceed:
            # If quoting agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {extraction_response}") #.output.answer}")
            return extraction_response #.output.answer
        ###
        
        # Call inventory agent to generate financial report
        inventory_prompt = f"""
            Classification: QUERY
            User Request: {context.original_request}
            Extraction Context: extraction_response
        """
        inventory_response = self.agents["inventory"].run_sync(
            inventory_prompt,
        )
        self.agent_usage_count["inventory"] += 1
        return inventory_response.output.answer

    def process_order(self, context: WorkflowContext) -> str:
        
        #items = self.extract_items_and_quantities(context.original_request)
        #print("Items ===> ", items)  

        #if not items:
        #    return "Sorry, requested items are not available."

        #print("Matched items ---> ", items)
        
        ### Extraction Agent
        # Call Extraction agent to find item_name, quantity and unit 
        extraction_prompt = f"""
            Customer Request: {context.original_request}
        """
        extraction_response = self.agents["extractor"].run_sync(
            extraction_prompt,
        )
        print()
        print("CHECKING Extraction Agent ===> ", self.agents["extractor"])
        print()
        print("CHECKING Extraction Prompt ===> ", extraction_prompt)
        print()
        print("CHECKING Extraction response ===> ", extraction_response)
        print()
        self.agent_usage_count["extractor"] += 1

        if not extraction_response: #.output.ok_to_proceed:
            # If quoting agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {extraction_response}") #.output.answer}")
            return extraction_response #.output.answer
        ###
        
        inventory_prompt = f"""
            Classification: ORDER
            User Request: {context.original_request}
            Extraction Context: extraction_response
        """
        # ITEMS_JSON: {json.dumps(items, indent=2)}

        print()
        print("CHECKING Inventory Agent ===> ", self.agents["inventory"])
        print()
        print("CHECKING Inventory Agent - prompt ===> ", inventory_prompt)
        print()
        # Call inventory agent to check stock levels and handle order for stock items        
        inventory_response = self.agents["inventory"].run_sync(
            inventory_prompt,
        )
        self.agent_usage_count["inventory"] += 1
        
        print(">>>>>>>>>> inventory_response.output.answer = ", inventory_response.output.answer)
        print(">>>>>>>>>> inventory_response.ok_to_proceed = ", inventory_response.output.ok_to_proceed)

        if not inventory_response.output.ok_to_proceed:
            # If inventory agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {inventory_response.output.answer}")
            return inventory_response.output.answer
            # print("Inventory warning, but proceeding to quote... ")

        # Call quoting agent to generate a quote based on the order
        quote_prompt = f"""
            Classification: ORDER
            User Request: {context.original_request}
            Inventory Context: {inventory_response.output.answer} 
        """
        print()
        print("CHECKING Quote Agent ===> ", self.agents["quoting"])
        print()
        print("CHECKING Quote Agent - prompt ===> ",quote_prompt)
        print()

        quoting_response = self.agents["quoting"].run_sync(
            quote_prompt,
        )

        self.agent_usage_count["quoting"] += 1
        
        print("==========>>>>>> quote-> ", quoting_response)
        #print("==========>>>>>> quote-> ", quoting_response.output.answer)
        #print("==========>>>>>> quote-> ", quoting_response.output.items)
        #print("==========>>>>>> quote-> ", quoting_response.output.total_price)
        #print("==========>>>>>> quote-> ", quoting_response.output.ok_to_proceed)
        if not quoting_response: #.output.ok_to_proceed:
            # If quoting agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {inventory_response}") #.output.answer}")
            return quoting_response #.output.answer
            # print("Inventory / Quote level warning, but proceeding to sale... ")

        # Call sales agent to finalize the order
        sales_prompt = f"""
            Classification: ORDER
            User Request: {context.original_request}
            Inventory Context: {inventory_response.output.answer}
            Quoting Context: {quoting_response}
        """
        # .answer}
        sales_response = self.agents["sales"].run_sync(
            sales_prompt,
        )
        print()
        print("CHECKING Sales Agent ===> ", self.agents["sales"])
        print()
        print("CHECKING Sales Agent - prompt ===> ",sales_prompt)
        print()

        self.agent_usage_count["sales"] += 1
        
        print("==========>>>>>> quote-> ", sales_response)
        #print("CHECKPOINT-4 ---> ", sales_response.output)
        #print("==========>>>>>> quote-> ", sales_response.output.answer)
        #print("==========>>>>>> quote-> ", sales_response.output.items)
        #print("==========>>>>>> quote-> ", sales_response.output.total_price)
        #print("==========>>>>>> quote-> ", sales_response.output.ok_to_proceed)
        if not sales_response: #.output.ok_to_proceed:
            # If sales agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {inventory_response}") #.output.answer}")
            return sales_response #.output.answer
            # print("Inventory / Quote / Sale level warning, but proceeding to next level... ")

        # Call receipt agent to generate an receipt for the order
        receipt_prompt = f"""
            Classification: ORDER
            User Request: {context.original_request}
            Inventory Context: {inventory_response.output.answer}
            Quoting Context: {quoting_response} 
            Sales Context: {sales_response} 
        """
        print()
        print("CHECKING Receipt Agent ===> ", self.agents["receipt"])
        print()
        print("CHECKING Receipt Agent - prompt ===> ",receipt_prompt)
        print()

        # Discount Context: {discount_response.output.answer}
        receipt_response = self.agents["receipt"].run_sync(
            receipt_prompt,
        )
        self.agent_usage_count["receipt"] += 1
        print("CHECKPOINT-5 ---> ", receipt_response.output)
        if not receipt_response.output.ok_to_proceed:
            # If receipt agent indicates order cannot proceed, return a message
            print(f"Order cannot be processed: {inventory_response.output.answer}")
            return receipt_response.output.answer

        # Return the final response from the receipt agent
        return receipt_response.output.answer

    def run(self, customer_request: str) -> str:
        """
        Run the multi-agent workflow for a given customer request.
        This method orchestrates the agents to handle the request and return a response.
        """
        # Create workflow context
        print("CHECKPOINT-0.0 ---> ", customer_request)
        context = WorkflowContext(
            request_id=f"REQ_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            original_request=customer_request
        )

        # Step 1: Call the orchestration agent to classify the request
        orchestration_response = self.agents["orchestrator"].run_sync(
            f"User Request:\n{customer_request}"
        )

        print(f"CHECKPOINT-0.1 ---> context - ", context)
        print(f"CHECKPOINT-0.2 ---> Orchestration Agent classified request as: {orchestration_response.output.classification}")

        # Step 2: Based on classification, route to appropriate agents
        if orchestration_response.output.classification == "QUERY":
            # Handle inquiries with quoting and evaluation agents
            response = self.process_query(context)
        elif orchestration_response.output.classification == "ORDER":
            # Handle orders with inventory and sales agents
            response = self.process_order(context)
        else:
            response = "Invalid classification received from orchestration agent."
        
        return response

# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample1.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return
    ''' Duplicate Code Hence commented
    # quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
    quote_requests_sample = pd.read_csv("quote_requests_sample1.csv")

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"]
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")
    '''

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    #################### CHeck Inventory Data
    print("Printing inventory Data... ")
    print(get_all_inventory("2025-04-01"))
    print("End printing inventory Data... ")
    ####################
    
    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    
    multi_agent_workflow = MultiAgentWorkflow()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        print(f"\n<--- Request {idx + 1} --->")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = multi_agent_workflow.run(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(0.3)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n<--- FINAL FINANCIAL REPORT --->")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    print("\n<--- Agent Usage Summary --->")
    for agent, count in multi_agent_workflow.agent_usage_count.items():
        print(f"{agent.capitalize()} Agent: {count} times")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
