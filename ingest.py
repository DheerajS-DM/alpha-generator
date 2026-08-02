import os
import time
import logging
import pandas as pd
import yfinance as yf
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pytickersymbols import PyTickerSymbols
from datetime import datetime, date, timedelta

# --- Configuration ---
OUTPUT_DIR = "data/raw"
MAX_WORKERS = 8
RETRY_LIMIT = 3

STANDARD_COLUMNS = {
    'date': 'date',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'adj_close': 'adj_close',
    'adj close': 'adj_close',
    'volume': 'volume'
}

CANONICAL_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']

# Define the Global Indices to target
# pytickersymbols natively supports major US and European indices.
TARGET_INDICES = [
    'S&P 500',      # US Large Cap
    'NASDAQ 100',   # US Tech
    'DAX',          # Germany
    'CAC 40',       # France
    'SMI',          # Switzerland
    'AEX'           # Netherlands
]

# Manual additions for major Asian/Global stocks not covered natively by pytickersymbols
# Note: Yahoo Finance uses suffixes for international markets (e.g., .T for Tokyo, .HK for Hong Kong)
SUPPLEMENTARY_ASIAN_TICKERS = [
    '7203.T', # Toyota (Japan)
    '6758.T', # Sony (Japan)
    '9984.T', # SoftBank (Japan)
    '0700.HK', # Tencent (Hong Kong)
    '9988.HK', # Alibaba (Hong Kong)
    'TSM',     # TSMC (Taiwan - US ADR)
    '005930.KS'# Samsung (South Korea)
]
# ---------------------

logging.basicConfig(filename='ingestion_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(message)s')

def get_global_tickers():
    """
    Dynamically fetches Yahoo Finance symbols for targeted global indices.
    """
    stock_data = PyTickerSymbols()
    tickers = set()
    
    print("Fetching dynamic ticker lists from global indices...")
    for index_name in TARGET_INDICES:
        try:
            stocks = stock_data.get_stocks_by_index(index_name)
            added_count = 0
            for stock in stocks:
                for symbol in stock['symbols']:
                    if symbol['yahoo']:
                        tickers.add(symbol['yahoo'])
                        added_count += 1
            print(f" -> {index_name}: Extracted {added_count} tickers.")
        except Exception as e:
            logging.error(f"Failed to fetch index {index_name}: {e}")
            print(f" -> {index_name}: Failed to fetch (Check logs).")

    # Add our supplementary Asian markets
    tickers.update(SUPPLEMENTARY_ASIAN_TICKERS)
    print(f" -> Custom Asian/Global Additions: Added {len(SUPPLEMENTARY_ASIAN_TICKERS)} tickers.")
    
    return list(tickers)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns.values]

    df = df.reset_index()
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

    rename_map = {}
    for col in df.columns:
        normalized = col.lower().strip()

        if normalized in STANDARD_COLUMNS:
            rename_map[col] = STANDARD_COLUMNS[normalized]
            continue

        if normalized.startswith('date'):
            rename_map[col] = 'date'
            continue

        for canonical in CANONICAL_COLUMNS:
            if normalized == canonical:
                rename_map[col] = canonical
                break
            if normalized.startswith(f"{canonical}_") or normalized.endswith(f"_{canonical}"):
                rename_map[col] = canonical
                break
            if normalized.replace('.', '_').startswith(f"{canonical}_"):
                rename_map[col] = canonical
                break

    df = df.rename(columns=rename_map)

    if 'date' not in df.columns:
        for col in df.columns:
            if 'date' in col:
                df = df.rename(columns={col: 'date'})
                break

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c in ('index', 'unnamed:_0', 'unnamed:_0.1')], errors='ignore')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    return df


def download_yfinance_data(ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame | None:
    kwargs = {'interval': '1d', 'progress': False}
    if start is None:
        kwargs['period'] = 'max'
    else:
        kwargs['start'] = start.strftime('%Y-%m-%d')
        kwargs['end'] = end.strftime('%Y-%m-%d') if end is not None else date.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, **kwargs)
    if df.empty:
        logging.error(f"{ticker} returned empty dataset.")
        return None

    df = normalize_columns(df)

    if 'date' not in df.columns:
        logging.error(f"{ticker} download missing date column: {df.columns.tolist()}")
        return None

    columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'adj_close' in df.columns:
        columns_to_keep.append('adj_close')

    df = df[[col for col in columns_to_keep if col in df.columns]]
    df = df.drop_duplicates(subset=['date']).sort_values('date')
    return df


def existing_data_range(file_path: str) -> tuple[date | None, date | None]:
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Unable to read existing parquet for {file_path}: {e}")
        return None, None

    date_col = None
    for col in df.columns:
        if col.lower().startswith('date'):
            date_col = col
            break

    if date_col is None:
        return None, None

    df = df.with_columns(pl.col(date_col).cast(pl.Date).alias('date'))
    if df.height == 0:
        return None, None

    return df['date'].min().to_pandas().date(), df['date'].max().to_pandas().date()


def process_ticker(ticker):
    """
    Worker function: downloads missing data or full history, normalizes columns, saves to Parquet.
    """
    file_path = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
    start_date = None
    end_date = date.today() + timedelta(days=1)

    if os.path.exists(file_path):
        first_date, last_date = existing_data_range(file_path)
        if first_date is not None and last_date is not None:
            if last_date >= date.today():
                return True
            start_date = last_date + timedelta(days=1)
            if start_date >= date.today():
                return True
            logging.info(f"Updating {ticker} from {start_date} to {date.today()}")
        else:
            logging.info(f"Existing {ticker} file is invalid or incomplete, re-downloading full history.")
            start_date = None

    for attempt in range(RETRY_LIMIT):
        try:
            df = download_yfinance_data(ticker, start=start_date, end=end_date)
            if df is None or df.empty:
                return False

            pldf = pl.from_pandas(df)
            pldf = pldf.rename({col: col.lower().replace(' ', '_') for col in pldf.columns})
            if 'date' in pldf.columns:
                pldf = pldf.with_columns(pl.col('date').cast(pl.Date))

            canonical_cols = [c for c in CANONICAL_COLUMNS if c in pldf.columns]
            pldf = pldf.select(canonical_cols)
            pldf.write_parquet(file_path, compression='zstd')
            return True

        except Exception as e:
            logging.error(f"Failed to process {ticker}: {e}")
            if '429' in str(e):
                time.sleep(5 * (attempt + 1))
            else:
                break

    return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dynamically Generate the Universe
    tickers = get_global_tickers()
    
    # Optional: Save a snapshot of the current universe for your backtester's reference
    pl.DataFrame({"Symbol": tickers}).write_csv(os.path.join(OUTPUT_DIR, "universe_snapshot.csv"))
    
    print(f"\nStarting ingestion for {len(tickers)} unique tickers...")
    print(f"Data will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    success_count = 0
    
    # 2. Thread Pool Execution (RAM-Safe Architecture)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_ticker, t): t for t in tickers}
        
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Downloading"):
            if future.result():
                success_count += 1

    print(f"\nIngestion Complete. Successfully saved {success_count}/{len(tickers)} files.")
    print("Check 'ingestion_errors.log' for any failures.")

if __name__ == "__main__":
    import pandas as pd 
    main()