"""
Data preprocessing steps involved in cleaning and preparing the raw KJV text for tokenization.
"""

#### Setup & Configuration

# Imports
import requests
from io import StringIO
from pathlib import Path
import tiktoken

# Configs
DATA_PATH = Path('data')
RAW_DATA_PATH = DATA_PATH/'raw_kjv_text.txt'
PROCESSED_DATA_PATH = DATA_PATH/'processed_kjv_text.txt'

URL = 'https://www.gutenberg.org/cache/epub/10/pg10.txt'

# Ensure data folder exist
DATA_PATH.mkdir(parents=True, exist_ok=True)


#### Download Raw Text

def download_kjv_text(url, save_path):
    """Download and save the KJV text."""
    response = requests.get(url)
    save_path.write_text(response.text, encoding="utf-8")

#### Load and Extract relevant portion

def load_text(text_path):
    """Load saved text."""
    return text_path.read_text(encoding="utf-8")

def get_text_position(substring, input_text):
    """Return start and end index positions for given substring in the input text."""
    positions = {}
    start = 0
    while True:
        pos = input_text.find(substring, start)
        if pos == -1:
            break
        start = pos + len(substring)
        positions[pos] = start
    return positions

def extract_bible_text(input_text):
    """Extract needed portion from input text."""
    start_marker = 'The First Book of Moses: Called Genesis'
    end_marker = 'The grace of our Lord Jesus Christ be with you all. Amen'
    start_positions = get_text_position(start_marker, input_text)
    end_positions = get_text_position(end_marker, input_text)
    start_pos = list(start_positions.keys())[1]
    end_pos = list(end_positions.values())[-1]
    return input_text[start_pos: end_pos]

#### Clean extracted text

def remove_whitespaces(input_text):
    text = input_text.split()
    text = " ".join(text)
    return text

def remove_verses_num(input_text):
    """Remove verse numbering in text"""
    text_without_verses = []
    splitted_text = input_text.split()
    for c in splitted_text:
        if not (c.split(':')[0].isdigit()):
            text_without_verses.append(c)
    text_without_verses = " ".join(text_without_verses)
    return text_without_verses

def extract_book_titles(input_text):
    """
    Get each book title in book listing for encoding for context purpose from
    "The First Book of Moses: Called Genesis" to "The Revelation of Saint John the Divine"
    """
    book_titles = []
    first_book = 'The First Book of Moses: Called Genesis'
    last_book = 'The Revelation of Saint John the Divine'
    first_book_listing_positions = get_text_position(first_book, input_text)
    last_book_listing_positions = get_text_position(last_book, input_text)
    first_book_title_idx_pos = list(first_book_listing_positions.keys())[0]
    last_book_title_idx_pos = list(last_book_listing_positions.values())[0]
    book_title_listings = input_text[first_book_title_idx_pos:last_book_title_idx_pos]
    book_title_listings = book_title_listings.split('\n')
    excluded_titles = ['', 'The New Testament of the King James Bible']
    for book in book_title_listings :
        if (book != excluded_titles[0]) and (book!= excluded_titles[1]):
            book_titles.append(book)
    return book_titles

# Create additional context tokens for book start, end and to separate OT and NT books
# List of all OT and NT books
OT_books = [
    "GENESIS", "EXODUS", "LEVITICUS", "NUMBERS", "DEUTERONOMY",
    "JOSHUA", "JUDGES", "RUTH", "1SAMUEL", "2SAMUEL",
    "1KINGS", "2KINGS", "1CHRONICLES", "2CHRONICLES", "EZRA",
    "NEHEMIAH", "ESTHER", "JOB", "PSALMS", "PROVERBS",
    "ECCLESIASTES", "SONGOFSONGS", "ISAIAH", "JEREMIAH", "LAMENTATIONS",
    "EZEKIEL", "DANIEL", "HOSEA", "JOEL", "AMOS",
    "OBADIAH", "JONAH", "MICAH", "NAHUM", "HABAKKUK",
    "ZEPHANIAH", "HAGGAI", "ZECHARIAH", "MALACHI"]

NT_books = [
    "MATTHEW", "MARK", "LUKE", "JOHN", "ACTS",
    "ROMANS", "1CORINTHIANS", "2CORINTHIANS", "GALATIANS", "EPHESIANS",
    "PHILIPPIANS", "COLOSSIANS", "1THESSALONIANS", "2THESSALONIANS",
    "1TIMOTHY", "2TIMOTHY", "TITUS", "PHILEMON", "HEBREWS",
    "JAMES", "1PETER", "2PETER", "1JOHN", "2JOHN", "3JOHN",
    "JUDE", "REVELATION"]

# Special tokens
testament_tokens = ["<TESTAMENT_START=OT>", "<TESTAMENT_END=OT>", "<TESTAMENT_START=NT>", "<TESTAMENT_END=NT>"]
book_start_tokens = [f"<BOOK_START={b}>" for b in OT_books + NT_books]
book_end_tokens =  [f"<BOOK_END={b}>" for b in OT_books + NT_books]

def include_text(main_text, insert_pos, text_to_include):
    """Include text at specific position in main text using StringIO."""
    output = StringIO()
    output.write(main_text[:insert_pos])
    output.write(text_to_include + ' ' + main_text[insert_pos:])
    modified_text = output.getvalue()
    return modified_text

# MAIN EXECUTION PIPELINE
def main():
    if not RAW_DATA_PATH.exists():
        download_kjv_text(URL, RAW_DATA_PATH)
    print("Preprocessing started!")
    raw_text = RAW_DATA_PATH.read_text(encoding="utf-8")
    extracted_text = extract_bible_text(raw_text)
    cleaned_text = remove_whitespaces(extracted_text)
    cleaned_text = remove_verses_num(cleaned_text)
    book_titles = extract_book_titles(raw_text)

    # Replace book titles at the begining of each book with its corresponding book_start_tokens
    for i in zip(book_titles, book_start_tokens):
        cleaned_text = cleaned_text.replace(i[0], i[1])

    # Include book end token at necessary locations
    for i in zip(book_start_tokens[1:], book_end_tokens[:-1]):
        insert_pos = cleaned_text.find(i[0])
        cleaned_text = include_text(cleaned_text, insert_pos, i[1])

    # Include book end token for Revelation
    cleaned_text = include_text(cleaned_text, len(cleaned_text), ' <BOOK_END=REVELATION>')

    # Include end of NT token -> <TESTAMENT_END=NT>
    cleaned_text = include_text(cleaned_text, len(cleaned_text), ' <TESTAMENT_END=NT>')

    # Append testament related tokens to signify start of OT and NT for context
    # "<TESTAMENT=OT>" is placed in the begining of the entire text
    cleaned_text = include_text(cleaned_text, 0, '<TESTAMENT_START=OT>')

    # Include start of NT token -> <TESTAMENT_START=NT>
    # NT introductory text in text body
    nt_intro_text = '*** The New Testament of the King James Bible'
    cleaned_text = cleaned_text.replace(nt_intro_text, '')
    nt_start_pos = cleaned_text.find('<BOOK_END=MALACHI>') + len('<BOOK_END=MALACHI>')
    cleaned_text = include_text(cleaned_text, nt_start_pos+1, '<TESTAMENT_START=NT>')

    # Include end of OT token -> <TESTAMENT_END=OT>
    nt_token_pos = cleaned_text.find('<TESTAMENT_START=NT>')
    cleaned_text = include_text(cleaned_text, nt_token_pos-1, ' <TESTAMENT_END=OT>')

    #### Save Processed text
    PROCESSED_DATA_PATH.write_text(cleaned_text, encoding="utf-8")
    print("Preprocessing complete!")

if __name__ == "__main__": 
    main()
