"""
Training custom tokenizer on kjv text using huggingface AutoTokenizer
"""
from pathlib import Path
from transformers import AutoTokenizer


def main():
    # Read processed txt in verbum-mini/data
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent 
    DATA_PATH = REPO_ROOT / "data" / "processed_kjv_text.txt"
    kjv_text = DATA_PATH.read_text(encoding="utf-8")

    # Split text into lines (iterator for tokenizer training)
    kjv_lines = kjv_text.split(".")

    # --- Create needed special tokens ---

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

    # Testament tokens
    testament_tokens = [
        "<TESTAMENT_START=OT>", "<TESTAMENT_END=OT>",
        "<TESTAMENT_START=NT>", "<TESTAMENT_END=NT>"]

    # Book start/end tokens
    book_start_tokens = [f"<BOOK_START={b}>" for b in OT_books + NT_books]
    book_end_tokens = [f"<BOOK_END={b}>" for b in OT_books + NT_books]

    # Common KJV words to keep as single tokens
    kjv_common_words = [
        "thou", "thee", "thine", "ye", "doth", "saith",
        "God", "Lord", "Jesus", "Christ", "Spirit",
        "heaven", "earth",  "Israel"]

    # Merge all special tokens
    special_tokens = testament_tokens + book_start_tokens + book_end_tokens + kjv_common_words

    # Load pretrained tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Add special tokens
    old_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Train new tokenizer
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=kjv_lines,
        vocab_size=10000)

    # Save the tokenizer
    new_tokenizer.save_pretrained(REPO_ROOT/'kjv_tokenizer')
    print(f"Saved tokenizer to {REPO_ROOT}")

if __name__ == "__main__": 
    main() 