# main.py
#
# Un-LOCC: Universal Lossy Optical Context Compression
# A tool for testing and replicating the findings of the Un-LOCC Research using Optical Needle-in-a-Haystack (O-NIH) evaluation.

import os
import random
import base64
import time
import json
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# --- CONFIGURATION ---
# OpenRouter API key can be provided via command line or environment variable
client = None

# --- CORE METHODOLOGY FUNCTIONS ---

def generate_needle(length=9):
    """
    Generates a random, unique, and visually unambiguous code (the 'needle').
    The character set is chosen to avoid common OCR confusion (e.g., 'O' vs '0', 'I' vs '1').
    """
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    result = ''
    for i in range(length):
        result += random.choice(chars)
        if (i + 1) % 3 == 0 and i < length - 1:
            result += '-'
    return result

def prepare_text(source_text, word_count):
    """
    Selects a random contiguous block of `word_count` words from the source_text
    and injects a randomly generated 'needle' into it.
    """
    words = source_text.split()
    if len(words) < word_count:
        raise ValueError(f"Source text has only {len(words)} words, but {word_count} were requested.")
    start_index = random.randint(0, len(words) - word_count)
    text_chunk = words[start_index : start_index + word_count]
    needle = generate_needle()
    injection_index = random.randint(1, len(text_chunk) - 1)
    text_chunk.insert(injection_index, needle)
    haystack = " ".join(text_chunk)
    return haystack, needle

def normalize_for_ocr(text):
    """
    Normalizes a string to account for common OCR errors, making the
    accuracy comparison more robust and "fuzzy".
    """
    replacements = {
        'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5',
        'B': '8', 'A': '4', '-': '', ' ': ''
    }
    text = text.upper()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two strings."""
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_similarity(s1, s2):
    """Calculates a normalized fuzzy similarity score (0.0 to 1.0)."""
    norm_s1 = normalize_for_ocr(s1)
    norm_s2 = normalize_for_ocr(s2)
    distance = levenshtein_distance(norm_s1, norm_s2)
    max_len = max(len(norm_s1), len(norm_s2))
    if max_len == 0: return 1.0
    return (max_len - distance) / max_len

def _wrap_text(text, font, max_width):
    """Helper function to wrap text into lines that fit within a max_width."""
    lines = []
    words = text.split()
    if not words: return []
    current_line = words[0]
    for word in words[1:]:
        if font.getlength(current_line + " " + word) <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def query_llm_for_code(image_path, model_id, client):
    """Sends the generated image to a VLM via OpenRouter and returns the response."""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_image}"
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "The document in the image contains a unique alphanumeric verification code formatted like XXX-XXX-XXX. Your task is to find this code. Strictly output *only* the code itself. Do not include any other text, explanation, or preamble."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  !! API Error: {e}")
        return ""

def find_max_word_count(source_words, font_path, font_size, size, padding):
    """
    Calculates the maximum number of words that can fit in an image for a given configuration.
    This is done by performing a binary search on the word count.
    """
    print(f"\n[INFO] Calculating maximum word capacity for {os.path.basename(font_path)}@{font_size}px on {size[0]}x{size[1]} image...")
    font = ImageFont.truetype(font_path, font_size)
    drawable_width = size[0] - 2 * padding
    drawable_height = size[1] - 2 * padding
    
    low_wc, high_wc = 1, len(source_words)
    max_words_fit = 0
    while low_wc <= high_wc:
        mid_wc = (low_wc + high_wc) // 2
        test_text = " ".join(source_words[:mid_wc])
        lines = _wrap_text(test_text, font, drawable_width)
        
        _, top, _, bottom = font.getbbox("A")
        line_height = (bottom - top) + (font.size // 4)
        total_block_height = len(lines) * line_height

        if total_block_height <= drawable_height:
            max_words_fit = mid_wc
            low_wc = mid_wc + 1
        else:
            high_wc = mid_wc - 1
            
    print(f"[INFO] Calculation complete. Max capacity: {max_words_fit} words.")
    return max_words_fit

def generate_image(text, font_path, font_size, size, padding):
    """
    Generates and saves a PNG image with the given text rendered at a fixed font size.
    """
    width, height = size
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    drawable_width = width - 2 * padding
    
    font = ImageFont.truetype(font_path, font_size)
    lines = _wrap_text(text, font, drawable_width)
    
    _, top, _, bottom = font.getbbox("A")
    line_height = (bottom - top) + (font_size // 4)
    
    y_text = padding
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill='black')
        y_text += line_height

    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    temp_path = f"{output_dir}/test_image_{int(time.time())}.png"
    image.save(temp_path)
    return temp_path

# --- MAIN EXECUTION LOGIC ---

def run_experiment(args, client):
    """
    Main function to run the O-NIH experiment based on command-line arguments.
    """
    # --- 1. Load and Validate Inputs ---
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"Font file not found at: {args.font_path}")
    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus file not found at: {args.corpus_path}")
        
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        source_text = f.read()
    
    # --- 2. Determine Word Count ---
    if args.word_count:
        word_count = args.word_count
    else:
        # If word_count is not provided, we must calculate the maximum possible.
        if not args.font_size:
            raise ValueError("Must provide --font_size when --word_count is not specified.")
        word_count = find_max_word_count(
            source_words=source_text.split(),
            font_path=args.font_path,
            font_size=args.font_size,
            size=args.size,
            padding=args.padding
        )
    
    if word_count == 0:
        print("\n[ERROR] Could not fit any words in the image with the given configuration. Aborting.")
        return

    # --- 3. Run the Test Loop ---
    accuracies = []
    print("\n" + "="*50)
    print("--- Starting Optical Needle-in-a-Haystack Test ---")
    print(f"Model:          {args.model_id}")
    print(f"Image Size:     {args.size[0]}x{args.size[1]}px")
    print(f"Font:           {os.path.basename(args.font_path)}")
    print(f"Font Size:      {args.font_size}px (used for calculation)")
    print(f"Word Count:     {word_count} (used for generation)")
    print(f"Number of Runs: {args.num_tests}")
    print("="*50 + "\n")

    for i in range(args.num_tests):
        print(f"--> Running test {i + 1}/{args.num_tests}...")
        
        haystack, needle = prepare_text(source_text, word_count)
        
        # We need to find the font size to render with. If not provided, we need to search for it.
        # This part is complex and was simplified for the main script. For true replication of the paper,
        # the fixed-font-size method is preferred. This script defaults to that superior method.
        if not args.font_size:
             raise ValueError("This script is designed for the fixed-font-size methodology. Please provide a --font_size.")
        
        image_path = generate_image(haystack, args.font_path, args.font_size, args.size, args.padding)
        print(f"  - Image generated at: {image_path}")
        
        start_time = time.time()
        llm_response = query_llm_for_code(image_path, args.model_id, client)
        duration = time.time() - start_time
        print(f"  - LLM responded in {duration:.2f}s")
        
        accuracy = fuzzy_similarity(needle, llm_response)
        accuracies.append(accuracy)
        
        print(f"  - Expected: {needle}")
        print(f"  - Received: {llm_response}")
        print(f"  - Fuzzy Accuracy: {accuracy:.2%}\n")
    
    # --- 4. Print Final Results ---
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print("-" * 50)
    print("--- Test Complete ---")
    print(f"Average Fuzzy Accuracy over {args.num_tests} runs: {average_accuracy:.2%}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Un-LOCC Optical Needle-in-a-Haystack (O-NIH) test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key. If not provided, will use OPENROUTER_API_KEY environment variable."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="qwen/qwen2.5-vl-72b-instruct",
        help="The OpenRouter model ID to test (e.g., 'google/gemini-2.5-flash')."
    )
    parser.add_argument(
        "--font_path", 
        type=str,
        default="fonts/AtkinsonHyperlegible-Regular.ttf", 
        help="Path to the .ttf font file to use for rendering."
    )
    parser.add_argument(
        "--font_size", 
        type=int, 
        default=14,
        help="The fixed font size (in pixels) to render the text with."
    )
    parser.add_argument(
        "--size", 
        type=int, 
        nargs=2, 
        default=[864, 864], 
        help="The dimensions of the output image (width height)."
    )
    parser.add_argument(
        "--word_count", 
        type=int, 
        default=None, 
        help="Number of words for the haystack. If not provided, it will be automatically calculated to be the maximum possible for the given configuration."
    )
    parser.add_argument(
        "--num_tests", 
        type=int, 
        default=10, 
        help="Number of times to run the test to get an average accuracy."
    )
    parser.add_argument(
        "--corpus_path", 
        type=str, 
        default="corpus.txt", 
        help="Path to the text file to use as the source for the haystack."
    )
    parser.add_argument(
        "--padding", 
        type=int, 
        default=20, 
        help="Padding in pixels inside the image."
    )
    
    args = parser.parse_args()
    
    # Convert size to a tuple
    args.size = tuple(args.size)
    
    # Initialize OpenRouter client
    OPENROUTER_API_KEY = None
    if args.api_key:
        OPENROUTER_API_KEY = args.api_key
    else:
        OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        print("Error: OpenRouter API key not found. Please provide via --api_key or set OPENROUTER_API_KEY environment variable.")
        exit()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    try:
        run_experiment(args, client)
    except FileNotFoundError as e:
        print(f"\n[ERROR] A required file was not found: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")