import os
import pandas as pd
import gradio as gr
import torch
import re
import time
import gc
from PIL import Image
import traceback
from typing import List, Dict, Any, Union, Optional, Tuple
import threading
from tabulate import tabulate
import tempfile
import shutil

# Import transformers modules
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("Error: Could not import Qwen2_5_VLForConditionalGeneration")
    print("Please install transformers from source:")
    print("pip install git+https://github.com/huggingface/transformers")

# Global variables for tracking progress
total_images = 0
processed_images = 0
successful_images = 0
failed_images = 0
print_lock = threading.Lock()
model = None
processor = None

# =============== QWEN BATCH EXTRACTOR FUNCTIONS ===============

def load_image(image_path: str, max_size: int = 1024) -> Image.Image:
    """
    Load an image from a file path and resize it if needed to save memory.
    
    Args:
        image_path: Path to the image
        max_size: Maximum dimension (width or height) for the image
    
    Returns:
        Resized PIL Image
    """
    try:
        image = Image.open(image_path)
        
        # Resize large images to save memory while maintaining aspect ratio
        width, height = image.size
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to load or resize image {image_path}: {str(e)}")


def process_vision_info(messages: List[Dict[str, Any]]) -> tuple:
    """Extract image inputs from messages."""
    image_inputs = []
    video_inputs = None  # Setting to None instead of empty list
    
    for message in messages:
        if message["role"] != "user":
            continue
        
        for content in message["content"]:
            if content["type"] == "image":
                if isinstance(content["image"], str):
                    # Load image if it's a path or URL
                    image = load_image(content["image"])
                    image_inputs.append(image)
                else:
                    # Assume it's already a PIL Image
                    image_inputs.append(content["image"])
    
    return image_inputs, video_inputs


def extract_fields_from_response(response: str) -> Tuple[str, str, str]:
    """
    Extract name, affiliation, and town from the model's response.
    
    Args:
        response: The response from the model
    
    Returns:
        Tuple containing (name, affiliation, town)
    """
    # Initialize default values
    name = ""
    affiliation = ""
    town = ""
    
    # Use regex to extract fields
    name_match = re.search(r"Name:\s*([^\n]+)", response)
    affiliation_match = re.search(r"Affiliation:\s*([^\n]+)", response)
    town_match = re.search(r"Town:\s*([^\n]+)", response)
    
    # Extract fields if matches found
    if name_match:
        name = name_match.group(1).strip()
    if affiliation_match:
        affiliation = affiliation_match.group(1).strip()
    if town_match:
        town = town_match.group(1).strip()
    
    return name, affiliation, town


def process_single_image(image_path: str, model, processor, device: str, 
                         prompt: str, max_image_size: int, max_tokens: int,
                         progress=None) -> Dict:
    """
    Process a single image and extract name, affiliation, and town.
    
    Args:
        image_path: Path to the image
        model: The loaded Qwen model
        processor: The loaded processor
        device: Device to run inference on ("cuda" or "cpu")
        prompt: Text prompt to send to the model
        max_image_size: Maximum dimension for input images
        max_tokens: Maximum number of tokens to generate
        progress: Gradio progress object
    
    Returns:
        Dictionary with extracted fields and metadata
    """
    global processed_images, successful_images, failed_images
    
    result = {
        "image_path": image_path,
        "name": "",
        "affiliation": "",
        "town": "",
        "success": False,
        "error": "",
        "time_taken": 0,
        "response": ""
    }
    
    try:
        t0 = time.time()
        
        # Load and prepare image
        image = load_image(image_path, max_size=max_image_size)
        
        # Create message format expected by Qwen models
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Check if video_inputs is None, and handle accordingly
        if video_inputs is None:
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        
        # Move inputs to the appropriate device
        inputs = inputs.to(device)
        
        # Free some memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Generate response with memory optimizations
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Use greedy decoding to save memory
                "use_cache": True,
            }
            
            generated_ids = model.generate(
                **inputs,
                **generate_kwargs
            )
        
        # Decode only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]  # Get first (and only) response
        
        time_taken = time.time() - t0
        
        # Extract fields from response
        name, affiliation, town = extract_fields_from_response(response)
        
        # Update result dictionary
        result["name"] = name
        result["affiliation"] = affiliation
        result["town"] = town
        result["success"] = True
        result["time_taken"] = time_taken
        result["response"] = response
        
        with print_lock:
            processed_images += 1
            successful_images += 1
            if progress is not None:
                progress(processed_images / total_images, f"Processed: {processed_images}/{total_images} (Success: {successful_images}, Failed: {failed_images})")
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        with print_lock:
            processed_images += 1
            failed_images += 1
            if progress is not None:
                progress(processed_images / total_images, f"Processed: {processed_images}/{total_images} (Success: {successful_images}, Failed: {failed_images})")
        
        result["error"] = error_msg
        result["time_taken"] = time.time() - t0
    
    # Clean up to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return result


def load_model_and_processor(model_name, device, half_precision):
    """Load model and processor for vision processing"""
    global model, processor
    
    # Set up dtype for model loading
    if half_precision and device == "cuda":
        dtype = torch.float16
    else:
        dtype = "auto"
    
    # Low memory options for CUDA
    attn_implementation = "sdpa" if device == "cuda" else None
    
    # Load model and processor
    print(f"Loading {model_name} model...")
    t0 = time.time()
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
            max_memory={0: "10GiB"} if device == "cuda" else None,  # Limit GPU memory usage
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        print(f"Model loaded in {time.time() - t0:.2f} s")
        
        return True, f"Model loaded successfully in {time.time() - t0:.2f}s"
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        return False, error_msg


def process_directory(directory_path: str, output_csv: str, model_name: str, 
                     prompt: str, device: str, half_precision: bool,
                     max_image_size: int, max_tokens: int, progress=None) -> List[Dict]:
    """
    Process all images in a directory and save results to CSV.
    
    Args:
        directory_path: Path to directory containing images
        output_csv: Path to output CSV file
        model_name: Name of the Qwen model to use
        prompt: Text prompt to send to the model
        device: Device to run inference on ("auto", "cuda", or "cpu")
        half_precision: Whether to use half precision for model
        max_image_size: Maximum dimension for input images
        max_tokens: Maximum number of tokens to generate
        progress: Gradio progress object
    
    Returns:
        List of results for each image
    """
    global total_images, processed_images, successful_images, failed_images, model, processor
    
    # Reset counters
    total_images = 0
    processed_images = 0
    successful_images = 0
    failed_images = 0
    
    # Validate directory
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Find all image files in directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    image_files = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and 
        f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")
    
    total_images = len(image_files)
    
    if progress is not None:
        progress(0, f"Found {total_images} images to process")
    
    # Enable garbage collection
    gc.enable()
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model is already loaded
    if model is None or processor is None:
        model_map = {
            "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
            "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
        success, message = load_model_and_processor(model_map[model_name], device, half_precision)
        if not success:
            return [], message
    
    results = []
    
    # Process images sequentially
    for i, image_path in enumerate(image_files):
        if progress is not None:
            progress(i / total_images, f"Processing image {i+1}/{total_images}: {os.path.basename(image_path)}")
        
        result = process_single_image(
            image_path, model, processor, device, 
            prompt, max_image_size, max_tokens, progress
        )
        results.append(result)
    
    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        import csv
        fieldnames = ['image_path', 'name', 'affiliation', 'town', 'success', 'error', 'time_taken']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            # Create a copy without the 'response' field for CSV output
            csv_result = {k: v for k, v in result.items() if k != 'response'}
            writer.writerow(csv_result)
    
    # Create summary
    summary = f"""
    Processing complete!
    Total images processed: {total_images}
    Successful extractions: {successful_images}
    Failed extractions: {failed_images}
    Results saved to: {output_csv}
    """
    
    if progress is not None:
        progress(1.0, f"Complete! Processed {total_images} images: {successful_images} successful, {failed_images} failed")
    
    return results, summary


# =============== DATA ANALYZER FUNCTIONS ===============

def load_data(file_path):
    """Load data from CSV file."""
    if not os.path.exists(file_path):
        return None, f"Error: File '{file_path}' not found."
    
    try:
        # Load CSV file with headers for name, affiliation, town
        df = pd.read_csv(file_path)
        
        # Ensure expected columns exist
        required_columns = ['name', 'affiliation', 'town']
        if not all(col.lower() in map(str.lower, df.columns) for col in required_columns):
            return None, f"Error: CSV must contain columns for name, affiliation, and town."
            
        # Standardize column names (case insensitive)
        column_map = {}
        for col in df.columns:
            if col.lower() == 'name':
                column_map[col] = 'name'
            elif col.lower() == 'affiliation':
                column_map[col] = 'affiliation'
            elif col.lower() == 'town':
                column_map[col] = 'town'
        
        df = df.rename(columns=column_map)
        
        # Convert all string columns to lowercase for case-insensitive operations
        for col in ['name', 'affiliation', 'town']:
            if df[col].dtype == object:  # Check if column contains strings
                df[col] = df[col].str.lower()
            
        return df, "Data loaded successfully"
    
    except Exception as e:
        return None, f"Error loading CSV file: {e}"

def summary_by_town(df):
    """Generate summary statistics by town - improved formatting."""
    if df is None or len(df) == 0:
        return "No data available for summary."
    
    town_summary = df.groupby('town').agg(
        total_people=('name', 'count'),
        affiliations=('affiliation', lambda x: len(set(x)))
    ).reset_index()
    
    town_summary = town_summary.sort_values('total_people', ascending=False)
    
    # Better column formatting
    display_summary = town_summary.copy()
    display_summary['town'] = display_summary['town'].str.title()
    display_summary.columns = ['Town', 'People', 'Affiliations']
    
    result = "\n" + "="*50 + "\n"
    result += "SUMMARY BY TOWN\n"
    result += "="*50 + "\n"
    
    result += tabulate(
        display_summary, 
        headers='keys', 
        tablefmt='psql',
        showindex=False,
        floatfmt='.0f'
    )
    
    # Display top affiliations for each town
    result += "\n\n" + "="*50 + "\n"
    result += "TOP AFFILIATIONS BY TOWN\n"
    result += "="*50 + "\n"
    
    for town in town_summary['town']:
        town_data = df[df['town'] == town]
        top_affiliations = town_data['affiliation'].value_counts().head(3)
        
        result += f"\n🏙️  {town.upper()}:\n"
        result += "   " + "-"*30 + "\n"
        
        for rank, (affiliation, count) in enumerate(top_affiliations.items(), 1):
            result += f"   {rank}. {affiliation.title():<20} → {count} people\n"
        
        if len(top_affiliations) == 0:
            result += "   No data available\n"
    
    return result

def summary_by_affiliation(df):
    """Generate summary statistics by affiliation - improved version of your current function."""
    if df is None or len(df) == 0:
        return "No data available for summary."
    
    affiliation_summary = df.groupby('affiliation').agg(
        total_people=('name', 'count'),
        towns=('town', lambda x: len(set(x)))
    ).reset_index()
    
    affiliation_summary = affiliation_summary.sort_values('total_people', ascending=False)
    
    # Better column formatting
    display_summary = affiliation_summary.copy()
    display_summary['affiliation'] = display_summary['affiliation'].str.title()
    display_summary.columns = ['Affiliation', 'People', 'Towns']
    
    result = "\n" + "="*50 + "\n"
    result += "SUMMARY BY AFFILIATION\n"
    result += "="*50 + "\n"
    
    # Use 'psql' format for better readability
    result += tabulate(
        display_summary, 
        headers='keys', 
        tablefmt='psql',  # Changed from 'simple' to 'psql'
        showindex=False,
        floatfmt='.0f'
    )
    
    # Display top towns for each affiliation
    result += "\n\n" + "="*50 + "\n"
    result += "TOP TOWNS BY AFFILIATION\n"
    result += "="*50 + "\n"
    
    for affiliation in affiliation_summary['affiliation'].head(5).tolist():
        affiliation_data = df[df['affiliation'] == affiliation]
        top_towns = affiliation_data['town'].value_counts().head(3)
        
        result += f"\n🏛️  {affiliation.upper()}:\n"
        result += "   " + "-"*30 + "\n"
        
        for rank, (town, count) in enumerate(top_towns.items(), 1):
            result += f"   {rank}. {town.title():<20} → {count} people\n"
        
        if len(top_towns) == 0:
            result += "   No data available\n"
    
    return result


def search_data(df, search_term, search_field=None):
    """Search for records by name, town, or affiliation."""
    if df is None or len(df) == 0:
        return "No data available for search."
    
    if not search_term:
        return "Please enter a search term."
    
    search_term = search_term.lower()  # Convert search term to lowercase for case-insensitive matching
    
    if search_field and search_field.lower() in ['name', 'town', 'affiliation']:
        # Search in specific field
        field = search_field.lower()
        results = df[df[field].str.contains(search_term, na=False)]
    else:
        # Search in all fields
        results = df[
            df['name'].str.contains(search_term, na=False) |
            df['town'].str.contains(search_term, na=False) |
            df['affiliation'].str.contains(search_term, na=False)
        ]
    
    if len(results) == 0:
        return f"No results found for '{search_term}'"
    else:
        # Format results for display, converting back to title case for readability
        display_results = results.copy()
        for col in ['name', 'town', 'affiliation']:
            display_results[col] = display_results[col].str.title()
        
        # Only select the columns we want to display
        display_results = display_results[['name', 'affiliation', 'town']]
        
        result = f"=== SEARCH RESULTS ({len(results)} matches) ===\n"
        result += tabulate(display_results, headers='keys', tablefmt='simple', showindex=False)
        return result


# =============== GRADIO APP INTERFACE ===============

def copy_to_temp_dir(file_list):
    """Copy uploaded files to a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for file in file_list:
        file_name = os.path.basename(file.name)
        dst_path = os.path.join(temp_dir, file_name)
        shutil.copy(file.name, dst_path)
        file_paths.append(dst_path)
    
    return temp_dir, file_paths


def unload_model():
    """Unload the model to free up GPU memory"""
    global model, processor
    
    if model is not None:
        del model
        model = None
    
    if processor is not None:
        del processor
        processor = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return "Model unloaded successfully"


def process_images_tab(files, model_name, prompt, device, half_precision, max_image_size, max_tokens, progress=gr.Progress()):
    """Function to handle the image processing tab"""
    if not files:
        return "", "Please upload some image files."
    
    try:
        # Copy uploaded files to a temporary directory
        temp_dir, _ = copy_to_temp_dir(files)
        
        # Process the directory of images
        output_csv = os.path.join(temp_dir, "name_tags_results.csv")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Process images
        results, summary = process_directory(
            directory_path=temp_dir,
            output_csv=output_csv,
            model_name=model_name,
            prompt=prompt,
            device=device,
            half_precision=half_precision,
            max_image_size=max_image_size,
            max_tokens=max_tokens,
            progress=progress
        )
        
        # Create a DataFrame from results
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'response'} for r in results])
        
        return output_csv, summary
    
    except Exception as e:
        return "", f"Error: {str(e)}\n{traceback.format_exc()}"


def analyze_csv_tab(csv_file):
    """Function to handle the CSV analysis tab"""
    if not csv_file:
        return "Please upload or generate a CSV file first."
    
    # Get the file path from the file object or string
    if isinstance(csv_file, str):
        file_path = csv_file
    else:
        file_path = csv_file.name
    
    # Load data from CSV
    df, message = load_data(file_path)
    if df is None:
        return message
    
    # Generate overview
    overview = f"""=== DATA OVERVIEW ===
Total records: {len(df)}
Unique towns: {df['town'].nunique()}
Unique affiliations: {df['affiliation'].nunique()}
    """
    
    return overview


def search_csv(csv_file, search_term, search_field):
    """Function to search the CSV data"""
    if not csv_file:
        return "Please upload or generate a CSV file first."
    
    if not search_term:
        return "Please enter a search term."
    
    # Get the file path from the file object or string
    if isinstance(csv_file, str):
        file_path = csv_file
    else:
        file_path = csv_file.name
    
    # Load data from CSV
    df, message = load_data(file_path)
    if df is None:
        return message
    
    # Search the data
    result = search_data(df, search_term, search_field)
    return result


def summary_csv(csv_file, summary_type):
    """Function to generate summaries from the CSV data"""
    if not csv_file:
        return "Please upload or generate a CSV file first."
    
    # Get the file path from the file object or string
    if isinstance(csv_file, str):
        file_path = csv_file
    else:
        file_path = csv_file.name
    
    # Load data from CSV
    df, message = load_data(file_path)
    if df is None:
        return message
    
    # Generate appropriate summary
    if summary_type == "By Town":
        result = summary_by_town(df)
    elif summary_type == "By Affiliation":
        result = summary_by_affiliation(df)
    else:
        result = "Please select a summary type."
    
    return result


# Create the Gradio interface
with gr.Blocks(title="People Tag Analyzer") as app:
    gr.Markdown("# People Tag Analyzer")
    gr.Markdown("This app processes images of name tags to extract information and provides analysis tools.")
    
    # Store CSV file path between tabs
    csv_file_path = gr.State("")
    
    with gr.Tabs():
        # Image Processing Tab
        with gr.Tab("Process Images"):
            gr.Markdown("### Step 1: Upload Images")
            with gr.Row():
                image_files = gr.File(file_count="multiple", label="Upload Name Tag Images")
            
            gr.Markdown("### Step 2: Configure Model")
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        choices=["qwen2-vl-2b", "qwen2.5-vl-3b", "qwen2.5-vl-7b"],
                        value="qwen2.5-vl-3b",
                        label="Vision Model"
                    )
                    device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="Device"
                    )
                
                with gr.Column():
                    half_precision = gr.Checkbox(
                        value=True,
                        label="Use Half Precision (FP16)"
                    )
                    max_image_size = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=768,
                        step=64,
                        label="Max Image Size"
                    )
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Max Output Tokens"
                    )
            
            gr.Markdown("### Step 3: Set Prompt")
            prompt = gr.Textbox(
                value="Extract 'name of the person', 'affiliation of the attendee' and also extract the town name you have to get it from the affiliation, then return the results in the format 'Name: Affiliation: Town:'",
                label="Prompt",
                lines=3
            )
            
            gr.Markdown("### Step 4: Process Images")
            process_button = gr.Button("Process Images")
            unload_button = gr.Button("Unload Model (Free Memory)")
            
            with gr.Row():
                output_csv = gr.Textbox(label="Output CSV Path")
                processing_output = gr.Textbox(label="Processing Status", lines=10)
            
            # Connect the process button
            process_button.click(
                fn=process_images_tab,
                inputs=[image_files, model_name, prompt, device, half_precision, max_image_size, max_tokens],
                outputs=[output_csv, processing_output],
                api_name="process_images"
            )
            
            # Connect the unload button
            unload_button.click(
                fn=unload_model,
                inputs=[],
                outputs=[processing_output]
            )
            
            # Update state when CSV is generated
            output_csv.change(
                fn=lambda x: x,
                inputs=[output_csv],
                outputs=[csv_file_path]
            )
        
        # Data Analysis Tab
        with gr.Tab("Analyze Data"):
            gr.Markdown("### Data Input")
            with gr.Row():
                csv_input = gr.File(label="Upload CSV File")
                use_processed = gr.Button("Use Processed CSV")
            
            csv_status = gr.Textbox(label="CSV Status", lines=5)
            
            # Analyze data when CSV is uploaded or selected
            csv_input.change(
                fn=analyze_csv_tab,
                inputs=[csv_input],
                outputs=[csv_status]
            )
            
            # Use processed CSV from first tab
            use_processed.click(
                fn=lambda x: x,
                inputs=[csv_file_path],
                outputs=[csv_input]
            ).then(
                fn=analyze_csv_tab,
                inputs=[csv_file_path],
                outputs=[csv_status]
            )
            
            gr.Markdown("### Summary")
            with gr.Row():
                summary_type = gr.Radio(
                    choices=["By Town", "By Affiliation"],
                    value="By Town",
                    label="Summary Type"
                )
                summary_button = gr.Button("Generate Summary")
            
            summary_output = gr.Textbox(label="Summary Results", lines=20)
            
            # Generate summary when button is clicked
            summary_button.click(
                fn=summary_csv,
                inputs=[csv_input, summary_type],
                outputs=[summary_output]
            )
            
            gr.Markdown("### Search")
            with gr.Row():
                with gr.Column():
                    search_term = gr.Textbox(label="Search Term")
                    search_field = gr.Dropdown(
                        choices=["All Fields", "name", "affiliation", "town"],
                        value="All Fields",
                        label="Search Field"
                    )
                    search_button = gr.Button("Search")
                
                search_output = gr.Textbox(label="Search Results", lines=15)
            
            # Search when button is clicked
            search_button.click(
                fn=search_csv,
                inputs=[csv_input, search_term, search_field],
                outputs=[search_output]
            )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
