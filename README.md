# Name Tag Image Analyzer

This application uses Qwen Vision-Language Models to extract information from name tag images and provides analysis tools for the extracted data. It demonstrates the power of open-source vision-language models for automated information extraction and analysis.

## Quick Start

1. Open terminal in VS Code:
   ```
   File > New Launcher > Terminal
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/ictBioRtc/image_analyzer.git
   ```

3. Navigate to the project directory:
   ```bash
   cd image_analyzer
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Gradio will start and provide a local URL (usually http://127.0.0.1:7860)

## Using the Application

### Step 1: Prepare Images
1. Unzip the accompanying `NameTags.zip` file
2. Make sure your images are clear and readable

### Step 2: Process Images
1. Go to the "Process Images" tab
2. Upload all name tag images using the upload button
3. Adjust settings:
   - Set "Max Image Size" to 1024 or higher
   - Set "Max Tokens" to 256 or higher
   - Leave other settings at default
4. Click "Process Images"
5. Watch the progress bar as images are processed
6. Wait for completion message

### Step 3: Analyze Data
1. Switch to the "Analyze Data" tab
2. View statistics:
   - Summary by Town
   - Summary by Affiliation
3. Use search functionality:
   - Search by name, affiliation, or town
   - View detailed results

## Features

### Image Processing
- Multiple image upload support
- Configurable model settings
- Progress tracking
- Memory management
- CSV output generation

### Data Analysis
- Summary statistics by town
- Summary statistics by affiliation
- Search functionality
- Data overview
- CSV file handling

## Model Options
- qwen2-vl-2b
- qwen2.5-vl-3b (default)
- qwen2.5-vl-7b

## Requirements
- Python 3.8+
- Torch
- Transformers
- Gradio
- Pandas
- PIL
- Other dependencies in requirements.txt

## What You'll Learn
This exercise demonstrates:
1. Capabilities of open-source vision-language models
2. Automated information extraction
3. Data analysis and visualization
4. Integration of multiple AI technologies
5. Practical application of ML for automation

## Notes
- Image processing time depends on your hardware and selected model
- GPU is recommended but not required
- Memory usage scales with image size and model choice
- Use "Unload Model" button to free memory when needed

## Customization
You can modify the prompt and parameters to adapt the system for different types of name tags or similar image processing tasks. The analysis tools can be customized for different data structures.

## Limitations
- Image quality affects extraction accuracy
- Processing large images requires more memory
- GPU memory constraints may require batch processing

## Support
For issues or questions:
- Check the GitHub repository issues
- Contact the development team
- Review the code documentation

Congratulations! You've successfully used a small but powerful vision-language model for automated information extraction. This demonstrates how open-source AI models can be effectively used for practical automation tasks.