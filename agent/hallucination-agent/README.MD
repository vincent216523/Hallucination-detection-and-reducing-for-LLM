# README

## Project Overview

This repository is part of the exploration into recursive dynamic generation abilities using hallucination classifiers trained on LLM attributes. This implementation uses a simple RNN classifier to predict hallucination, but can be replaced with more effective classifers.  In addition, an ensemble module is also prepared for future work so you can explore inference through multiple classifiers!

## Repository Structure

- **generate.py**: Main script for generating data or predictions.
- **process_results.py**: Script for processing results.
- **data/**: Contains input datasets and related files.
  - `trivia_qa/`: Example dataset directory.
- **models/**: Contains model definitions and pre-trained models.
  - `trained/`: Directory for storing trained model files.
- **ref/**: Reference materials, including Jupyter notebooks and setup scripts.
- **result/**: Directory for storing generated results and intermediate files.
- **terminal/**: Contains job scripts for running tasks on clusters or schedulers.
- **utils/**: Utility scripts for data handling, model utilities, and classification.

## Setting Up the Environment

1. **Install Python**: Ensure you have Python 3.10 or later installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Create and Install Dependencies**:
   ```bash
   bash setup.sh
   ```

## Running Scripts

### Running `generate.py`

The `generate.py` script is used to generate data or perform specific tasks. Follow these steps to run it:

1. **Activate the Virtual Environment**:
   ```bash
   source activate hallucination_slurm
   ```

2. **Run the Script**:
   Execute the script with Python. Include input arguments of --start and --end of questions. Also include --n to indicate number of turns.

   Flags:
   - --start: starting idx of TriviaQA question
   - --end: ending idx of TriviaQA question
   - --n: number of multi turns for generation
   - --data_mining: flag to capture all attributes for every generate (defaulse = False). VERY DATA INTENSIVE
   - --debug: prints the intermediary outputs and multiturn prompts in terminal (default = False)
   - --sensitivity: sensitivity parameter for what level of sensitivity to used (default = 0.05)

   Example:
   ```bash
   python generate.py --start 0 --end 100 --sensitivity 0.1
   ```

### Modifying modules

The `utils/classifier.py` module can be changed to include your own classifier by simply defining your own classiifer loading script and mapping the classifer foward func under MODELS dict along with a model name.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
