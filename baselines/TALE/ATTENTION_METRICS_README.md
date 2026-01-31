# Attention Metrics Implementation

This implementation adds kurtosis and max_inf_norm calculation capabilities to the inference script, based on the OutEffHop reference implementation.

## Features Added

### 1. Metrics Calculation Functions

- **`calculate_kurtosis(activations)`**: Calculates the average kurtosis of model activations
- **`calculate_max_inf_norm(activations)`**: Calculates the maximum infinity norm of activations
- **`calculate_attention_metrics(attention_weights_list)`**: Computes both metrics from a list of attention weight tensors

### 2. Integration with LLMModel

- Added `attention_weights` storage in `HuggingFaceModel`
- Added `get_attention_weights()` and `clear_attention_weights()` methods
- Extended `LLMModel` class with attention weight access methods

### 3. Enhanced Inference Functions

- **`inference_local()`**: Now calculates and logs attention metrics for local models
- **`inference_api()`**: Includes metrics structure (set to 0.0 for API models)
- Results now include a metrics section with kurtosis, max_inf_norm, accuracy, and token costs

## Usage

### Running with Local Models

```bash
python inference2.py --model DeepSeek-R1-Distill-Qwen-1.5B --data_name GSM8K-Zero --reasoning --start_index 0 --end_index 10
```

### Running with API Models

```bash
python inference2.py --model gpt-4o-mini --data_name GSM8K-Zero --reasoning --start_index 0 --end_index 10
```

## Output

The script now logs:
- Average Kurtosis: Measures distribution "tailedness" of attention weights
- Max Infinity Norm: Maximum absolute value in attention weights
- Standard accuracy and token cost metrics

Results are saved to JSONL with an additional metrics entry containing all calculated values.

## Technical Details

### Kurtosis Calculation
- Uses Fisher's definition (normal distribution = 0 kurtosis)
- Calculates kurtosis for each feature dimension
- Returns average across all dimensions

### Max Infinity Norm
- Calculates infinity norm for each sample
- Returns maximum across all samples
- Handles multi-dimensional tensors appropriately

### Attention Weight Access
- Currently implemented for HuggingFace models using vLLM
- API models return empty lists (metrics set to 0.0)
- Weights are cleared between inference runs

## Dependencies

- `torch`: For tensor operations
- `numpy`: For numerical computations
- `scipy.stats`: For kurtosis calculation
- Existing project dependencies (vLLM, transformers, etc.)

## Notes

- Attention weights are only available for local models
- The implementation follows the OutEffHop reference for metric calculation
- Metrics are calculated after batch processing for efficiency
- Results include both individual sample data and aggregate metrics
