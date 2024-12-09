# Llama-3.2 1B 4-bit Quantized Model

## Model Overview
- **Base Model**: Meta-Llama/Llama-3.2-1B
- **Model Name**: rautaditya/llama-3.2-1b-4bit-gptq
- **Quantization**: 4-bit GPTQ (Generative Pretrained Transformer Quantization)

## Model Description
This is a 4-bit quantized version of the Llama-3.2 1B model, designed to reduce model size and inference latency while maintaining reasonable performance. The quantization process allows for more efficient deployment on resource-constrained environments.

### Key Features
- Reduced model size
- Faster inference times
- Compatible with Hugging Face Transformers
- GPTQ quantization for optimal compression

## Quantization Details
- **Quantization Method**: GPTQ (Generative Pretrained Transformer Quantization)
- **Bit Depth**: 4-bit
- **Base Model**: Llama-3.2 1B
- **Quantization Library**: AutoGPTQ

## Installation Requirements
```bash
pip install transformers accelerate auto-gptq torch
```

## Usage

### Transformers Pipeline
```python
from transformers import AutoTokenizer, pipeline

ModelFolder = "rautaditya/llama-3.2-1b-4bit-gptq"
tokenizer = AutoTokenizer.from_pretrained(ModelFolder)
pipe = pipeline(
    "text-generation",
    model=ModelFolder,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = "What is the meaning of life?"
generated_text = pipe(prompt, max_length=100)
print(generated_text)
```

### Direct Model Loading
```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model_name = "rautaditya/llama-3.2-1b-4bit-gptq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoGPTQForCausalLM.from_pretrained(
    model_name, 
    device_map="auto"
)
```

## Performance Considerations
- **Memory Efficiency**: Significantly reduced memory footprint compared to full-precision model
- **Inference Speed**: Faster inference due to reduced computational requirements
- **Potential Accuracy Trade-off**: Minor performance degradation compared to full-precision model

## Limitations
- May show slight differences in output quality compared to the original model
- Performance can vary based on specific use case and inference environment

## Recommended Use Cases
- Low-resource environments
- Edge computing
- Mobile applications
- Embedded systems
- Rapid prototyping

## License
Please refer to the original Meta Llama 3.2 model license for usage restrictions and permissions.

## Citation
If you use this model, please cite:
```
@misc{llama3.2_4bit_quantized,
  title={Llama-3.2 1B 4-bit Quantized Model},
  author={Raut, Aditya},
  year={2024},
  publisher={Hugging Face}
}
```

## Contributions and Feedback
- Open to suggestions and improvements
- Please file issues on the GitHub repository for any bugs or performance concerns

## Acknowledgments
- Meta AI for the base Llama-3.2 model
- Hugging Face Transformers team
- AutoGPTQ library contributors
