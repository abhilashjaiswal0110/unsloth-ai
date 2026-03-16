# Examples & Prompts

This document provides real-world examples and ready-to-use prompts for common Unsloth workflows.

## Table of Contents

- [Quick Examples](#quick-examples)
- [Customer Support Bot](#customer-support-bot)
- [SQL Query Generator](#sql-query-generator)
- [Document Summarizer](#document-summarizer)
- [Translation Model](#translation-model)
- [Sentiment Analyzer](#sentiment-analyzer)
- [Creative Writing Assistant](#creative-writing-assistant)
- [RAG-Enhanced Q&A](#rag-enhanced-qa)
- [Multi-Task Model](#multi-task-model)
- [Prompt Templates](#prompt-templates)

## Quick Examples

### Minimal Fine-Tuning (5 lines)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=2048, args=TrainingArguments(per_device_train_batch_size=2, max_steps=60, output_dir="outputs")).train()
model.save_pretrained_gguf("my-model", tokenizer, quantization_method="q4_k_m")
```

---

## Customer Support Bot

### Use Case
Train a model to handle customer support inquiries with consistent, professional responses.

### Dataset

```python
from datasets import Dataset

support_data = [
    {
        "instruction": "A customer asks about return policy",
        "input": "Hi, I bought a laptop last week and I'd like to return it. What's the process?",
        "output": """Thank you for reaching out! I'd be happy to help you with the return process.

Our return policy allows returns within 30 days of purchase for a full refund, provided:
1. The item is in its original packaging
2. All accessories are included
3. The item shows no signs of damage

To initiate your return:
1. Log into your account at our website
2. Go to Order History → Select the order
3. Click "Request Return" and follow the prompts
4. Print the prepaid shipping label

You'll receive your refund within 5-7 business days after we receive the item.

Is there anything else I can help you with?"""
    },
    {
        "instruction": "A customer reports a billing issue",
        "input": "I was charged twice for my subscription this month!",
        "output": """I sincerely apologize for the inconvenience! Let me help resolve this right away.

I can see the duplicate charge on your account. Here's what I'll do:
1. I'll process a refund for the duplicate charge immediately
2. The refund will appear on your statement within 3-5 business days

To prevent this from happening again, I've flagged your account for our billing team to review.

For your records:
- Original charge: Will remain active for your subscription
- Duplicate charge: Refund initiated

Is there anything else I can assist you with?"""
    },
]

dataset = Dataset.from_list(support_data)
```

### Training

```python
support_prompt = """You are a professional customer support agent. Respond helpfully and empathetically.

### Customer Inquiry:
{instruction}

### Customer Message:
{input}

### Agent Response:
{output}"""

def format_support(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        texts.append(support_prompt.format(
            instruction=examples["instruction"][i],
            input=examples["input"][i],
            output=examples["output"][i],
        ) + tokenizer.eos_token)
    return {"text": texts}

dataset = dataset.map(format_support, batched=True)
```

### Test Prompts

```
You are a professional customer support agent. Respond helpfully and empathetically.

### Customer Inquiry:
A customer asks about shipping status

### Customer Message:
Where is my order? I placed it 5 days ago and haven't received any updates.

### Agent Response:
```

---

## SQL Query Generator

### Use Case
Train a model to convert natural language questions into SQL queries.

### Dataset

```python
sql_data = [
    {
        "instruction": "Convert the following question to a SQL query",
        "input": "How many customers signed up last month?",
        "output": "SELECT COUNT(*) FROM customers WHERE signup_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND signup_date < DATE_TRUNC('month', CURRENT_DATE);"
    },
    {
        "instruction": "Convert the following question to a SQL query",
        "input": "What are the top 5 products by revenue?",
        "output": "SELECT p.product_name, SUM(o.quantity * o.unit_price) AS total_revenue FROM products p JOIN order_items o ON p.id = o.product_id GROUP BY p.product_name ORDER BY total_revenue DESC LIMIT 5;"
    },
    {
        "instruction": "Convert the following question to a SQL query",
        "input": "Show me all employees in the engineering department who were hired this year",
        "output": "SELECT e.first_name, e.last_name, e.hire_date, e.position FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Engineering' AND e.hire_date >= DATE_TRUNC('year', CURRENT_DATE) ORDER BY e.hire_date DESC;"
    },
]
```

### Test Prompts

```
Convert the following question to a SQL query:

Database schema:
- users (id, name, email, created_at, plan_type)
- orders (id, user_id, total, status, created_at)
- products (id, name, price, category)

Question: What is the average order value for premium users in the last 90 days?

SQL:
```

---

## Document Summarizer

### Use Case
Fine-tune a model to produce concise summaries of long documents.

### Training

```python
from datasets import load_dataset

# Use CNN/DailyMail for summarization
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")

summary_prompt = """Summarize the following article in 3-5 bullet points.

### Article:
{article}

### Summary:
{highlights}"""

def format_summary(examples):
    texts = []
    for article, highlights in zip(examples["article"], examples["highlights"]):
        texts.append(summary_prompt.format(
            article=article[:2000],  # Truncate long articles
            highlights=highlights,
        ) + tokenizer.eos_token)
    return {"text": texts}

dataset = dataset.map(format_summary, batched=True)
```

### Test Prompts

```
Summarize the following article in 3-5 bullet points.

### Article:
[Paste your article text here]

### Summary:
```

---

## Translation Model

### Use Case
Fine-tune for domain-specific translation tasks.

```python
translation_data = [
    {
        "instruction": "Translate the following English text to Spanish",
        "input": "The quarterly earnings report shows a 15% increase in revenue.",
        "output": "El informe de ganancias trimestrales muestra un aumento del 15% en los ingresos."
    },
]
```

### Test Prompts

```
Translate the following English text to French.

English: Our machine learning pipeline processes over 10 million requests per day with 99.9% uptime.

French:
```

---

## Sentiment Analyzer

### Use Case
Train a model for structured sentiment analysis.

```python
sentiment_data = [
    {
        "instruction": "Analyze the sentiment of the following review",
        "input": "The product arrived on time and works perfectly. Great value for the price!",
        "output": '{"sentiment": "positive", "confidence": 0.95, "aspects": {"delivery": "positive", "quality": "positive", "value": "positive"}}'
    },
    {
        "instruction": "Analyze the sentiment of the following review",
        "input": "Decent product but shipping took forever and the packaging was damaged.",
        "output": '{"sentiment": "mixed", "confidence": 0.80, "aspects": {"product": "neutral", "shipping": "negative", "packaging": "negative"}}'
    },
]
```

### Test Prompts

```
Analyze the sentiment of the following review. Return a JSON object with sentiment, confidence, and aspect-level analysis.

Review: "I love the design of this phone, but the battery life is disappointing. The camera is amazing though!"

Analysis:
```

---

## Creative Writing Assistant

### Use Case
Fine-tune a model for creative writing with specific styles.

### Test Prompts

```
Write a short story opening in the style of science fiction.

Setting: A space station orbiting a dying star
Character: A maintenance engineer who discovers something unusual
Mood: Mysterious and contemplative

Story:
```

```
Write a product description for an e-commerce website.

Product: Wireless noise-canceling headphones
Target audience: Remote workers
Key features: 40-hour battery, ANC, comfortable fit, microphone
Tone: Professional but approachable

Description:
```

---

## RAG-Enhanced Q&A

### Use Case
Train a model to answer questions using provided context passages.

```python
rag_prompt = """Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""
```

### Test Prompts

```
Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

### Context:
Unsloth is an open-source library that accelerates LLM fine-tuning by 2-5x with 70% less memory usage. It supports popular models like Llama, Qwen, Gemma, and Mistral. The library uses custom Triton kernels for GPU optimization and supports 4-bit QLoRA training. Unsloth is licensed under Apache 2.0 for the core library.

### Question:
What license does Unsloth use?

### Answer:
```

---

## Multi-Task Model

### Use Case
Train a single model that handles multiple task types.

```python
multi_task_data = [
    {"task": "summarize", "input": "Long article text...", "output": "Summary here..."},
    {"task": "translate", "input": "Hello world", "output": "Hola mundo"},
    {"task": "classify", "input": "Great product!", "output": "positive"},
    {"task": "generate", "input": "Write a poem about", "output": "Poem text..."},
]

multi_task_prompt = """Perform the following task.

### Task: {task}
### Input: {input}
### Output: {output}"""
```

---

## Prompt Templates

### Instruction Template (Alpaca)

```
### Instruction:
{Your instruction here}

### Input:
{Optional input/context}

### Response:
```

### Chat Template (Llama 3)

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{Your message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Chat Template (Qwen 2.5)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{Your message}<|im_end|>
<|im_start|>assistant
```

### Structured Output Template

```
### Task:
{task_description}

### Input:
{input_data}

### Output Format:
Return a valid JSON object with the following fields:
- field1: description
- field2: description

### Output:
```

### Chain-of-Thought Template

```
### Problem:
{problem_statement}

### Instructions:
Think through this step by step. Show your reasoning before giving the final answer.

### Solution:
Let me think through this step by step:

Step 1:
Step 2:
...

Therefore, the answer is:
```

## Next Steps

- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — Detailed training configurations
- [Data Preparation](DATA_PREPARATION.md) — Prepare your custom datasets
- [Model Export Guide](MODEL_EXPORT_GUIDE.md) — Deploy your trained models
