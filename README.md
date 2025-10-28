# 🧠 StockTweetCLIP: Contrastive Learning Between Tweets and Stock Time Series

This project implements a **multimodal contrastive learning framework** that connects **financial social media text (tweets)** with **historical stock price time series**.  
It uses **Sentence-T5** for text encoding and a **PatchTST-style Transformer encoder** for temporal features, trained with a **CLIP-like contrastive objective**.

---

## 🚀 Features

- **Tweet Processing & Encoding**
  - Cleans and groups tweets by stock and date.
  - Uses [`sentence-transformers/sentence-t5-base`](https://huggingface.co/sentence-transformers/sentence-t5-base) to generate dense embeddings.
  - Aggregates multiple tweets per day into averaged embeddings.

- **Stock Time Series Encoding**
  - Loads stock data from Yahoo Finance.
  - Builds fixed-length context (past) and prediction (future) windows.
  - Uses a **PatchTST Transformer encoder** with patch embeddings and positional encodings.

- **Contrastive Learning Framework**
  - Pairs tweet embeddings with aligned stock price windows.
  - Learns a joint representation space via **symmetric InfoNCE (contrastive) loss**.
  - Evaluates performance with **Recall@K** and **MRR** metrics.

---

## 📂 Project Structure

