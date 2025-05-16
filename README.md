# 📱 Speech Recognition on Edge Devices using Wav2Vec2 and ExecuTorch

## 🎯 Project Overview

This project demonstrates an end-to-end pipeline for building a **speech recognition system** optimized for **mobile and edge deployment**. It uses **Wav2Vec2.0**, a self-supervised transformer-based model for speech representation learning, and includes steps for **fine-tuning**, **optimization**, and **export**.

---

## 🧪 Dataset

- Self recorded wav-files (16kHz) on /Data/Train
- Transcription (transcript.csv)

---

## 🧠 Key Objectives

- Fine-tune a pre-trained **Wav2Vec2ForCTC** model using a custom audio dataset.
- Optimize the model for efficient inference on low-power devices:
  - **Dynamic Quantization (INT8)**
  - **Unstructured Pruning**
  - Exporting to **TorchScript** or **ExecuTorch**

---

## 🏗️ Project Architecture

1. **Data Preprocessing and Model Fine-Tuning**
   - Load and resample audio.
   - Tokenize using a pre-trained tokenizer.
   - Augment data (noise, volume shift, speed).
   - Use `Wav2Vec2ForCTC` from Hugging Face Transformers.
   - Custom PyTorch training loop.
   - Evaluation with `WER` and `CER` metrics.
   - Early exit

3. **Optimization Pipeline**
   - Apply **pruning** to reduce model size.
   - Apply **dynamic quantization** (INT8).
   - Export to TorchScript for mobile compatibility.
   - 

4. **Expot**
   - Convert to **ExecuTorch format** for deployment.

---

## 🧪 Tools & Libraries

- `PyTorch`
- `Transformers (Hugging Face)`
- `torchaudio`
- `datasets`
- `execuTorch`

---

