# Training Log Comparison Report

## Overview

- **Previous Run**: `logs/training_log_prev.txt`
- **New Run (Resumed)**: `logs/training_log.txt`
- **Comparison Date**: 2025-10-23 09:24:19

## ✅ Resume Information

- **Resumed from Epoch**: 3
- **Resume Timestamp**: 2025-10-23 08:37:24

## Configuration Comparison

| Parameter | Previous Run | New Run | Match |
|-----------|--------------|---------|-------|
| Device | CUDA:0 - NVIDIA GeForce RTX 3060 Laptop GPU | CUDA:0 - NVIDIA GeForce RTX 3060 Laptop GPU | ✅ |
| Classes | ['airplane', 'coin'] | ['airplane', 'coin'] | ✅ |
| Dataset Samples | Pool=123350, Planned=27000 | Pool=123350, Planned=27000 | ✅ |


## Epoch-by-Epoch Metrics

| Epoch | Prev Loss | New Loss | Prev IoU | New IoU | Prev LR | New LR |
|-------|-----------|----------|----------|---------|---------|--------|
| 1 | 1.0480 | N/A | 0.4991 | N/A | 0.000100 | N/A |
| 2 | 1.0319 | N/A | 0.4991 | N/A | 0.000100 | N/A |
| 3 | 1.0229 | N/A | 0.4991 | N/A | 0.000010 | N/A |
| 4 | 1.0186 | 1.0179 | 0.4991 | 0.4991 | 0.000010 | 0.000010 |
| 5 | 1.0179 | N/A | 0.4991 | N/A | 0.000010 | N/A |
| 6 | 1.0172 | 1.0172 | 0.4991 | 0.4991 | 0.000001 | 0.000001 |
| 7 | 1.0167 | 1.0167 | 0.4991 | 0.4991 | 0.000001 | 0.000001 |
| 8 | 1.0167 | 1.0167 | 0.4991 | 0.4991 | 0.000001 | 0.000001 |
| 9 | 1.0166 | 1.0166 | 0.4991 | 0.4991 | 0.000000 | 0.000000 |
| 10 | 1.0165 | 1.0165 | 0.4991 | 0.4991 | 0.000000 | 0.000000 |


## ⚠️ Detected Differences

❌ **Epoch mismatch**: Previous run had epochs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], new run has [4, 6, 7, 8, 9, 10]

ℹ️  **Epoch 4 Duration**: Prev=0:02:54, New=0:03:08

ℹ️  **Epoch 6 Duration**: Prev=0:02:55, New=0:03:06

ℹ️  **Epoch 7 Duration**: Prev=0:02:55, New=0:03:07

ℹ️  **Epoch 8 Duration**: Prev=0:02:54, New=0:03:07

ℹ️  **Epoch 9 Duration**: Prev=0:02:54, New=0:03:07

ℹ️  **Epoch 10 Duration**: Prev=0:02:54, New=0:03:07

## Training Duration

- **Previous Run**: 0:30:05
- **New Run**: 0:23:46
