# Support Ticket Classifier (ML.NET)

A .NET machine learning project that classifies support tickets into categories using **ML.NET** and **C#**.  
This project demonstrates a real-world multiclass text classification use case in .NET.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

Support teams handle large volumes of incoming tickets. Automating categorization improves response times and routing efficiency.

## Features
- Train custom ML model with historical support ticket data
- Multiclass classification using SDCA in ML.NET
- Real-time prediction from console application
- Save and reuse trained model
- Easy to extend to web APIs or Azure Functions

## Architecture
1. Load training data
2. Build text-feature pipeline
3. Train the model
4. Persist model to disk
5. Load model for real-time prediction

## Getting Started

### Prerequisites
- .NET 7.0 SDK or later
- Visual Studio / VS Code

```bash
dotnet restore
dotnet build
