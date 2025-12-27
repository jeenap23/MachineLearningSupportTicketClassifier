using MachineLearningSupportTicketClassifier.Model;
using Microsoft.ML;

var tickets = new List<SupportTicket>
{
    new() { TicketText = "Unable to reset my password", Category = "Account" },
    new() { TicketText = "Invoice amount is incorrect", Category = "Billing" },
    new() { TicketText = "Website shows 500 error", Category = "Technical" },
    new() { TicketText = "Need to update payment method", Category = "Billing" },
    new() { TicketText = "Login not working after update", Category = "Account" }
};

var mlContext = new MLContext(seed: 1);

var dataView = mlContext.Data.LoadFromEnumerable(tickets);

// Build the ML Pipeline
var pipeline =
    mlContext.Transforms.Conversion
        .MapValueToKey("Label", nameof(SupportTicket.Category))
    .Append(mlContext.Transforms.Text
        .FeaturizeText("Features", nameof(SupportTicket.TicketText)))
    .Append(mlContext.MulticlassClassification
        .Trainers.SdcaMaximumEntropy())
    .Append(mlContext.Transforms.Conversion
        .MapKeyToValue(
            outputColumnName: nameof(TicketPrediction.PredictedCategory),
            inputColumnName: "PredictedLabel"));

// Train the Model
var model = pipeline.Fit(dataView);

// Create a Prediction Engine
var predictionEngine =
    mlContext.Model.CreatePredictionEngine<SupportTicket, TicketPrediction>(model);

// Test with Real-Time Input
var newTicket = new SupportTicket
{
    TicketText = "My payment failed but money was deducted"
};

var prediction = predictionEngine.Predict(newTicket);

Console.WriteLine($"Predicted Category: {prediction.PredictedCategory}");

// Save the Model for Reuse
mlContext.Model.Save(model, dataView.Schema, "SupportTicketModel.zip");