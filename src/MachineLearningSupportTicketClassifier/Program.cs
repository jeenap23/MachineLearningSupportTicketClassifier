using MachineLearningSupportTicketClassifier.ML;
using MachineLearningSupportTicketClassifier.Model;
using Microsoft.ML;

namespace MachineLearningSupportTicketClassifier;

internal class Program
{
    static void Main()
    {
        // Initialize ML.NET context (fixed seed for reproducibility)
        var mlContext = new MLContext(seed: 1);

        // Training data (could come from DB or CSV)
        var tickets = new List<SupportTicket>
        {
            new() { TicketText = "Unable to reset my password", Category = "Account" },
            new() { TicketText = "Invoice amount is incorrect", Category = "Billing" },
            new() { TicketText = "Website shows 500 error", Category = "Technical" },
            new() { TicketText = "Need to update payment method", Category = "Billing" },
            new() { TicketText = "Login not working after update", Category = "Account" }
        };

        // Train the ML model
        var model = ModelTrainer.TrainModel(
            mlContext,
            tickets,
            out var modelSchema);

        // Save trained model for reuse
        Directory.CreateDirectory("Models");
        var modelPath = Path.Combine("Models", "SupportTicketModel.zip");

        ModelTrainer.SaveModel(
            mlContext,
            model,
            modelSchema,
            modelPath);

        Console.WriteLine("Model trained and saved successfully.");

        // Create prediction engine for real-time inference
        var predictionEngine =
            mlContext.Model.CreatePredictionEngine<SupportTicket, TicketPrediction>(model);

        // Test with a new incoming support ticket
        var newTicket = new SupportTicket
        {
            TicketText = "My payment failed but money was deducted"
        };

        var prediction = predictionEngine.Predict(newTicket);

        Console.WriteLine($"Predicted Category: {prediction.PredictedCategory}");
    }
}
