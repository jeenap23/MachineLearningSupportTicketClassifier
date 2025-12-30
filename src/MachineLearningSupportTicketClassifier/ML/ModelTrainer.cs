using MachineLearningSupportTicketClassifier.Model;
using Microsoft.ML;

namespace MachineLearningSupportTicketClassifier.ML
{
    public static class ModelTrainer
    {
        public static ITransformer TrainModel(
            MLContext mlContext,
            IEnumerable<SupportTicket> tickets,
            out DataViewSchema modelSchema)
        {
            // Convert training data into ML.NET format
            var dataView = mlContext.Data.LoadFromEnumerable(tickets);

            // Build ML pipeline
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

            // Train the model
            var model = pipeline.Fit(dataView);

            // Capture schema for model persistence
            modelSchema = dataView.Schema;

            return model;
        }

        public static void SaveModel(
            MLContext mlContext,
            ITransformer model,
            DataViewSchema schema,
            string modelPath)
        {
            mlContext.Model.Save(model, schema, modelPath);
        }
    }
}
