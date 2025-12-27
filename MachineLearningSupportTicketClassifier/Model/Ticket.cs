namespace MachineLearningSupportTicketClassifier.Model;

public class SupportTicket
{
    public string TicketText { get; set; } = null!;
    public string Category { get; set; } = null!;
}

public class TicketPrediction
{
    public string PredictedCategory { get; set; } = null!;
}
