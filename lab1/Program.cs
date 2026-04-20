using Microsoft.ML;
using Microsoft.ML.Data;

string trainDataPath = Path.Combine("..", "..", "..", "Data", "taxi-fare-train.csv");
string testDataPath = Path.Combine("..", "..", "..", "Data", "taxi-fare-test.csv");

MLContext mlContext = new MLContext(seed: 0);

Console.WriteLine("Starting model training...");
var trainedModel = Train(mlContext, trainDataPath);

Console.WriteLine("Evaluating model quality...");
Evaluate(mlContext, trainedModel, testDataPath);

Console.WriteLine("Testing a single prediction...");
TestSinglePrediction(mlContext, trainedModel);

Console.WriteLine("Execution completed.");

ITransformer Train(MLContext ctx, string dataPath)
{
    IDataView dataView = ctx.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

    var pipeline = ctx.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
        .Append(ctx.Transforms.Categorical.OneHotEncoding("VendorIdEncoded", "VendorId"))
        .Append(ctx.Transforms.Categorical.OneHotEncoding("RateCodeEncoded", "RateCode"))
        .Append(ctx.Transforms.Categorical.OneHotEncoding("PaymentTypeEncoded", "PaymentType"))
        .Append(ctx.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
        .Append(ctx.Regression.Trainers.FastTree());

    return pipeline.Fit(dataView);
}

void Evaluate(MLContext ctx, ITransformer model, string testPath)
{
    IDataView dataView = ctx.Data.LoadFromTextFile<TaxiTrip>(testPath, hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(dataView);
    var metrics = ctx.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine("\n*************************************************");
    Console.WriteLine("* Model quality metrics evaluation ");
    Console.WriteLine("*------------------------------------------------");
    Console.WriteLine($"* RSquared Score: {metrics.RSquared:0.##}");
    Console.WriteLine($"* Root Mean Squared Error: {metrics.RootMeanSquaredError:0.##}");
}

void TestSinglePrediction(MLContext ctx, ITransformer model)
{
    var predictionFunction = ctx.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // Value to predict
    };

    var prediction = predictionFunction.Predict(taxiTripSample);

    Console.WriteLine("\n*******************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine("*******************************************************************\n");
}

public class TaxiTrip
{
    [LoadColumn(0)] public string? VendorId;
    [LoadColumn(1)] public string? RateCode;
    [LoadColumn(2)] public float PassengerCount;
    [LoadColumn(3)] public float TripTime;
    [LoadColumn(4)] public float TripDistance;
    [LoadColumn(5)] public string? PaymentType;
    [LoadColumn(6)] public float FareAmount;
}

public class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount;
}
