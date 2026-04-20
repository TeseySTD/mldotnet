using System.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

string dataDirPath = Path.GetFullPath(Path.Combine("..", "..", "..", "Data"));
Directory.CreateDirectory(dataDirPath);
string dbFilePath = Path.Combine(dataDirPath, "DailyDemand.mdf");
string modelPath = Path.GetFullPath(Path.Combine("..", "..", "..",  "MLModel.zip"));

if (!File.Exists(dbFilePath))
{
    Console.WriteLine("Downloading database file...");
    using var client = new HttpClient();
    var bytes = client.GetByteArrayAsync("https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/samples/csharp/getting-started/Forecasting_BikeSharingDemand/BikeDemandForecasting/Data/DailyDemand.mdf").Result;
    File.WriteAllBytes(dbFilePath, bytes);
    Console.WriteLine("Download complete.\n");
}

string connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";

MLContext mlContext = new MLContext();

Console.WriteLine("Loading data from database...");
DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();
string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";
DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, query);
IDataView dataView = loader.Load(dbSource);

IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

Console.WriteLine("Training the model (SSA)...");
var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");

SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

// Evaulation
Console.WriteLine("Evaluating the model...");
Evaluate(secondYearData, forecaster, mlContext);

// Saving model
Console.WriteLine("Saving the model...");
var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
forecastEngine.CheckPoint(mlContext, modelPath);

Console.WriteLine("Forecasting demand for the next 7 days...");
Forecast(secondYearData, 7, forecastEngine, mlContext);

Console.WriteLine("Execution completed.");

void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{
    IDataView predictions = model.Transform(testData);

    IEnumerable<float> actual = mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
        .Select(observed => observed.TotalRentals);

    IEnumerable<float> forecast = mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
        .Select(prediction => prediction.ForecastedRentals[0]);

    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    var MAE = metrics.Average(error => Math.Abs(error)); 
    var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); 

    Console.WriteLine("\nEvaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
    Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
}

void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
{
    ModelOutput forecast = forecaster.Predict();

    IEnumerable<string> forecastOutput = mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
        .Take(horizon)
        .Select((ModelInput rental, int index) =>
        {
            string rentalDate = rental.RentalDate.ToShortDateString();
            float actualRentals = rental.TotalRentals;
            float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
            float estimate = forecast.ForecastedRentals[index];
            float upperEstimate = forecast.UpperBoundRentals[index];
            
            return $"Date: {rentalDate}\n" +
                   $"Actual Rentals: {actualRentals}\n" +
                   $"Lower Estimate: {lowerEstimate:F3}\n" +
                   $"Forecast: {estimate:F3}\n" +
                   $"Upper Estimate: {upperEstimate:F3}\n";
        });

    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");
    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}

public class ModelInput
{
    public DateTime RentalDate { get; set; }
    public float Year { get; set; }
    public float TotalRentals { get; set; }
}

public class ModelOutput
{
    public float[] ForecastedRentals { get; set; } = Array.Empty<float>();
    public float[] LowerBoundRentals { get; set; } = Array.Empty<float>();
    public float[] UpperBoundRentals { get; set; } = Array.Empty<float>();
}
