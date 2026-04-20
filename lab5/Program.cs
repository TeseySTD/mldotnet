using Microsoft.ML;
using Microsoft.ML.Data;

string appPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
string dataPath = Path.Combine(appPath, "Data", "product-sales.csv");
const int docsize = 36;

//  DOWNLOAD DATA 
if (!Directory.Exists(Path.Combine(appPath, "Data")))
{
    Directory.CreateDirectory(Path.Combine(appPath, "Data"));
}

if (!File.Exists(dataPath))
{
    Console.WriteLine("Downloading dataset...");
    using var client = new HttpClient();
    var content = await client.GetStringAsync(
        "https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/samples/csharp/getting-started/AnomalyDetection_Sales/SpikeDetection/Data/product-sales.csv");
    await File.WriteAllTextAsync(dataPath, content);
    Console.WriteLine("Download complete.\n");
}

//  INITIALIZE CONTEXT 
MLContext context = new MLContext();

IDataView dataView =
    context.Data.LoadFromTextFile<ProductSalesData>(path: dataPath, hasHeader: true, separatorChar: ',');

//  DETECT SPIKES 
Console.WriteLine("=============== Detecting Spikes ===============");
DetectSpike(context, docsize, dataView);

//  DETECT CHANGE POINTS 
Console.WriteLine("=============== Detecting Change Points ===============");
DetectChangepoint(context, docsize, dataView);

Console.WriteLine("\nExecution completed. Press any key to exit.");

void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95d,
        pvalueHistoryLength: docSize / 4);

    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
    IDataView transformedData = iidSpikeTransform.Transform(productSales);

    var predictions =
        mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value");
    foreach (var p in predictions)
    {
        if (p.Prediction != null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";
            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }

            Console.WriteLine(results);
        }
    }

    Console.WriteLine("");
}

void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95d,
        changeHistoryLength: docSize / 4);

    ITransformer iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
    IDataView transformedData = iidChangePointTransform.Transform(productSales);

    var predictions =
        mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
    foreach (var p in predictions)
    {
        if (p.Prediction != null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";
            if (p.Prediction[0] == 1)
            {
                results += " <-- alert is on, predicted changepoint";
            }

            Console.WriteLine(results);
        }
    }
}

IDataView CreateEmptyDataView(MLContext mlContext)
{
    return mlContext.Data.LoadFromEnumerable(new List<ProductSalesData>());
}

// --- DATA CLASSES ---
public class ProductSalesData
{
    [LoadColumn(0)] public string? Month;
    [LoadColumn(1)] public float numSales;
}

public class ProductSalesPrediction
{
    [VectorType(3)] // For Spikes
    public double[]? Prediction { get; set; }
}