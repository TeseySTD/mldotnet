using System.IO.Compression;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

string dataDirPath = Path.GetFullPath(Path.Combine("..", "..", "..", "Data"));
Directory.CreateDirectory(dataDirPath);

string dataPath = Path.Combine(dataDirPath, "yelp_labelled.txt");
string zipPath = Path.Combine(dataDirPath, "sentiment_sentences.zip");

if (!File.Exists(dataPath))
{
    Console.WriteLine("Downloading dataset archive...");
    using var client = new HttpClient();
    
    var bytes = client.GetByteArrayAsync("https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip").Result;
    File.WriteAllBytes(zipPath, bytes);
    Console.WriteLine("Download complete. Extracting...");

    ZipFile.ExtractToDirectory(zipPath, dataDirPath, overwriteFiles: true);

    var extractedFilePath = Path.Combine(dataDirPath, "sentiment labelled sentences", "yelp_labelled.txt");
    
    if (File.Exists(extractedFilePath))
    {
        File.Copy(extractedFilePath, dataPath, overwrite: true);
    }

    Console.WriteLine("Extraction complete.\n");
}
MLContext context = new MLContext();

TrainTestData splitDataView = LoadData(context, dataPath);

ITransformer globalModel = BuildAndTrainModel(context, splitDataView.TrainSet);

Evaluate(context, globalModel, splitDataView.TestSet);

UseModelWithSingleItem(context, globalModel);

UseModelWithBatchItems(context, globalModel);

Console.WriteLine("=============== End of process ===============");

TrainTestData LoadData(MLContext mlContext, string path)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(path, hasHeader: false);

    // Split the dataset into train and test datasets (80% / 20%)
    TrainTestData splitDataViewLocal = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    return splitDataViewLocal;
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text
        .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
            featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);

    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions);

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
        mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    SentimentData sampleStatement = new SentimentData
    {
        SentimentText = "This was a very bad steak"
    };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
    Console.WriteLine();
    Console.WriteLine(
        $"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments = new[]
    {
        new SentimentData { SentimentText = "This was a horrible meal" },
        new SentimentData { SentimentText = "I love this spaghetti." }
    };

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
    IDataView predictions = model.Transform(batchComments);

    IEnumerable<SentimentPrediction> predictedResults =
        mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine(
            $"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }

    Console.WriteLine("=============== End of predictions ===============");
}


public class SentimentData
{
    [LoadColumn(0)] public string? SentimentText;

    [LoadColumn(1), ColumnName("Label")] public bool Sentiment;
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}