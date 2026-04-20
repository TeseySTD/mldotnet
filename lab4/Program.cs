using Microsoft.ML;
using GitHubIssueClassification;

string appPath = Path.GetDirectoryName(Environment.ProcessPath) ?? ".";
string dataRoot = Path.GetFullPath(Path.Combine(appPath, "..", "..", "..", "Data"));
string modelRoot = Path.GetFullPath(Path.Combine(appPath, "..", "..", "..", "Models"));

Directory.CreateDirectory(dataRoot);
Directory.CreateDirectory(modelRoot);

string trainDataPath = Path.Combine(dataRoot, "issues_train.tsv");
string testDataPath = Path.Combine(dataRoot, "issues_test.tsv");
string modelPath = Path.Combine(modelRoot, "model.zip");

if (!File.Exists(trainDataPath) || !File.Exists(testDataPath))
{
    Console.WriteLine("Downloading datasets...");
    using var client = new HttpClient();
    var trainData = await client.GetByteArrayAsync(
        "https://raw.githubusercontent.com/dotnet/samples/main/machine-learning/tutorials/GitHubIssueClassification/Data/issues_train.tsv");
    await File.WriteAllBytesAsync(trainDataPath, trainData);

    var testData = await client.GetByteArrayAsync(
        "https://raw.githubusercontent.com/dotnet/samples/main/machine-learning/tutorials/GitHubIssueClassification/Data/issues_test.tsv");
    await File.WriteAllBytesAsync(testDataPath, testData);
    Console.WriteLine("Download complete.\n");
}

MLContext context = new MLContext(seed: 0);
PredictionEngine<GitHubIssue, IssuePrediction> predEngine;
ITransformer trainedModel;

var trainingDataViewGloabal =
    context.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);

var globalPipeline = ProcessData();

BuildAndTrainModel(trainingDataViewGloabal, globalPipeline);

Evaluate(trainingDataViewGloabal.Schema);

PredictIssue();


IEstimator<ITransformer> ProcessData()
{
    return context.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Description",
            outputColumnName: "DescriptionFeaturized"))
        .Append(context.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(context);
}

void BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    trainedModel = trainingPipeline.Fit(trainingDataView);

    predEngine = context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);

    GitHubIssue issue = new GitHubIssue()
    {
        Title = "WebSockets communication is slow in my machine",
        Description =
            "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    var prediction = predEngine.Predict(issue);
    Console.WriteLine(
        $"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
}

void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = context.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
    var testMetrics = context.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

    Console.WriteLine($"*************************************************************************");
    Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data ");
    Console.WriteLine($"*------------------------------------------------------------------------");
    Console.WriteLine($"* MicroAccuracy: {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"* MacroAccuracy: {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"* LogLoss: {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************");

    SaveModelAsFile(context, trainingDataViewSchema, trainedModel);
}

void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
}

void PredictIssue()
{
    ITransformer loadedModel = context.Model.Load(modelPath, out _);

    GitHubIssue singleIssue = new GitHubIssue()
    {
        Title = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing"
    };

    predEngine = context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var prediction = predEngine.Predict(singleIssue);

    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}

namespace GitHubIssueClassification
{
    using Microsoft.ML.Data;

    public class GitHubIssue
    {
        [LoadColumn(0)] public string? ID { get; set; }
        [LoadColumn(1)] public string? Area { get; set; }
        [LoadColumn(2)] public required string Title { get; set; }
        [LoadColumn(3)] public required string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")] public string? Area;
    }
}