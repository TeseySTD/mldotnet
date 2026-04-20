using System.IO.Compression;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommendation;

string dataDirPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Data"));
Directory.CreateDirectory(dataDirPath);

string zipPath = Path.Combine(dataDirPath, "ml-latest-small.zip");
string extractedDir = Path.Combine(dataDirPath, "ml-latest-small");
string ratingsPath = Path.Combine(extractedDir, "ratings.csv");

//  DOWNLOAD & PREPARE DATASOURCE 
if (!File.Exists(ratingsPath))
{
    Console.WriteLine("Downloading MovieLens dataset...");
    using var client = new HttpClient();
    var bytes = client.GetByteArrayAsync("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip").Result;
    File.WriteAllBytes(zipPath, bytes);

    Console.WriteLine("Extracting data...");
    ZipFile.ExtractToDirectory(zipPath, dataDirPath, true);
    string folderInZip = Directory.GetDirectories(dataDirPath, "ml-latest-small*").FirstOrDefault();
    if (folderInZip != null) ratingsPath = Path.Combine(folderInZip, "ratings.csv");
}

MLContext mlContext = new MLContext();

//  LOAD & SPLIT DATA
Console.WriteLine("Loading and splitting data...");
IDataView fullData = mlContext.Data.LoadFromTextFile<MovieRating>(ratingsPath, hasHeader: true, separatorChar: ',');

var trainTestSplit = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
IDataView trainingDataView = trainTestSplit.TrainSet;
IDataView testDataView = trainTestSplit.TestSet;

//  BUILD AND TRAIN

Console.WriteLine("=============== Training the model ===============");
var estimator = mlContext.Transforms.Conversion
    .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded",
        inputColumnName: "movieId"));

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "userIdEncoded",
    MatrixRowIndexColumnName = "movieIdEncoded",
    LabelColumnName = "Label",
    NumberOfIterations = 20,
    ApproximationRank = 100
};

var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
ITransformer model = trainerEstimator.Fit(trainingDataView);

//  EVALUATE
Console.WriteLine("=============== Evaluating the model ===============");
var prediction = model.Transform(testDataView);
var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

Console.WriteLine($"Root Mean Squared Error : {metrics.RootMeanSquaredError:F4}");
Console.WriteLine($"RSquared: {metrics.RSquared:F4}");

//  PREDICT
Console.WriteLine("=============== Making a prediction ===============");
var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

var testInput = new MovieRating { userId = 1, movieId = 10 };
var movieRatingPrediction = predictionEngine.Predict(testInput);

Console.WriteLine(movieRatingPrediction.Score > 3.5
    ? $"Movie {testInput.movieId} is recommended for user {testInput.userId} (Predicted Score: {movieRatingPrediction.Score:F2})"
    : $"Movie {testInput.movieId} is NOT recommended for user {testInput.userId} (Predicted Score: {movieRatingPrediction.Score:F2})");

Console.WriteLine("\nExecution completed.");

namespace MovieRecommendation
{
    using Microsoft.ML.Data;

    public class MovieRating
    {
        [LoadColumn(0)] public float userId;
        [LoadColumn(1)] public float movieId;
        [LoadColumn(2)] public float Label; // рейтинг
    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
}