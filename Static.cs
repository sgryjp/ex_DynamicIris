using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace DynamicIris
{
    static class Static
    {
        public class IrisRecord
        {
            [LoadColumn(0)]
            public float SepalLength { get; set; }
            [LoadColumn(1)]
            public float SepalWidth { get; set; }
            [LoadColumn(2)]
            public float PetalLength { get; set; }
            [LoadColumn(3)]
            public float PetalWidth { get; set; }
        }

        public static IDataView LoadIrisStatic(MLContext ctx, string filePath)
        {
            if (ctx is null)
            {
                throw new ArgumentNullException(nameof(ctx));
            }

            return ctx.Data.LoadFromTextFile<IrisRecord>(filePath, separatorChar: ',');
        }

        public static void DoClustering(MLContext ctx, IDataView data)
        {
            if (ctx is null)
            {
                throw new ArgumentNullException(nameof(ctx));
            }
            if (data is null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            var columnNames = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" }; // Preknown information
            var featuresColumnName = "Features";
            var trainer = ctx.Clustering.Trainers.KMeans(
                featuresColumnName,
                numberOfClusters: 3); // Preknown information
            var pipeline = ctx.Transforms
                .Concatenate(featuresColumnName, columnNames)
                .Append(trainer);
            var model = pipeline.Fit(data);

            var predictor = ctx.Model.CreatePredictionEngine<IrisRecord, ClusterPrediction>(model);
            var newSample = new IrisRecord()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };
            Console.WriteLine("[Prediction]");
            var prediction = predictor.Predict(newSample);
            Console.WriteLine("    Cluster: {0}", prediction.PredictedClusterId);
            Console.WriteLine("    Distances: {0}", string.Join(" ", prediction.Distances));
            Console.WriteLine();
        }
    }
}
