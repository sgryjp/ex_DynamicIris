using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace DynamicIris
{
    class ClusterPrediction
    {
        public uint PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var filePath = "../../iris.csv";
            IDataView data;
            var ctx = new MLContext();

            Console.WriteLine("------ Statically typed example ------");
            data = Static.LoadIrisStatic(ctx, filePath);
            Console.WriteLine("[Loaded Data]");
            foreach (var column in data.Schema)
            {
                Console.WriteLine($"    {column.Name} <{column.Type} ({column.Type.RawType})>");
            }
            Static.DoClustering(ctx, data);

            Console.WriteLine("------ Dynamically typed example ------");
            data = Dynamic.LoadIris(ctx, filePath);
            Console.WriteLine("[Loaded Data]");
            foreach (var column in data.Schema)
            {
                Console.WriteLine($"    {column.Name} <{column.Type} ({column.Type.RawType})>");
            }
            Dynamic.DoClustering(ctx, data);
        }

    }
}
