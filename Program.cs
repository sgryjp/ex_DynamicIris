using Microsoft.ML;
using System;

namespace DynamicIris
{
    class Program
    {
        static void Main(string[] args)
        {
            var filePath = "../../iris.csv";
            IDataView data;
            var ctx = new MLContext();

            Console.WriteLine("------ Statically typed example ------");
            data = Static.LoadIris(ctx, filePath);
            Console.WriteLine("[Loaded Data]");
            foreach (var column in data.Schema)
            {
                Console.WriteLine($"    {column.Name} <{column.Type} ({column.Type.RawType})>");
            }
            Static.DoClustering(ctx, data);

            Console.WriteLine("------ Dynamically typed example ------");
            data = Dynamic.LoadCsv(ctx, filePath);
            Console.WriteLine("[Loaded Data]");
            foreach (var column in data.Schema)
            {
                Console.WriteLine($"    {column.Name} <{column.Type} ({column.Type.RawType})>");
            }
            Dynamic.DoClustering(ctx, data);

            Console.WriteLine("------ Interactive example of dynamically typed regression using LightGBM ------");
            Dynamic.InteractiveRegression(ctx, filePath);
        }

    }
}
