using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DynamicIris
{
    static class Dynamic
    {
        public class DynamicMatrix : IDataView
        {
            readonly object[][] _data;

            public DynamicMatrix(object[][] data, IEnumerable<string> columnNames)
            {
                // Check arguments
                if (data is null)
                {
                    throw new ArgumentNullException(nameof(data));
                }
                if (data.Length == 0)
                {
                    throw new ArgumentException("Input data must contains at least 1 row");
                }
                if (columnNames is null)
                {
                    throw new ArgumentNullException(nameof(columnNames));
                }

                // Determine data type of each column from the first row
                var collectedColumnNames = columnNames.ToArray();
                var builder = new DataViewSchema.Builder();
                for (var i = 0; i < collectedColumnNames.Length; i++)
                {
                    var columnName = collectedColumnNames[i];
                    var firstValue = data[0][i];

                    DataViewType type;
                    if (firstValue is float)
                    {
                        type = NumberDataViewType.Single;
                    }
                    else if (firstValue is DateTime)
                    {
                        type = DateTimeDataViewType.Instance;
                    }
                    else if (firstValue is string)
                    {
                        type = TextDataViewType.Instance;
                    }
                    else
                    {
                        throw new ArgumentException($"Unsupported type of value detected: {firstValue.GetType()}");
                    }
                    builder.AddColumn(columnName, type);
                }
                Schema = builder.ToSchema();

                // Reference all values ensuring its type
                var rows = new List<object[]>();
                foreach (var row in data)
                {
                    //TODO: Check data type consistency here
                    rows.Add(row);
                }
                _data = rows.ToArray();
            }

            public bool CanShuffle { get { return true; } }

            public DataViewSchema Schema { get; }

            public long? GetRowCount()
            {
                return _data.Length;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                return new Cursor(this); // No support for randomization
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                return new[] { new Cursor(this) }; // No support for parallel processing
            }

            private sealed class Cursor : DataViewRowCursor
            {
                readonly DynamicMatrix _owner;
                long _rowIndex;
                readonly Delegate[] _getters;

                public Cursor(DynamicMatrix owner)
                {
                    _owner = owner ?? throw new ArgumentNullException(nameof(owner));
                    _rowIndex = -1;

                    var getters = new List<Delegate>();
                    for (var i = 0; i < _owner.Schema.Count; i++)
                    {
                        var column = _owner.Schema[i];
                        if (column.Type == NumberDataViewType.Single)
                        {
                            getters.Add(new ValueGetter<float>((ref float value) =>
                            {
                                value = (float)_owner._data[_rowIndex][column.Index];
                            }));
                        }
                        else if (column.Type == DateTimeDataViewType.Instance)
                        {
                            getters.Add(new ValueGetter<DateTime>((ref DateTime value) =>
                            {
                                value = (DateTime)_owner._data[_rowIndex][column.Index];
                            }));
                        }
                        else if (column.Type == TextDataViewType.Instance)
                        {
                            getters.Add(new ValueGetter<ReadOnlyMemory<char>>((ref ReadOnlyMemory<char> value) =>
                            {
                                value = (ReadOnlyMemory<char>)_owner._data[_rowIndex][column.Index];
                            }));
                        }
                        else
                        {
                            throw new Exception($"Unsupported type of column detected: {column.Type}");
                        }
                    }
                    _getters = getters.ToArray();
                }

                public override long Position { get { return _rowIndex; } }

                public override long Batch { get { return 0; } }

                public override DataViewSchema Schema { get { return _owner.Schema; } }

                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    if (_getters.Length <= column.Index)
                    {
                        throw new ArgumentOutOfRangeException();
                    }
                    return (ValueGetter<TValue>)_getters[column.Index];
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return (ref DataViewRowId value) => { value = new DataViewRowId((ulong)_rowIndex, 0); };
                }

                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    return column.Index < _getters.Length;
                }

                public override bool MoveNext()
                {
                    if (_rowIndex + 1 < _owner._data.Length)
                    {
                        _rowIndex++;
                        return true;
                    }
                    return false;
                }
            }
        }

        public static IDataView LoadIris(MLContext ctx, string filePath)
        {
            if (ctx is null)
            {
                throw new ArgumentNullException(nameof(ctx));
            }
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentException("message", nameof(filePath));
            }

            // Read CSV file content as object[][] holding typed elements (not strings)
            var columnNames = new List<string>();
            var dataRows = new List<object[]>();
            foreach (var line in File.ReadLines(filePath, Encoding.UTF8))
            {
                // For the first row, parse the fields as column names
                if (columnNames.Count == 0)
                {
                    columnNames.AddRange(line.Split(','));
                    continue;
                }

                // For the rest, parse the fields as data
                var row = new List<object>();
                foreach (var field in line.Split(','))
                {
                    object value;
                    if (float.TryParse(field, out float f))
                    {
                        value = f;
                    }
                    else if (DateTime.TryParse(field, out DateTime d))
                    {
                        value = d;
                    }
                    else
                    {
                        value = field;
                    }
                    row.Add(value);
                }
                dataRows.Add(row.ToArray());
            }
            object[][] objects = dataRows.ToArray();

            return new DynamicMatrix(objects, columnNames);
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

            // Prepare to make a column which contains a vector made from "{sepal|petal}x{length|width}"
            var featuresColumnName = "Features";
            var inputColumnNames = data.Schema.Take(data.Schema.Count - 1).Select(s => s.Name).ToArray();

            // Train
            var estimatorChain = ctx.Transforms
                .Concatenate(featuresColumnName, inputColumnNames) // Make a feature vector column
                .Append(ctx.Transforms.SelectColumns(featuresColumnName)) // Keep only the feature vector column
                .Append(ctx.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
            var transformerChain = estimatorChain.Fit(data);

            // Get output schema by examining the transformation pipeline
            var builder = new DataViewSchema.Builder();
            builder.AddColumn(featuresColumnName, new VectorDataViewType(NumberDataViewType.Single, inputColumnNames.Length));
            var outputSchema = transformerChain.GetOutputSchema(data.Schema);
            Console.WriteLine("[Transformation]");
            Console.WriteLine("    From:");
            foreach (var column in data.Schema)
            {
                Console.WriteLine($"        {column.Name} <{column.Type} ({column.Type.RawType})>");
            }
            Console.WriteLine("    To:");
            foreach (var column in outputSchema)
            {
                Console.WriteLine($"        {column.Name} <{column.Type} ({column.Type.RawType})> ");
            }
            Console.WriteLine();

            // Extract model parameters
            int k;
            var centroids = new VBuffer<float>[3];
            var parameters = transformerChain.LastTransformer.Model;
            parameters.GetClusterCentroids(ref centroids, out k);
            Console.WriteLine("[Model Parameters]");
            Console.WriteLine($"    k: {k}");
            Console.WriteLine($"    centroids:");
            foreach (var centroid in centroids)
            {
                Console.WriteLine($"        ({string.Join(", ", centroid.DenseValues())})");
            }
            Console.WriteLine();

            // Predict a new sample
            var newSamples = new DynamicMatrix(new object[][]
            {
                new object[] { 5.1f, 3.5f, 1.4f, 0.2f },
            },
            inputColumnNames);
            var prediction = transformerChain.Transform(newSamples);
            using (var cursor = prediction.GetRowCursor(prediction.Schema))
            {

                var getFeatures = cursor.GetGetter<VBuffer<float>>(prediction.Schema[featuresColumnName]);
                var getPredictedLabel = cursor.GetGetter<UInt32>(prediction.Schema["PredictedLabel"]);
                var getScore = cursor.GetGetter<VBuffer<float>>(prediction.Schema["Score"]);
                Console.WriteLine("[Prediction]");
                while (cursor.MoveNext())
                {
                    var features = new VBuffer<float>();
                    UInt32 predictedLabel = 0;
                    var score = new VBuffer<float>();

                    getFeatures(ref features);
                    getPredictedLabel(ref predictedLabel);
                    getScore(ref score);
                    Console.WriteLine("    Input Feature:   {0}", string.Join(", ", features.DenseValues()));
                    Console.WriteLine("    Predicted Label: {0}", predictedLabel);
                    Console.WriteLine("    Score:           {0}", string.Join(", ", score.DenseValues()));
                }
                Console.WriteLine();
            }
        }
    }
}
