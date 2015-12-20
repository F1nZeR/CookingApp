using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using AForge.Neuro;
using LemmaSharp.Classes;
using Newtonsoft.Json;
using TFIDFExample;

namespace Cooking
{
    class Program
    {
        static void Main(string[] args)
        {
            RunWork();
            Console.ReadKey();
        }

        private static List<CookingInfo> ParseInfo(bool isLearn)
        {
            var filePath = "../../" + (isLearn ? "train.json" : "test.json");
            var items = JsonConvert.DeserializeObject<List<CookingInfo>>(File.ReadAllText(filePath));
            return items;
        }

        private static Tuple<List<CookingInfo>, double[][]> GetItems(bool isLearn)
        {
            var items = ParseInfo(isLearn).ToList();
            var file = File.OpenRead("../../full7z-mlteast-en.lem");
            var lemmatizer = new Lemmatizer(file);
            foreach (var cookingInfo in items)
            {
                var ingrs = cookingInfo.Ingredients.Select(ingredient => lemmatizer.Lemmatize(ingredient.ToLower()).Trim());
                var asOneIngr = string.Join(" ", ingrs);
                cookingInfo.IngredientsAsOneString = asOneIngr;
            }

            var transformedItems = TFIDF.Transform(items.Select(x => x.IngredientsAsOneString).ToArray(), 3);
            var normalizedItems = TFIDF.Normalize(transformedItems);

            return Tuple.Create(items, normalizedItems);
        }

        private static Tuple<List<CookingInfo>, double[][]> GetItems(int from, int to)
        {
            var items = ParseInfo(true).Skip(from).Take(to).ToList();
            if (from == 15000)
            {
                items[0].Ingredients = new List<string>() {"QWOTIUQOPTUJYQWIOKTRFJQWOJTIOWQK"};
            }
            var file = File.OpenRead("../../full7z-mlteast-en.lem");
            var lemmatizer = new Lemmatizer(file);
            foreach (var cookingInfo in items)
            {
                var ingrs = cookingInfo.Ingredients.Select(ingredient => lemmatizer.Lemmatize(ingredient.ToLower()).Trim());
                var asOneIngr = string.Join(" ", ingrs);
                cookingInfo.IngredientsAsOneString = asOneIngr;
            }

            var transformedItems = TFIDF.Transform(items.Select(x => x.IngredientsAsOneString).ToArray(), 3);
            var normalizedItems = TFIDF.Normalize(transformedItems);

            return Tuple.Create(items, normalizedItems);
        }

        private static void RunWork()
        {
            var train = GetItems(true);
            var trainItems = train.Item1;
            var normTrainItems = train.Item2;

            var distinctResults = trainItems.Select(x => x.Cuisine).Distinct().ToList();
            var outputLabels = trainItems.Select(x => distinctResults.IndexOf(x.Cuisine)).ToArray();
            var outputs = Accord.Statistics.Tools.Expand(outputLabels, 20, -1, +1);

            var network = new ActivationNetwork(new BipolarSigmoidFunction(), normTrainItems[0].Length, 500, 20);
            new NguyenWidrow(network).Randomize();
            var teacher = new ParallelResilientBackpropagationLearning(network);
            Console.Out.WriteLine("Network created");

            var error = double.PositiveInfinity;
            while (error > 21500)
            {
                error = teacher.RunEpoch(normTrainItems, outputs);
                Console.Out.WriteLine("error = {0}", error);
            }

            var test = GetItems(false);
            var testItems = test.Item1;
            var testTrainItems = test.Item2;
            var resList = new List<string> { "id,cuisine" };
            for (int i = 0; i < testItems.Count; i++)
            {
                var answer = network.Compute(testTrainItems[i]);
                int actual = answer.ToList().IndexOf(answer.Max());
                var resCousine = distinctResults[actual];
                resList.Add($"{testItems[i].Id},{resCousine}");
            }

            File.WriteAllLines("outresult.csv", resList);

            //var test = GetItems(15000, 500);
            //var testItems = test.Item1;
            //var testTrainItems = test.Item2;
            //var res = testTrainItems[0].Where(x => !double.IsNaN(x));

            //var step = 0;
            //var bestPercent = double.NegativeInfinity;
            //double error = double.PositiveInfinity;
            //while (step <= 5)
            //{
            //    error = teacher.RunEpoch(normTrainItems, outputs);
            //    int correct = 0;
            //    for (int i = 0; i < testItems.Count; i++)
            //    {
            //        var cousine = distinctResults.IndexOf(testItems[i].Cuisine);

            //        var answer = network.Compute(testTrainItems[i]);
            //        int actual = answer.ToList().IndexOf(answer.Max());
            //        int expected = cousine;

            //        if (actual == expected) correct++;
            //    }

            //    var percent = correct * 100f / test.Item1.Count;
            //    if (percent > bestPercent)
            //    {
            //        bestPercent = percent;
            //        step = 0;
            //    }
            //    else
            //    {
            //        step++;
            //    }

            //    Console.Out.WriteLine("error = {0} (correct: {1:p})", error, percent);
            //}

            //Console.Out.WriteLine("Done! Error = {0}", error);


            //var dictInput = new Dictionary<string, int>();
            //for (int i = 0; i < distinctIgredients.Count; i++)
            //{
            //    dictInput.Add(distinctIgredients[i], i);
            //}

            //var dictOutput = new Dictionary<string, int>();
            //for (int i = 0; i < distinctResults.Count; i++)
            //{
            //    dictOutput.Add(distinctResults[i], i);
            //}

            //var inputs = new double[items.Count][];
            //for (int i = 0; i < items.Count; i++)
            //{
            //    var ingredients = items[i].Ingredients;
            //    var obj = new double[distinctIgredients.Count];
            //    foreach (var curIngredient in ingredients)
            //    {
            //        obj[dictInput[curIngredient]] = 1;
            //    }

            //    inputs[i] = obj;
            //}

            //var labels = items.Select(x => dictOutput[x.Cuisine]).ToArray();
            //var outputs = Accord.Statistics.Tools.Expand(labels, distinctResults.Count, -1, +1);

            //var network = new ActivationNetwork(new BipolarSigmoidFunction(), distinctIgredients.Count,
            //   distinctIgredients.Count/2, distinctResults.Count);
            //new NguyenWidrow(network).Randomize();
            //var teacher = new ParallelResilientBackpropagationLearning(network);
            //Console.Out.WriteLine("Network created");

            //double error = double.PositiveInfinity;
            //while (error > 1)
            //{
            //    Console.Out.WriteLine("started");
            //    error = teacher.RunEpoch(inputs, outputs);
            //    Console.Out.WriteLine("error = {0}", error);
            //}

            //var correct = 0;
            //for (int i = 0; i < inputs.Length; i++)
            //{
            //    double[] answer = network.Compute(inputs[i]);

            //    int expected = dictOutput[items[i].Cuisine];
            //    int actual = answer.ToList().IndexOf(answer.Max());

            //    if (expected == actual) correct++;
            //}

            //var percent = correct*100f/inputs.Length;
            //Console.Out.WriteLine($"Result: {percent:N2}%");
            //Console.Out.WriteLine("Done!");
        }
    }
}
