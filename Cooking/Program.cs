using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.Neuro;
using Accord.Neuro.Learning;
using AForge.Neuro;
using Newtonsoft.Json;

namespace Cooking
{
    class Program
    {
        static void Main(string[] args)
        {
            RunWork();
            Console.ReadKey();
        }

        private static List<CookingInfo> ParseInfo()
        {
            var filePath = "../../train.json";
            var items = JsonConvert.DeserializeObject<List<CookingInfo>>(File.ReadAllText(filePath));
            return items;
        }

        private static void RunWork()
        {
            var items = ParseInfo().Take(1000).ToList();

            var distinctIgredients = items.SelectMany(x => x.Ingredients).Distinct().ToList();
            var distinctResults = items.Select(x => x.Cuisine).Distinct().ToList();
            //var maxLength = items.Select(x => x.Ingredients.Count).Max();

            var dictInput = new Dictionary<string, int>();
            for (int i = 0; i < distinctIgredients.Count; i++)
            {
                dictInput.Add(distinctIgredients[i], i);
            }

            var dictOutput = new Dictionary<string, int>();
            for (int i = 0; i < distinctResults.Count; i++)
            {
                dictOutput.Add(distinctResults[i], i);
            }

            var inputs = new double[items.Count][];
            for (int i = 0; i < items.Count; i++)
            {
                var ingredients = items[i].Ingredients;
                var obj = new double[distinctIgredients.Count];
                foreach (var curIngredient in ingredients)
                {
                    obj[dictInput[curIngredient]] = 1;
                }

                inputs[i] = obj;
            }

            var labels = items.Select(x => dictOutput[x.Cuisine]).ToArray();
            var outputs = Accord.Statistics.Tools.Expand(labels, distinctResults.Count, -1, +1);

            var network = new ActivationNetwork(new BipolarSigmoidFunction(), distinctIgredients.Count,
               100, distinctResults.Count);
            new NguyenWidrow(network).Randomize();
            var teacher = new ParallelResilientBackpropagationLearning(network);
            Console.Out.WriteLine("Network created");

            double error = double.PositiveInfinity;
            while (error > 1)
            {
                error = teacher.RunEpoch(inputs, outputs);
            }

            var correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] answer = network.Compute(inputs[i]);

                int expected = dictOutput[items[i].Cuisine];
                int actual = answer.ToList().IndexOf(answer.Max());

                if (expected == actual) correct++;
            }

            var percent = correct*100f/inputs.Length;
            Console.Out.WriteLine($"Result: {percent:N2}%");
            Console.Out.WriteLine("Done!");
        }
    }
}
