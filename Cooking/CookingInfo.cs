using System.Collections.Generic;

namespace Cooking
{
    public class CookingInfo
    {
        public int Id { get; set; }
        public string Cuisine { get; set; }
        public List<string> Ingredients { get; set; }
        public string IngredientsAsOneString { get; set; }
    }
}