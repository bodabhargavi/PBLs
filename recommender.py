import math
import re


# =========================================================
# FOOD DATASET
# =========================================================

foods = [
    {"food": "Chicken Biryani", "cuisine": "Indian", "diet": "Non-Vegetarian",
     "ingredients": "chicken rice spices", "calories": 650, "protein": 30},
    {"food": "Paneer Tikka", "cuisine": "Indian", "diet": "Vegetarian",
     "ingredients": "paneer yogurt spices", "calories": 350, "protein": 20},
    {"food": "Masala Dosa", "cuisine": "South Indian", "diet": "Vegetarian",
     "ingredients": "rice lentils potato", "calories": 300, "protein": 10},
    {"food": "Veg Fried Rice", "cuisine": "Chinese", "diet": "Vegetarian",
     "ingredients": "rice vegetables soy sauce", "calories": 450, "protein": 9},
    {"food": "Chicken Salad", "cuisine": "Continental", "diet": "Non-Vegetarian",
     "ingredients": "chicken lettuce tomato cucumber", "calories": 280, "protein": 35},
    {"food": "Dal Tadka", "cuisine": "Indian", "diet": "Vegetarian",
     "ingredients": "lentils tomato onion spices", "calories": 250, "protein": 12},
    {"food": "Palak Paneer", "cuisine": "Indian", "diet": "Vegetarian",
     "ingredients": "spinach paneer spices", "calories": 320, "protein": 18},
    {"food": "Grilled Chicken", "cuisine": "Continental", "diet": "Non-Vegetarian",
     "ingredients": "chicken herbs pepper", "calories": 300, "protein": 40},
    {"food": "Idli Sambar", "cuisine": "South Indian", "diet": "Vegetarian",
     "ingredients": "rice lentils vegetables", "calories": 220, "protein": 8},
    {"food": "Chole Bhature", "cuisine": "North Indian", "diet": "Vegetarian",
     "ingredients": "chickpeas flour spices", "calories": 600, "protein": 15},
    {"food": "Veg Pulao", "cuisine": "Indian", "diet": "Vegetarian",
     "ingredients": "rice vegetables spices", "calories": 400, "protein": 10},
    {"food": "Fish Curry", "cuisine": "Indian", "diet": "Non-Vegetarian",
     "ingredients": "fish coconut spices", "calories": 450, "protein": 32},
    {"food": "Fruit Salad", "cuisine": "International", "diet": "Vegan",
     "ingredients": "apple banana orange grapes", "calories": 180, "protein": 3},
    {"food": "Oats Bowl", "cuisine": "International", "diet": "Vegetarian",
     "ingredients": "oats milk banana nuts", "calories": 300, "protein": 12},
    {"food": "Chicken Tikka", "cuisine": "Indian", "diet": "Non-Vegetarian",
     "ingredients": "chicken yogurt spices", "calories": 350, "protein": 30},
]


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text):
    """Convert text to a list of lowercase words."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# Each food becomes one "document": cuisine + diet + ingredients
documents = []

for food in foods:
    text = food["cuisine"] + " " + food["diet"] + " " + food["ingredients"]
    documents.append(tokenize(text))


# =========================================================
# CREATE VOCABULARY
# =========================================================

vocabulary = set()

for document in documents:
    for word in document:
        vocabulary.add(word)

vocabulary = sorted(vocabulary)


# =========================================================
# CALCULATE IDF
# =========================================================

total_documents = len(documents)

idf = {}

for word in vocabulary:
    document_count = 0
    for document in documents:
        if word in document:
            document_count += 1
    idf[word] = math.log(total_documents / (1 + document_count)) + 1


# =========================================================
# CREATE TF-IDF VECTORS
# =========================================================

vectors = []

for document in documents:
    vector = {}
    total_words = len(document)

    for word in vocabulary:
        word_count = document.count(word)
        tf = word_count / total_words
        vector[word] = tf * idf[word]

    vectors.append(vector)


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(vector1, vector2):
    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0

    for word in vocabulary:
        value1 = vector1[word]
        value2 = vector2[word]

        dot_product += value1 * value2
        magnitude1 += value1 * value1
        magnitude2 += value2 * value2

    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# =========================================================
# CREATE SIMILARITY MATRIX
# =========================================================

similarity_matrix = []

for i in range(total_documents):
    row = []
    for j in range(total_documents):
        row.append(cosine_similarity(vectors[i], vectors[j]))
    similarity_matrix.append(row)


# =========================================================
# HELPER FUNCTIONS FOR THE USER INTERFACE
# =========================================================

def get_all_foods():
    """Return the complete dataset (list of dictionaries)."""
    return foods


def get_food_names():
    return [food["food"] for food in foods]


def get_cuisines():
    return sorted(set(food["cuisine"] for food in foods))


def get_diets():
    return sorted(set(food["diet"] for food in foods))


def get_max_calories():
    return max(food["calories"] for food in foods)


def get_max_protein():
    return max(food["protein"] for food in foods)


def get_shared_words(index1, index2):
    """Words that two foods have in common (used for explanation)."""
    common = set(documents[index1]) & set(documents[index2])
    return sorted(common)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_food(
    food_name=None,
    cuisine=None,
    diet=None,
    max_calories=None,
    min_protein=None,
    number_of_recommendations=5
):
    """
    Returns a dictionary:
      {
        "status":  "ok" | "no_match" | "invalid_food",
        "message": text message,
        "results": list of dictionaries (food details + similarity)
      }
    """

    # ---------- APPLY FILTERS ----------
    filtered_indexes = []

    for i, food in enumerate(foods):

        if diet is not None:
            if food["diet"].lower() != diet.lower():
                continue

        if cuisine is not None:
            if food["cuisine"].lower() != cuisine.lower():
                continue

        if max_calories is not None:
            if food["calories"] > max_calories:
                continue

        if min_protein is not None:
            if food["protein"] < min_protein:
                continue

        filtered_indexes.append(i)

    # ---------- CHECK RESULTS ----------
    if len(filtered_indexes) == 0:
        return {
            "status": "no_match",
            "message": "No foods match your selected criteria.",
            "results": []
        }

    # ---------- FIND SELECTED FOOD ----------
    selected_index = -1

    if food_name is not None:

        for i, food in enumerate(foods):
            if food["food"].lower() == food_name.lower():
                selected_index = i
                break

        if selected_index == -1:
            return {
                "status": "invalid_food",
                "message": "Food not found: " + str(food_name),
                "results": []
            }

        # ---------- CALCULATE SIMILARITY ----------
        recommendations = []

        for index in filtered_indexes:
            if index == selected_index:
                continue
            score = similarity_matrix[selected_index][index]
            recommendations.append((index, score))

        # ---------- SORT BY SIMILARITY ----------
        recommendations.sort(key=lambda x: x[1], reverse=True)

    else:
        # No food selected: just return the filtered foods (no similarity)
        recommendations = [(index, None) for index in filtered_indexes]

    # ---------- BUILD OUTPUT ----------
    results = []

    for index, score in recommendations[:number_of_recommendations]:
        item = dict(foods[index])          # copy of the food details
        item["similarity"] = score
        if selected_index != -1:
            item["shared_words"] = get_shared_words(selected_index, index)
        else:
            item["shared_words"] = []
        results.append(item)

    if len(results) == 0:
        return {
            "status": "no_match",
            "message": "No other foods match your selected criteria.",
            "results": []
        }

    return {
        "status": "ok",
        "message": "Found " + str(len(results)) + " recommendation(s).",
        "results": results
    }