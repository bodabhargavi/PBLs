import streamlit as st
import recommender as rec

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Food Recommendation System",
    page_icon="🍽️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {text-align:center; font-size:2.6rem; font-weight:700; margin-bottom:0;}
    .sub-title  {text-align:center; font-size:1.2rem; color:gray; margin-bottom:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<p class="main-title">🍽️ AI Food Recommendation System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Smart and Personalized Food Recommendations Using AI</p>',
            unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------
# HELPER: emoji for each diet
# ---------------------------------------------------------
def diet_icon(diet):
    if diet == "Non-Vegetarian":
        return "🍗"
    if diet == "Vegan":
        return "🥗"
    return "🥦"


# ---------------------------------------------------------
# SIDEBAR: USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("🎛️ Your Preferences")

NONE_OPTION = "-- None (only filter foods) --"

selected_food = st.sidebar.selectbox(
    "1. Select Food",
    [NONE_OPTION] + rec.get_food_names()
)

selected_diet = st.sidebar.selectbox(
    "2. Select Diet",
    ["All"] + rec.get_diets()
)

selected_cuisine = st.sidebar.selectbox(
    "3. Select Cuisine",
    ["All"] + rec.get_cuisines()
)

max_calories = st.sidebar.slider(
    "4. Maximum Calories (kcal)",
    min_value=0,
    max_value=rec.get_max_calories(),
    value=rec.get_max_calories(),
    step=10
)

min_protein = st.sidebar.slider(
    "5. Minimum Protein (g)",
    min_value=0,
    max_value=rec.get_max_protein(),
    value=0,
    step=1
)

number_of_recommendations = st.sidebar.selectbox(
    "6. Number of Recommendations",
    [3, 5, 10],
    index=1
)

recommend_clicked = st.sidebar.button(
    "🔍 Recommend Food",
    type="primary",
    use_container_width=True
)

st.sidebar.info("Choose your preferences and click **Recommend Food**.")


# ---------------------------------------------------------
# RECOMMENDATION RESULTS
# ---------------------------------------------------------
if not recommend_clicked:
    st.info("👈 Use the sidebar to choose your preferences, then click "
            "**🔍 Recommend Food**.")

else:
    # Convert UI choices into function inputs
    food_name = None if selected_food == NONE_OPTION else selected_food
    diet = None if selected_diet == "All" else selected_diet
    cuisine = None if selected_cuisine == "All" else selected_cuisine

    output = rec.recommend_food(
        food_name=food_name,
        cuisine=cuisine,
        diet=diet,
        max_calories=max_calories,
        min_protein=min_protein,
        number_of_recommendations=number_of_recommendations
    )

    status = output["status"]
    results = output["results"]

    if status == "no_match":
        st.warning("⚠️ No foods match your selected criteria.\n\n"
                   "Try changing the calorie, protein, diet, or cuisine filters.")

    elif status == "invalid_food":
        st.error("❌ The selected food was not found in the dataset. "
                 "Please choose a food from the dropdown.")

    else:
        if food_name is not None:
            st.success("✅ Showing foods similar to **" + food_name + "**")
        else:
            st.success("✅ Showing foods that match your filters")

        st.subheader("⭐ Recommended Foods")

        # ----- cards, 3 per row -----
        for start in range(0, len(results), 3):
            row_items = results[start:start + 3]
            columns = st.columns(3)

            for column, item in zip(columns, row_items):
                with column:
                    with st.container(border=True):
                        st.markdown("### " + diet_icon(item["diet"]) + " " + item["food"])
                        st.write("**Cuisine:** " + item["cuisine"])
                        st.write("**Diet:** " + item["diet"])
                        st.write("🔥 **Calories:** " + str(item["calories"]) + " kcal")
                        st.write("💪 **Protein:** " + str(item["protein"]) + " g")

                        if item["similarity"] is not None:
                            score = float(item["similarity"])
                            st.write("🎯 **Similarity:** " + format(score, ".3f"))
                            st.progress(min(max(score, 0.0), 1.0))

        # ----- WHY THIS FOOD -----
        st.subheader("💡 Why this food?")
        st.write(
            "Recommendations are produced in these steps:\n"
            "1. **Filtering:** foods are removed if they do not match the chosen "
            "diet, cuisine, maximum calories or minimum protein.\n"
            "2. **TF-IDF:** each food's text (cuisine + diet + ingredients) is "
            "converted into a numeric TF-IDF vector.\n"
            "3. **Cosine similarity:** the selected food's vector is compared with "
            "every remaining food.\n"
            "4. **Ranking:** foods are sorted from highest to lowest similarity."
        )
        st.caption("Note: calories and protein are used as filters only. "
                   "They are not part of the TF-IDF vectors.")

        if food_name is not None:
            for item in results:
                shared = ", ".join(item["shared_words"]) if item["shared_words"] else "none"
                st.write(
                    "• **" + item["food"] + "** — similarity "
                    + format(float(item["similarity"]), ".3f")
                    + " | words in common with " + food_name + ": *" + shared + "*"
                )
        else:
            st.info("Select a food in the sidebar to see similarity scores.")


# ---------------------------------------------------------
# DATASET SECTION
# ---------------------------------------------------------
st.divider()
st.subheader("📊 Food Dataset")

with st.expander("Click to view the complete food dataset"):
    table_data = []
    for item in rec.get_all_foods():
        table_data.append({
            "Food": item["food"],
            "Cuisine": item["cuisine"],
            "Diet": item["diet"],
            "Calories": item["calories"],
            "Protein": item["protein"]
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

st.caption("College Project: AI-Based Food Recommendation System | "
           "Built with Python and Streamlit")