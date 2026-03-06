import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Cocktail MILP", layout="wide")

st.title("🍸 Cocktail Menu Optimizer (MILP)")
st.markdown("This uses pure Linear Programming. It calculates the exact number of each cocktail to mix to extract the absolute maximum profit from your limited physical inventory.")

# --- 1. INVENTORY DATA (The Constraints) ---
st.subheader("📦 Ingredient Inventory")
st.markdown("Define how much of each ingredient you have in stock, and what it costs you per ounce.")

default_inventory = pd.DataFrame({
    "Ingredient": ["Vodka", "Gin", "Tonic Water", "Lime Juice", "Simple Syrup"],
    "Cost_Per_Oz": [0.80, 1.20, 0.10, 0.30, 0.05],
    "Qty_Available_Oz": [100.0, 80.0, 300.0, 50.0, 40.0]
})

inv_df = st.data_editor(default_inventory, num_rows="dynamic", use_container_width=True)

# --- 2. RECIPE DATA (The Bill of Materials) ---
st.subheader("📜 Cocktail Menu (Recipes & Pricing)")
st.markdown("Define your menu. The algorithm will decide how many of each to make.")

default_menu = pd.DataFrame({
    "Cocktail": ["Vodka Tonic", "Gin & Tonic", "Gimlet", "Vodka Gimlet"],
    "Sell_Price": [12.0, 14.0, 15.0, 13.0],
    "Vodka_oz": [2.0, 0.0, 0.0, 2.0],
    "Gin_oz": [0.0, 2.0, 2.5, 0.0],
    "Tonic Water_oz": [4.0, 4.0, 0.0, 0.0],
    "Lime Juice_oz": [0.5, 0.5, 0.75, 0.75],
    "Simple Syrup_oz": [0.0, 0.0, 0.5, 0.5]
})

menu_df = st.data_editor(default_menu, num_rows="dynamic", use_container_width=True)

run_solver = st.button("🚀 Run MILP Optimization", type="primary", use_container_width=True)

# --- 3. THE MILP SOLVER ---
if run_solver:
    # 1. Initialize the Problem
    # We want to MAXIMIZE profit
    prob = pulp.LpProblem("Cocktail_Optimization", pulp.LpMaximize)

    # 2. Define the Decision Variables
    # These are the things the AI gets to decide: "How many of each drink should I make?"
    # "Integer" means we can't sell half a cocktail. "LowBound=0" means we can't make negative drinks.
    drink_vars = {}
    for _, row in menu_df.iterrows():
        drink_name = row["Cocktail"]
        drink_vars[drink_name] = pulp.LpVariable(drink_name, lowBound=0, cat='Integer')

    # 3. Define the Objective Function (Profit)
    # Profit = (Sell Price - Cost of Ingredients) * Qty Made
    profit_terms = []
    
    # Calculate exact profit margin for each drink
    margins = {}
    for _, drink in menu_df.iterrows():
        drink_name = drink["Cocktail"]
        revenue = drink["Sell_Price"]
        
        cost = 0
        for _, inv in inv_df.iterrows():
            ing_name = inv["Ingredient"]
            if ing_name in menu_df.columns:
                oz_used = drink[ing_name]
                cost += oz_used * inv["Cost_Per_Oz"]
                
        margin = revenue - cost
        margins[drink_name] = margin
        profit_terms.append(margin * drink_vars[drink_name])

    # Add the equation to the problem
    prob += pulp.lpSum(profit_terms), "Total_Profit"

    # 4. Define the Constraints (Inventory Limits)
    # The sum of Vodka used across ALL drinks cannot exceed the Vodka we have in stock.
    for _, inv in inv_df.iterrows():
        ing_name = inv["Ingredient"]
        available_qty = inv["Qty_Available_Oz"]
        
        if ing_name in menu_df.columns:
            usage_terms = []
            for _, drink in menu_df.iterrows():
                drink_name = drink["Cocktail"]
                oz_used = drink[ing_name]
                usage_terms.append(oz_used * drink_vars[drink_name])
                
            # Add the hard constraint to the problem
            prob += pulp.lpSum(usage_terms) <= available_qty, f"Limit_{ing_name}"

    # 5. Solve the Math
    with st.spinner("Calculating Global Optimum..."):
        prob.solve()

    # --- 4. OUTPUT DASHBOARD ---
    st.markdown("---")
    st.subheader("🏆 Optimal Production Plan")
    
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        st.success(f"Mathematical Global Optimum Found! Maximum Profit: **£{pulp.value(prob.objective):,.2f}**")
        
        # Display the results
        results = []
        for drink_name, var in drink_vars.items():
            qty_to_make = var.varValue
            if qty_to_make > 0:
                results.append({
                    "Cocktail to Mix": drink_name,
                    "Qty to Produce": int(qty_to_make),
                    "Unit Margin": f"£{margins[drink_name]:.2f}",
                    "Total Profit Generated": f"£{(qty_to_make * margins[drink_name]):.2f}"
                })
                
        st.table(pd.DataFrame(results))
        
        # Display Inventory Usage
        st.subheader("📊 Inventory Consumption")
        usage_data = []
        for _, inv in inv_df.iterrows():
            ing_name = inv["Ingredient"]
            available = inv["Qty_Available_Oz"]
            
            # Calculate how much we actually used
            used = 0
            for _, drink in menu_df.iterrows():
                if ing_name in menu_df.columns:
                    qty_made = drink_vars[drink["Cocktail"]].varValue
                    used += qty_made * drink[ing_name]
                    
            status = "Depleted (Bottleneck)" if used == available else "Surplus"
            
            usage_data.append({
                "Ingredient": ing_name,
                "Starting Inventory (oz)": available,
                "Amount Used (oz)": used,
                "Remaining (oz)": available - used,
                "Status": status
            })
            
        st.dataframe(pd.DataFrame(usage_data), use_container_width=True)

    else:
        st.error("The solver could not find a valid solution with these constraints.")
