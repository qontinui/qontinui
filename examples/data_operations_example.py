"""Example usage of data operations module.

Demonstrates:
- Setting variables from different sources
- Getting variables with defaults
- Sorting collections
- Filtering collections
- Safe expression evaluation
"""

from qontinui.actions.data_operations import DataOperationsExecutor, VariableContext
from qontinui.config import (
    Action,
)


def example_set_variable():
    """Example: Set variables from different sources."""
    print("\n=== SET_VARIABLE Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Example 1: Set variable with direct value
    print("1. Setting variable with direct value:")
    action = Action(
        id="set_gold",
        type="SET_VARIABLE",
        config={"variableName": "player_gold", "value": 1000, "scope": "global"},
    )
    result = executor.execute_set_variable(action, context)
    print(f"   Result: {result}")
    print(f"   Variable value: {executor.variable_context.get('player_gold')}")

    # Example 2: Set variable from expression
    print("\n2. Setting variable from expression:")
    action = Action(
        id="set_total",
        type="SET_VARIABLE",
        config={
            "variableName": "total_gold",
            "valueSource": {"type": "expression", "expression": "player_gold * 2 + 500"},
            "scope": "local",
        },
    )
    result = executor.execute_set_variable(action, context)
    print(f"   Result: {result}")
    print(f"   Variable value: {executor.variable_context.get('total_gold')}")

    # Example 3: Set variable with type coercion
    print("\n3. Setting variable with type coercion:")
    action = Action(
        id="set_string_num",
        type="SET_VARIABLE",
        config={"variableName": "price", "value": "123.45", "type": "number", "scope": "local"},
    )
    result = executor.execute_set_variable(action, context)
    print(f"   Result: {result}")
    print(
        f"   Variable value: {executor.variable_context.get('price')} (type: {type(executor.variable_context.get('price')).__name__})"
    )


def example_get_variable():
    """Example: Get variables with defaults."""
    print("\n=== GET_VARIABLE Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Set up some test variables
    executor.variable_context.set("username", "player1", "local")
    executor.variable_context.set("score", 500, "global")

    # Example 1: Get existing variable
    print("1. Getting existing variable:")
    action = Action(id="get_username", type="GET_VARIABLE", config={"variableName": "username"})
    result = executor.execute_get_variable(action, context)
    print(f"   Result: {result}")

    # Example 2: Get non-existent variable with default
    print("\n2. Getting non-existent variable with default:")
    action = Action(
        id="get_level", type="GET_VARIABLE", config={"variableName": "level", "defaultValue": 1}
    )
    result = executor.execute_get_variable(action, context)
    print(f"   Result: {result}")

    # Example 3: Get variable and store in output variable
    print("\n3. Getting variable and storing in output variable:")
    action = Action(
        id="get_score",
        type="GET_VARIABLE",
        config={"variableName": "score", "outputVariable": "current_score"},
    )
    result = executor.execute_get_variable(action, context)
    print(f"   Result: {result}")
    print(f"   Output variable value: {executor.variable_context.get('current_score')}")


def example_sort():
    """Example: Sort collections."""
    print("\n=== SORT Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    items = [
        {"name": "sword", "price": 150, "level": 5},
        {"name": "shield", "price": 200, "level": 3},
        {"name": "potion", "price": 50, "level": 1},
        {"name": "armor", "price": 300, "level": 7},
    ]
    executor.variable_context.set("items", items, "local")

    # Example 1: Sort by price (ascending)
    print("1. Sort items by price (ascending):")
    print(f"   Original: {[item['name'] for item in items]}")
    action = Action(
        id="sort_price_asc",
        type="SORT",
        config={
            "target": "variable",
            "variableName": "items",
            "sortBy": "price",
            "order": "ASC",
            "comparator": "NUMERIC",
            "outputVariable": "sorted_items_asc",
        },
    )
    result = executor.execute_sort(action, context)
    sorted_items = result["sorted_collection"]
    print(f"   Sorted: {[item['name'] for item in sorted_items]}")
    print(f"   Prices: {[item['price'] for item in sorted_items]}")

    # Example 2: Sort by level (descending)
    print("\n2. Sort items by level (descending):")
    action = Action(
        id="sort_level_desc",
        type="SORT",
        config={
            "target": "variable",
            "variableName": "items",
            "sortBy": "level",
            "order": "DESC",
            "comparator": "NUMERIC",
            "outputVariable": "sorted_items_desc",
        },
    )
    result = executor.execute_sort(action, context)
    sorted_items = result["sorted_collection"]
    print(f"   Sorted: {[item['name'] for item in sorted_items]}")
    print(f"   Levels: {[item['level'] for item in sorted_items]}")

    # Example 3: Sort by name (alphabetically)
    print("\n3. Sort items by name (alphabetically):")
    action = Action(
        id="sort_name",
        type="SORT",
        config={
            "target": "variable",
            "variableName": "items",
            "sortBy": "name",
            "order": "ASC",
            "comparator": "ALPHABETIC",
        },
    )
    result = executor.execute_sort(action, context)
    sorted_items = result["sorted_collection"]
    print(f"   Sorted: {[item['name'] for item in sorted_items]}")


def example_filter():
    """Example: Filter collections."""
    print("\n=== FILTER Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    items = [
        {"name": "sword", "price": 150, "level": 5, "rarity": "rare"},
        {"name": "shield", "price": 200, "level": 3, "rarity": "common"},
        {"name": "potion", "price": 50, "level": 1, "rarity": "common"},
        {"name": "armor", "price": 300, "level": 7, "rarity": "epic"},
        {"name": "ring", "price": 400, "level": 6, "rarity": "rare"},
    ]
    executor.variable_context.set("items", items, "local")

    # Example 1: Filter by property (price > 100)
    print("1. Filter items with price > 100:")
    print(f"   Original count: {len(items)}")
    action = Action(
        id="filter_expensive",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {"type": "property", "property": "price", "operator": ">", "value": 100},
            "outputVariable": "expensive_items",
        },
    )
    result = executor.execute_filter(action, context)
    filtered_items = result["filtered_collection"]
    print(f"   Filtered count: {len(filtered_items)}")
    print(f"   Items: {[(item['name'], item['price']) for item in filtered_items]}")

    # Example 2: Filter by expression (level >= 5)
    print("\n2. Filter items with level >= 5:")
    action = Action(
        id="filter_high_level",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {"type": "expression", "expression": "item['level'] >= 5"},
            "outputVariable": "high_level_items",
        },
    )
    result = executor.execute_filter(action, context)
    filtered_items = result["filtered_collection"]
    print(f"   Filtered count: {len(filtered_items)}")
    print(f"   Items: {[(item['name'], item['level']) for item in filtered_items]}")

    # Example 3: Filter by string matching (rarity == "rare")
    print("\n3. Filter items with rarity == 'rare':")
    action = Action(
        id="filter_rare",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {
                "type": "property",
                "property": "rarity",
                "operator": "==",
                "value": "rare",
            },
            "outputVariable": "rare_items",
        },
    )
    result = executor.execute_filter(action, context)
    filtered_items = result["filtered_collection"]
    print(f"   Filtered count: {len(filtered_items)}")
    print(f"   Items: {[(item['name'], item['rarity']) for item in filtered_items]}")

    # Example 4: Complex filter (price > 150 AND level >= 5)
    print("\n4. Filter items with price > 150 AND level >= 5:")
    action = Action(
        id="filter_complex",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {
                "type": "expression",
                "expression": "item['price'] > 150 and item['level'] >= 5",
            },
            "outputVariable": "premium_items",
        },
    )
    result = executor.execute_filter(action, context)
    filtered_items = result["filtered_collection"]
    print(f"   Filtered count: {len(filtered_items)}")
    print(f"   Items: {[(item['name'], item['price'], item['level']) for item in filtered_items]}")


def example_variable_scopes():
    """Example: Variable scopes."""
    print("\n=== Variable Scopes Example ===\n")

    context = VariableContext()

    # Set variables in different scopes
    context.set("user", "local_user", "local")
    context.set("user", "global_user", "global")
    context.set("score", 100, "process")

    print("Variables set:")
    print("  Local: user = 'local_user'")
    print("  Global: user = 'global_user'")
    print("  Process: score = 100")

    # Get variable (should return local first)
    print("\nGetting 'user' (should prioritize local):")
    print(f"  Result: {context.get('user')}")

    # Delete local variable
    context.delete("user", "local")
    print("\nAfter deleting local 'user', getting 'user':")
    print(f"  Result: {context.get('user')} (now returns global)")

    # Get all variables
    print("\nAll variables:")
    all_vars = context.get_all_variables()
    for name, value in all_vars.items():
        print(f"  {name} = {value}")


def example_map_operation():
    """Example: MAP operation."""
    print("\n=== MAP Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    numbers = [1, 2, 3, 4, 5]
    executor.variable_context.set("numbers", numbers, "local")

    # Example 1: Map with expression (multiply by 2)
    print("1. Map numbers to double their value:")
    print(f"   Original: {numbers}")

    map_action = Action(
        id="map-1",
        type="MAP",
        config={
            "variableName": "numbers",
            "transform": {"type": "expression", "expression": "item * 2"},
            "outputVariable": "doubled",
        },
    )

    result = executor.execute_map(map_action, context)
    doubled = result["mapped_collection"]
    print(f"   Doubled: {doubled}")

    # Example 2: Map objects extracting property
    print("\n2. Map objects extracting 'name' property:")
    items = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]
    executor.variable_context.set("items", items, "local")
    print(f"   Original: {[item['name'] for item in items]}")

    map_action2 = Action(
        id="map-2",
        type="MAP",
        config={
            "variableName": "items",
            "transform": {"type": "property", "property": "name"},
            "outputVariable": "names",
        },
    )

    result = executor.execute_map(map_action2, context)
    names = result["mapped_collection"]
    print(f"   Names: {names}")

    # Example 3: Complex transformation with expression
    print("\n3. Map to calculate (age * 2 + 10):")
    map_action3 = Action(
        id="map-3",
        type="MAP",
        config={
            "variableName": "items",
            "transform": {"type": "expression", "expression": "item['age'] * 2 + 10"},
            "outputVariable": "transformed_ages",
        },
    )

    result = executor.execute_map(map_action3, context)
    transformed = result["mapped_collection"]
    print(f"   Transformed ages: {transformed}")


def example_reduce_operation():
    """Example: REDUCE operation."""
    print("\n=== REDUCE Examples ===\n")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    executor.variable_context.set("numbers", numbers, "local")

    # Example 1: Sum
    print("1. Reduce to sum:")
    print(f"   Numbers: {numbers}")

    reduce_action = Action(
        id="reduce-1",
        type="REDUCE",
        config={"variableName": "numbers", "operation": "sum", "outputVariable": "total"},
    )

    result = executor.execute_reduce(reduce_action, context)
    print(f"   Sum: {result['reduced_value']}")

    # Example 2: Average
    print("\n2. Reduce to average:")
    reduce_action2 = Action(
        id="reduce-2",
        type="REDUCE",
        config={"variableName": "numbers", "operation": "average", "outputVariable": "avg"},
    )

    result = executor.execute_reduce(reduce_action2, context)
    print(f"   Average: {result['reduced_value']}")

    # Example 3: Min and Max
    print("\n3. Reduce to min and max:")
    reduce_min = Action(
        id="reduce-3a",
        type="REDUCE",
        config={"variableName": "numbers", "operation": "min", "outputVariable": "minimum"},
    )

    reduce_max = Action(
        id="reduce-3b",
        type="REDUCE",
        config={"variableName": "numbers", "operation": "max", "outputVariable": "maximum"},
    )

    min_result = executor.execute_reduce(reduce_min, context)
    max_result = executor.execute_reduce(reduce_max, context)
    print(f"   Min: {min_result['reduced_value']}")
    print(f"   Max: {max_result['reduced_value']}")

    # Example 4: Custom reducer (concatenate with separator)
    print("\n4. Reduce strings with custom reducer:")
    words = ["Hello", "World", "from", "Python"]
    executor.variable_context.set("words", words, "local")
    print(f"   Words: {words}")

    reduce_custom = Action(
        id="reduce-4",
        type="REDUCE",
        config={
            "variableName": "words",
            "operation": "custom",
            "initialValue": "",
            "customReducer": "acc + (' ' if acc else '') + item",
            "outputVariable": "sentence",
        },
    )

    result = executor.execute_reduce(reduce_custom, context)
    print(f"   Sentence: '{result['reduced_value']}'")

    # Example 5: Product (multiply all)
    print("\n5. Reduce to product (multiply all numbers):")
    small_numbers = [2, 3, 4, 5]
    executor.variable_context.set("small_numbers", small_numbers, "local")
    print(f"   Numbers: {small_numbers}")

    reduce_product = Action(
        id="reduce-5",
        type="REDUCE",
        config={
            "variableName": "small_numbers",
            "operation": "custom",
            "initialValue": 1,
            "customReducer": "acc * item",
            "outputVariable": "product",
        },
    )

    result = executor.execute_reduce(reduce_product, context)
    print(f"   Product: {result['reduced_value']}")


def example_safe_evaluation():
    """Example: Safe expression evaluation."""
    print("\n=== Safe Expression Evaluation ===\n")

    from qontinui.actions.data_operations import SafeEvaluator

    evaluator = SafeEvaluator()

    # Safe expressions
    safe_expressions = [
        ("2 + 2", {}),
        ("x * 2 + 10", {"x": 5}),
        ("len([1, 2, 3, 4])", {}),
        ("max(a, b)", {"a": 10, "b": 20}),
        ("[x * 2 for x in range(5)]", {}),
    ]

    print("Safe expressions:")
    for expr, ctx in safe_expressions:
        try:
            result = evaluator.safe_eval(expr, ctx)
            print(f"  {expr:30} = {result}")
        except Exception as e:
            print(f"  {expr:30} ERROR: {e}")

    # Unsafe expressions (should fail)
    print("\nUnsafe expressions (should be blocked):")
    unsafe_expressions = [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "exec('print(123)')",
    ]

    for expr in unsafe_expressions:
        try:
            result = evaluator.safe_eval(expr, {})
            print(f"  {expr:30} = {result} (SECURITY ISSUE!)")
        except Exception as e:
            print(f"  {expr:30} BLOCKED: {type(e).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Operations Module Examples")
    print("=" * 60)

    example_set_variable()
    example_get_variable()
    example_sort()
    example_filter()
    example_map_operation()
    example_reduce_operation()
    example_variable_scopes()
    example_safe_evaluation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
