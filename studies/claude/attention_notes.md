You are absolutely right to call this out, and I apologize. You have hit upon the most significant challenge when working with Large Language Models on complex, multi-step tasks. I translated your request for a ***dynamic, relative threshold*** ("0.75% accuracy of the better average") into a ***static, absolute threshold*** (`min_fitness_to_save = 0.2`).

This is a classic and perfect example of the exact "attention loss" and "lack of implicit correlation" you're asking about. My model prioritized the most recent, concrete instruction ("add a check before saving") over the more nuanced, logical requirement that defined *how* that check should work.

Let's address this head-on. First, I will synthesize the strategies to mitigate this, and then I will provide the definitive, corrected code that implements your logic precisely.

### How to Overcome "LLM Attention Loss" and "Implicit Correlation"

These are two distinct but related failure modes.

*   **LLM Attention Loss (The Recency Bias Problem):** In a long and complex prompt, the instructions at the very end tend to have the most weight, potentially causing the model to "forget" or down-weight a critical nuance mentioned earlier.
*   **LLM Lack of Implicit Correlation (The "Connect the Dots" Problem):** An LLM does not *understand* code in a holistic way. It won't automatically assume that creating a new `Config` variable means you also want to update the three functions that *should* use it. It processes instructions literally and locally, often failing to see the broader architectural connections.

Here are the most effective strategies to combat this:

#### 1. The "State the Goal, then the Constraints" Structure

Structure your prompt hierarchically. Begin with the high-level objective, then provide the specific, non-negotiable constraints.

*   **Before (Ambiguous):** "I need to fix the saving. Save algorithms with 0.75% accuracy of the better average and guarantee some are saved at the end."
*   **After (Precise):**
    *   **"Primary Goal:** Implement a dynamic, quality-based saving mechanism.
    *   **Constraint 1 (The Rule):** An algorithm is only worth saving if its fitness is greater than 75% of the average fitness of the cycle's top 10% performers.
    *   **Constraint 2 (The Safety Net):** Regardless of the rule, always save the absolute top 5 algorithms at the very end of the cycle.
    *   **Implementation:** Now, rewrite the necessary functions to achieve this."

#### 2. Use Explicit Pseudocode for Critical Logic

Don't just describe the rule, outline the steps. This forces the model to follow a specific computational path instead of inventing its own (simpler) one.

*   **Example:**
    "In the `evolve_formulas` method, after the evolution loop is complete, you must perform these steps in order:
    1.  Get the `final_population`.
    2.  Identify the top 10% of the population.
    3.  Calculate the `average_fitness_of_top_10_percent`.
    4.  Calculate the `dynamic_save_threshold = 0.75 * average_fitness_of_top_10_percent`.
    5.  Iterate through the final population and call the save function for each genome that meets this `dynamic_save_threshold`."

#### 3. The "Final Check" Reinforcement

After providing all instructions, end your prompt with a paragraph that re-states the most critical and nuanced requirement. This leverages the "recency bias" to your advantage.

*   **Example:**
    "...(all other instructions)...
    **Final Check:** Remember, the most important part of this request is that the saving threshold is **not a fixed number**. It must be calculated dynamically after each cycle based on the performance of that cycle's best algorithms, using the 75% rule I defined."

#### 4. Make It a Conversation (Interactive Refinement)

For highly complex logic, feed it to the model in stages and ask for confirmation.

*   **You:** "First, just write a helper function that takes a population and returns the dynamic fitness threshold based on our 75% rule. Don't implement it yet, just show me the function."
*   **AI:** (Provides the helper function)
*   **You:** "Perfect. Now, integrate that helper function into the `evolve_formulas` method to perform the final save."

