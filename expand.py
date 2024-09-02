import json
import random

# Original dataset
original_data = [
    {"question": "How does rabies work in dogs?", "answer": "Dr. Vikas said that - Rabies is a viral disease that affects the nervous system of mammals, including dogs. It is transmitted through the bite of an infected animal and leads to inflammation of the brain."},
    {"question": "What are the symptoms of rabies in dogs?", "answer": "Dr. Vikas said that - Symptoms of rabies in dogs include behavioral changes, aggression, difficulty swallowing, excessive salivation, and paralysis."},
    {"question": "How is rabies transmitted?", "answer": "Dr. Vikas said that - Rabies is transmitted through the saliva of an infected animal, usually via bites or scratches. The virus enters the body through broken skin and travels through the nervous system."},
    {"question": "Can rabies be prevented in dogs?", "answer": "Dr. Vikas said that - Yes, rabies can be prevented through vaccination. Dogs should receive their rabies vaccine as part of their routine vaccination schedule to protect them from this fatal disease."},
    {"question": "What should you do if a dog is bitten by a potentially rabid animal?", "answer": "Dr. Vikas said that - If a dog is bitten by a potentially rabid animal, it is crucial to seek veterinary care immediately. The vet may administer a rabies booster shot and assess the need for additional treatment based on the exposure risk."},
    {"question": "What is the incubation period for rabies in dogs?", "answer": "Dr. Vikas said that - The incubation period for rabies in dogs can vary, but it is typically between one to three months. During this time, the virus travels from the site of the bite to the central nervous system before symptoms appear."},
    {"question": "Can humans get rabies from dogs?", "answer": "Dr. Vikas said that - Yes, humans can contract rabies from dogs if they are bitten or scratched by an infected animal. It is important for anyone exposed to a potentially rabid animal to receive prompt medical attention and post-exposure prophylaxis."},
    {"question": "What are the stages of rabies in dogs?", "answer": "Dr. Vikas said that - Rabies in dogs typically progresses through three stages: the prodromal stage (behavioral changes), the excitatory stage (aggression, hallucinations), and the paralytic stage (paralysis, death)."},
    {"question": "How is rabies diagnosed in dogs?", "answer": "Dr. Vikas said that - Rabies is diagnosed through laboratory tests on brain tissue. However, diagnosis is usually based on clinical signs and history of exposure. Testing for rabies in living animals is challenging, and post-mortem examination is often required."},
    {"question": "What is the treatment for rabies in dogs?", "answer": "Dr. Vikas said that - There is no effective treatment for rabies once clinical symptoms have appeared. Rabies is almost always fatal once it reaches the symptomatic stage. Prevention through vaccination is the most effective strategy."},
    {"question": "Can rabies be treated if detected early?", "answer": "Dr. Vikas said that - In dogs, once rabies symptoms appear, treatment is not effective. However, if a dog is exposed to rabies and has not shown symptoms, immediate vaccination and possible post-exposure prophylaxis can prevent the disease."},
    {"question": "What are the legal requirements regarding rabies vaccination for dogs?", "answer": "Dr. Vikas said that - Rabies vaccination laws vary by region, but most places require dogs to be vaccinated against rabies and have proof of vaccination. Check local regulations for specific requirements in your area."},
    {"question": "How does rabies affect the brain?", "answer": "Dr. Vikas said that - Rabies virus primarily affects the brain and central nervous system, causing severe inflammation and disruption of normal brain function, which leads to the behavioral and neurological symptoms observed in infected animals."}
]

def generate_expanded_data(original_data, num_entries):
    expanded_data = []
    
    # Helper function to create a variation of a question and answer
    def create_variation(item):
        question_variations = [
            item['question'] + " Please elaborate.",
            item['question'] + " Can you explain further?",
            item['question'] + " What more can you tell about this?",
            item['question'] + " Could you provide more details?",
            "What is the detailed process of " + item['question'],
            "Can you describe the " + item['question'].lower() + " in more detail?",
            "How does " + item['question'].lower() + " impact dogs?"
        ]
        
        answer_variations = [
            item['answer'] + " Here’s a more detailed explanation: [Additional information].",
            item['answer'] + " For a deeper understanding: [Extra context].",
            item['answer'] + " More insights include: [Supplementary details].",
            item['answer'] + " To elaborate further: [Expanded context].",
            item['answer'] + " Additionally: [More in-depth information]."
        ]
        
        question = random.choice(question_variations)
        answer = random.choice(answer_variations)
        return {"question": question, "answer": answer}
    
    while len(expanded_data) < num_entries:
        for item in original_data:
            if len(expanded_data) >= num_entries:
                break
            expanded_data.append(create_variation(item))
    
    return expanded_data

# Generate dataset with 1000 entries
expanded_dataset = generate_expanded_data(original_data, 3000)

# Save to JSON file
with open('expanded_dataset.json', 'w') as f:
    json.dump(expanded_dataset, f, indent=4)

print("Expanded dataset saved to 'expanded_dataset.json'.")
