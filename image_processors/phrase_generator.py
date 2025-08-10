"""
Human Situation Phrase Generator

A script that generates random phrases depicting various situations humans might find themselves in.
Uses modular building blocks (verbs, adjectives, nouns, etc.) stored in dictionaries to create
diverse and interesting scenario descriptions.
"""

import random
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
from datetime import datetime

class PhraseGenerator:
    def __init__(self):
        # Building block dictionaries
        self.verbs_action = [
            "driving", "cooking", "building", "painting", "writing", "reading", "singing", 
            "dancing", "running", "walking", "climbing", "swimming", "flying", "sailing",
            "gardening", "teaching", "learning", "fixing", "cleaning", "organizing",
            "creating", "designing", "crafting", "exploring", "discovering", "investigating",
            "performing", "practicing", "training", "exercising", "meditating", "relaxing"
        ]
        
        self.verbs_being = [
            "being", "becoming", "remaining", "staying", "appearing", "seeming", "feeling",
            "looking", "sounding", "acting", "proving", "turning", "growing", "getting"
        ]
        
        self.adjectives_mood = [
            "happy", "excited", "calm", "peaceful", "energetic", "tired", "focused",
            "determined", "curious", "adventurous", "creative", "thoughtful", "confident",
            "nervous", "relaxed", "enthusiastic", "patient", "careful", "bold", "gentle"
        ]
        
        self.adjectives_descriptive = [
            "beautiful", "complex", "simple", "colorful", "ancient", "modern", "rustic",
            "elegant", "sturdy", "delicate", "massive", "tiny", "smooth", "rough",
            "bright", "dark", "loud", "quiet", "fast", "slow", "hot", "cold"
        ]
        
        self.nouns_objects = [
            "tractor", "bicycle", "motorcycle", "boat", "airplane", "train", "car",
            "computer", "camera", "guitar", "piano", "book", "painting", "sculpture",
            "garden", "kitchen", "workshop", "studio", "library", "museum", "theater",
            "mountain", "river", "forest", "beach", "desert", "city", "village"
        ]
        
        self.nouns_food = [
            "meal", "soup", "bread", "cake", "pizza", "salad", "pasta", "curry",
            "sandwich", "smoothie", "coffee", "tea", "wine", "cheese", "chocolate",
            "fruit", "vegetables", "herbs", "spices", "sauce", "dessert"
        ]
        
        self.nouns_activities = [
            "lesson", "presentation", "performance", "concert", "game", "sport", "race",
            "competition", "exhibition", "workshop", "seminar", "meeting", "conversation",
            "interview", "ceremony", "celebration", "festival", "party", "adventure"
        ]
        
        self.nouns_places = [
            "home", "office", "school", "hospital", "park", "beach", "mountains",
            "countryside", "city center", "marketplace", "restaurant", "cafe", "gym",
            "studio", "workshop", "laboratory", "library", "gallery", "theater"
        ]
        
        self.prepositions = [
            "in", "on", "at", "by", "near", "under", "over", "through", "around",
            "beside", "behind", "in front of", "next to", "above", "below"
        ]
        
        self.articles = ["a", "an", "the"]
        
        self.conjunctions = ["while", "as", "when", "after", "before", "during"]
        
        self.adverbs_manner = [
            "carefully", "quickly", "slowly", "gently", "skillfully", "creatively",
            "passionately", "enthusiastically", "patiently", "confidently", "gracefully",
            "efficiently", "thoughtfully", "playfully", "seriously", "casually"
        ]
        
        self.weather_conditions = [
            "sunny", "rainy", "cloudy", "windy", "snowy", "foggy", "stormy",
            "clear", "overcast", "humid", "dry", "cool", "warm", "freezing"
        ]
        
        # Phrase templates
        self.templates = [
            "{adverb_manner} {verb_action} {article} {adjective_descriptive} {noun_object}",
            "{verb_action} {article} {noun_food} {preposition} {article} {adjective_descriptive} {noun_place}",
            "{verb_being} {adjective_mood} {conjunction} {verb_action} {preposition} {article} {noun_place}",
            "{verb_action} {article} {noun_activity} {preposition} {article} {noun_place}",
            "{verb_action} {adverb_manner} {preposition} {article} {weather_condition} day",
            "{adjective_mood} person {verb_action} {article} {adjective_descriptive} {noun_object}",
            "{verb_action} {article} {noun_food} {conjunction} {verb_being} {adjective_mood}",
            "{verb_action} {preposition} {article} {adjective_descriptive} {noun_place}",
            "{verb_being} {adjective_mood} {conjunction} {verb_action} {article} {noun_activity}",
            "{adverb_manner} {verb_action} {conjunction} the weather is {weather_condition}"
        ]

    def get_random_word(self, category):
        """Get a random word from the specified category"""
        category_map = {
            'verb_action': self.verbs_action,
            'verb_being': self.verbs_being,
            'adjective_mood': self.adjectives_mood,
            'adjective_descriptive': self.adjectives_descriptive,
            'noun_object': self.nouns_objects,
            'noun_food': self.nouns_food,
            'noun_activity': self.nouns_activities,
            'noun_place': self.nouns_places,
            'preposition': self.prepositions,
            'article': self.articles,
            'conjunction': self.conjunctions,
            'adverb_manner': self.adverbs_manner,
            'weather_condition': self.weather_conditions
        }
        
        return random.choice(category_map.get(category, ['unknown']))

    def choose_article(self, next_word):
        """Choose appropriate article (a/an) based on the next word"""
        if next_word.lower().startswith(('a', 'e', 'i', 'o', 'u')):
            return 'an'
        return 'a'

    def generate_phrase(self):
        """Generate a random phrase using the templates and word categories"""
        template = random.choice(self.templates)
        
        # First pass: replace non-article placeholders
        result = template
        placeholders = ['verb_action', 'verb_being', 'adjective_mood', 'adjective_descriptive',
                       'noun_object', 'noun_food', 'noun_activity', 'noun_place',
                       'preposition', 'conjunction', 'adverb_manner',
                       'weather_condition']
        
        for placeholder in placeholders:
            while f'{{{placeholder}}}' in result:
                word = self.get_random_word(placeholder)
                result = result.replace(f'{{{placeholder}}}', word, 1)
        
        # Second pass: handle articles intelligently
        import re
        article_pattern = r'\{article\}(\s+)(\w+)'
        
        def replace_article(match):
            space = match.group(1)
            next_word = match.group(2)
            article = self.choose_article(next_word)
            return f'{article}{space}{next_word}'
        
        result = re.sub(article_pattern, replace_article, result)
        
        # Clean up any remaining article placeholders
        result = result.replace('{article}', 'a')
        
        # Capitalize first letter
        result = result[0].upper() + result[1:] if result else ""
        
        return result

    def generate_multiple_phrases(self, count=10):
        """Generate multiple unique phrases"""
        phrases = set()
        max_attempts = count * 3  # Prevent infinite loop
        attempts = 0
        
        while len(phrases) < count and attempts < max_attempts:
            phrase = self.generate_phrase()
            phrases.add(phrase)
            attempts += 1
        
        return list(phrases)


class PhraseGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Situation Phrase Generator")
        self.root.geometry("800x600")
        
        self.generator = PhraseGenerator()
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="Human Situation Phrase Generator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Number of phrases
        ttk.Label(controls_frame, text="Number of phrases:").grid(row=0, column=0, padx=(0, 10))
        self.phrase_count = tk.IntVar(value=10)
        count_spinbox = ttk.Spinbox(controls_frame, from_=1, to=50, 
                                   textvariable=self.phrase_count, width=10)
        count_spinbox.grid(row=0, column=1, padx=(0, 20))
        
        # Generate button
        generate_btn = ttk.Button(controls_frame, text="Generate Phrases", 
                                 command=self.generate_phrases)
        generate_btn.grid(row=0, column=2, padx=(0, 10))
        
        # Single phrase button
        single_btn = ttk.Button(controls_frame, text="Generate One", 
                               command=self.generate_single_phrase)
        single_btn.grid(row=0, column=3, padx=(0, 10))
        
        # Save button
        save_btn = ttk.Button(controls_frame, text="Save to File", 
                             command=self.save_phrases)
        save_btn.grid(row=0, column=4, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(controls_frame, text="Clear", 
                              command=self.clear_text)
        clear_btn.grid(row=0, column=5)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Generated Phrases", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        
        # Text area with scrollbar
        self.text_area = scrolledtext.ScrolledText(output_frame, width=80, height=25,
                                                  wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to generate phrases")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Initial generation
        self.generate_phrases()
        
    def generate_phrases(self):
        """Generate multiple phrases and display them"""
        try:
            count = self.phrase_count.get()
            self.status_var.set(f"Generating {count} phrases...")
            self.root.update()
            
            phrases = self.generator.generate_multiple_phrases(count)
            
            self.text_area.delete(1.0, tk.END)
            for i, phrase in enumerate(phrases, 1):
                self.text_area.insert(tk.END, f"{i:2d}. {phrase}\n")
            
            self.status_var.set(f"Generated {len(phrases)} unique phrases")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating phrases: {str(e)}")
            self.status_var.set("Error generating phrases")
    
    def generate_single_phrase(self):
        """Generate and append a single phrase"""
        try:
            phrase = self.generator.generate_phrase()
            
            # Count existing lines
            content = self.text_area.get(1.0, tk.END).strip()
            line_count = len(content.split('\n')) if content else 0
            
            self.text_area.insert(tk.END, f"{line_count + 1:2d}. {phrase}\n")
            self.text_area.see(tk.END)
            
            self.status_var.set(f"Added new phrase: {phrase[:50]}...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating phrase: {str(e)}")
    
    def clear_text(self):
        """Clear the text area"""
        self.text_area.delete(1.0, tk.END)
        self.status_var.set("Text area cleared")
    
    def save_phrases(self):
        """Save generated phrases to a file"""
        try:
            content = self.text_area.get(1.0, tk.END).strip()
            if not content:
                messagebox.showwarning("Warning", "No phrases to save")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_phrases_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Human Situation Phrases\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
            
            messagebox.showinfo("Success", f"Phrases saved to {filename}")
            self.status_var.set(f"Phrases saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = PhraseGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
