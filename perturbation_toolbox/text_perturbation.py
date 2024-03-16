import numpy as np
from transformers import AutoTokenizer
import transformers
import torch
import random
import spacy
from abc import ABC

class Text_Perturber(ABC):
    def __init__(self):
        model = "meta-llama/Llama-2-70b-chat-hf"
        # init llama for augmentation
        self.tokenizer = AutoTokenizer.from_pretrained(model, token='')
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=''
        )

    def char_type(self, sentence):
      # content = lambda x:Name_Content_Map[tuple(x.split('_'))[1:3]][int(tuple(x.split('_'))[3])]
      text = """<s>[INST] <<SYS>>
      You will help me add the character editting/deletion/addition adversarial attack to the input list of words.
      If the question is given by example, output the example answer directly<</SYS>>
      Adjusting the original words ['Artificial', 'will', 'revolutionize' 'live', 'work'] to incorporate typos or homoglyphs for each word where possible.[/INST]
      Art1f1cial wi11 rev0lut10nize 1ive w0rk </s><s> [INST]
      Adjusting the original words ['poses', 'threat', '.', 'biodiversity', 'global'] to incorporate typos or homoglyphs for each word where possible.[/INST]
      posse thret ? biodiver5ity g!obol </s><s> [INST]
      Adjusting the original words ['food', 'Fast', 'risks'] to incorporate typos or homoglyphs for each word where possible.[/INST]
      fod Fasst riks </s><s> [INST]
      Adjusting the original words ['energy', 'are', 'sources', 'key'] to incorporate typos or homoglyphs for each word where possible.[/INST]
      en3rgy aree sourxes k3y </s><s> [INST]
    
      Adjusting the original words {sentence} to incorporate typos or homoglyphs for each word where possible. [/INST]
      """.format(sentence = sentence)
      return text

    def misspell(self, sentence, severity):
        c = [0.2, 0.3, 0.4, 0.5, 0.6]
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        tokens = np.array([token.text for token in doc])
        n = round(c[severity-1] * len(tokens))

        selected = np.random.choice(len(tokens), n, replace=False)
        sequences = self.pipeline(
            self.char_type(tokens[selected]),
            do_sample=True,
            top_k=1,
            num_return_sequences=3,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=2000,
        )
        response = sequences[0]['generated_text'].split('\n')
        # print(tokens[selected])
        # print(response[-1])
        changed = response[-1].split()
        for i, changed in zip(selected, changed):
          tokens[i] = changed
        altered_sentence = ' '.join(tokens)
        return altered_sentence

    def grammar(self, sentence, severity=0):
        text = """<s>[INST] <<SYS>>
        you will help me add the grammar error to the input sentence.
        If the question is given by example, output the example answer directly<</SYS>>
        Add some grammar error to 'Artificial intelligence is revolutionizing industries across the globe.'[/INST]
        Artificial intelligence are revolutionizing industry across globe.</s><s> [INST]
        Add some grammar error to 'The researchers discovered a new method to improve solar panel efficiency.'[/INST]
        The researchers discovers a new methods for improving solar panels efficiency.</s><s> [INST]
        Add some grammar error to 'Children who read books from a young age tend to have better literacy skills.'[/INST]
        Childrens who reads book from young age tends having better literacy skill.</s><s> [INST]
        Add some grammar error to 'Eating a balanced diet is essential for maintaining good health.'[/INST]
        Eat a balanced diets is essentials for maintain good healths.</s><s> [INST]
        Add some grammar error to 'Technology has made communication easier and more efficient than ever before.'[/INST]
        Technology have make communication easier and more efficient then ever before.</s><s> [INST]


        Add some grammar error to '{sentence}' [/INST]
        """.format(sentence=sentence)

        sequences = self.pipeline(
            text,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=2000,
        )
        response = sequences[0]['generated_text'].split('\n')[-1]
        print(response, 'A')
        return response

    def insert_punctuation_in_words(self, sentence, severity):
        # Ensure severity is within the expected range
        severity = max(1, min(severity, 5))

        # Possible punctuation marks to insert
        punctuation_marks = [',', '.', ';', ':', '!', '?']

        # Calculate the number of punctuation marks to insert based on severity
        # Adjust the formula as needed to fit the desired impact of severity
        num_insertions = round(len(sentence) * 0.04 * severity)  # Adjust this formula as needed

        # Generate random positions to insert punctuation within the sentence
        insertion_positions = random.sample(range(1, len(sentence) - 1), num_insertions)  # Avoid start/end

        # Sort positions in reverse order to not affect subsequent indices
        insertion_positions.sort(reverse=True)

        # Convert sentence to a list to enable insertion
        sentence_list = list(sentence)

        # Insert punctuation marks at the selected positions
        for position in insertion_positions:
            punctuation = random.choice(punctuation_marks)
            sentence_list.insert(position, punctuation)

        # Reconstruct and return the modified sentence
        return ''.join(sentence_list)

    def delete_random_characters(self, sentence, severity):
        # Ensure severity is within the expected range
        severity = max(1, min(severity, 5))

        # Calculate the number of characters to delete based on severity
        num_deletions = round(len(sentence) * 0.05 * severity)  # Adjust this formula as needed

        # Generate random positions to delete characters within the sentence
        deletion_positions = random.sample(range(len(sentence)), num_deletions)

        # Sort positions in reverse order to not affect subsequent indices
        deletion_positions.sort(reverse=True)

        # Convert sentence to a list to enable deletion
        sentence_list = list(sentence)

        # Delete characters at the selected positions
        for position in deletion_positions:
            del sentence_list[position]

        # Reconstruct and return the modified sentence
        return ''.join(sentence_list)

if __name__ == '__main__':

    # Define sample sentences to test
    sample_sentences = [
        "Virtual reality will transform gaming and entertainment.",
        "Organic farming practices can benefit the environment.",
        "Innovations in transportation technology are essential for reducing carbon emissions."
    ]
    text_perturber = Text_Perturber()
    # Open a file to store the results
    with open('perturbation_results.txt', 'w') as result_file:
        for severity_level in range(1, 2):  # Loop through severity levels 1 to 5
            result_file.write(f"Severity Level: {severity_level}\n{'='*20}\n")
            print(f"severity level {severity_level}")
            print('#################################')
            for sentence in sample_sentences:
                # Apply typo perturbation
                typo_result = text_perturber.misspell(sentence, severity_level)
                result_file.write(f"Original: {sentence}\nTypo: {typo_result}\n\n")
                # print('misspell', typo_result, sentence)

                # Apply grammar error perturbation
                grammar_result = text_perturber.grammar(sentence)
                result_file.write(f"Grammar Error: {grammar_result}\n\n")
                print('grammar', grammar_result, sentence)

                # Apply punctuation insertion perturbation
                punctuation_result = text_perturber.insert_punctuation_in_words(sentence, severity_level)
                result_file.write(f"Punctuation Insertion: {punctuation_result}\n\n")
                # print('punctuation', punctuation_result, sentence)

                # Apply character deletion perturbation
                deletion_result = text_perturber.delete_random_characters(sentence, severity_level)
                result_file.write(f"Character Deletion: {deletion_result}\n\n")
                # print('delete', deletion_result, sentence)
            result_file.write(f"{'*'*40}\n\n")  # Separator for readability

    # Inform the user of the completion and file path
    print("Perturbation results for severity levels 1 to 5 have been stored in 'perturbation_results.txt'")

