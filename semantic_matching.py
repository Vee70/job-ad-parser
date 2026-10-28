import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch 
import numpy as np 

filename = "skillGroups_en"
csv_file_path = f"{filename}.csv"
df = pd.read_csv(csv_file_path)

model_name = 'sentence-transformers/all-MiniLM-L12-v2'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_text(text):
    # ref: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    # handle `nan` input
    if pd.isna(text): return
    # text tokenization
    tokenized_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    # get text embedding
    with torch.no_grad():
        model_output = model(**tokenized_text)

    ##### convert the output into sentence embedding
    # token_embeddings shape: [num_sentences, num_tokens, hidden_state_dim]
    token_embeddings = model_output['last_hidden_state']
    # add extra dim and expand it to match token_embedding shape
    attention_mask_exp = (tokenized_text['attention_mask']
                            .unsqueeze(-1)
                            .expand(token_embeddings.shape)
                            .float())
    # multiply the embeddings matrix by attention mask to exclude padding tokens
    # (multiply by 0 for padding tokens, 1 for the actual tokens)
    masked_embedding = token_embeddings * attention_mask_exp
    # get the vector that represent the input text by
    #  taking the average of each token embedding vector
    #  so the resulting sentence embedding has the shape of
    #  [num_sentence, hidden_state_dim]
    sum_embedding = torch.sum(masked_embedding, dim=1)
    # only consider tokens where attention mask is 1
    #  use min=1e-9 to avoid dividing by 0
    counts = torch.clamp(
        tokenized_text['attention_mask'].sum(dim=1), min=1e-9
    )
    return sum_embedding / counts

def sim_func(x, y_list):
    return pd.Series([torch.nn.functional.cosine_similarity(x, y).item()
                if y is not None else -float('inf') for y in y_list])

# encode all descriptions into text embeddings here so as to match top-k out of these later
embeddings = [encode_text(text) for text in df["description"].values]

def parse_k_nearest(k=3, given_job_title=""):
    # Encode the given sentence
    given_sentence_embedding = encode_text(given_job_title)

    # `embeddings` is a list and cannot call `.tolist()`, so I remove it
    similarities = sim_func(given_sentence_embedding, embeddings)
    # remove duplicate values (keep the one with the smallest index)
    # sort the Series and get indices of the top k values
    return (similarities
            .loc[~similarities.sort_values().duplicated()]
            .sort_values(ascending=False)
            .iloc[:k]
            .index
            .tolist())

def parse_job_title(fname):
    import spacy
    from spacy.matcher import Matcher
    ### The code for this function is taken from `return_title` in part 1
    nlp = spacy.load('en_core_web_sm')

    # load skill ontology file into a DataFrame
    filename = 'skillGroups_en'
    csv_file_path = f'{filename}.csv'

    # read job ads and process the text using `nlp`
    with open(fname, 'r') as f:
        doc_ads = nlp(f.read())

    ### get sentence that starts with "JOB TITLE: "
    matcher_title = Matcher(nlp.vocab)
    pattern_title = [
        {'LOWER': 'job'},
        {'LOWER': 'title'},
        {'IS_PUNCT': True},
        {'IS_SPACE': True, 'OP': '+'},
        {'IS_SPACE': False, 'OP': '+'}
    ]
    # get the longest match
    matcher_title.add('Title', [pattern_title], greedy='LONGEST')
    matches_title = matcher_title(doc_ads)

    job_titles = []
    for _, start, end in matches_title:
        # remove "JOB TITLE: " from the text
        titles = doc_ads[start:end].text.split('JOB TITLE:')[1]
        # splite the title if it contain "|"
        job_titles.extend(
            [title.strip() for title in titles.split('|')]
        )

    ### get match for each title
    df_ontology = pd.read_csv(csv_file_path)
    # get unique skills from ontology
    skills_list = df_ontology['preferredLabel'].tolist()
    
    best_matches = []
    for title in job_titles:
        doc_title = nlp(title)
        # dict to store all matches
        #   key: index of "preferedLabel" in ontology
        #   values: matche output
        # the number of matches is obtained by using len(values)
        matches_dict = {}
        for i, skill in enumerate(skills_list):
            matcher_skill = Matcher(nlp.vocab)
            # add skill specific pattern
            # exclude conjuctions and non alphabetical characters
            pattern_skill = [
                {'LOWER': tok.text.lower(), 'OP': '?'} for tok in nlp(skill)
                    if tok.pos_ != 'CCONJ' and tok.is_alpha
            ]
            matcher_skill.add(f'Skill_{i}', [pattern_skill])
            matches_skill = matcher_skill(doc_title)
            # if there's a match, add to the dict
            if matches_skill and matches_skill not in matches_dict.values():
                matches_dict[i] = matches_skill
        # get all skill indices with largest number of matches
        # if there are more than 1 key (skill) with max number of match
        #  select the one with the shortest len (smallest number of token)
        #  because it less likely to contain irrelevant information
        #   (for example, job title, pucntuation, or other random words)
        max_len = max(len(val) for val in matches_dict.values())
        skill_indices = [idx for idx, val in matches_dict.items()
                             if len(val) == max_len]
        # get all skills with the largest matches
        matches_skill_dict = {}
        for idx in skill_indices:
            matches_skill_dict[idx] = len(nlp.make_doc(skills_list[idx]))
        # select one with the smallest number of tokens
        best_match_idx = min(matches_skill_dict, key=matches_skill_dict.get)
        best_matches.append(skills_list[best_match_idx])

    job_titles = best_matches
    return job_titles
